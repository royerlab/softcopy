import itertools
import json
import logging
import multiprocessing as mp
import os
import time
from datetime import timedelta
from logging import Logger
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from queue import Empty
from shutil import copyfile
from typing import Literal

import numpy as np
from watchdog.events import FileCreatedEvent, FileDeletedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer, ObserverType

from . import zarr_utils
from .copier import AbstractCopier
from .packed_name import PackedName

LOG = logging.getLogger(__name__)

ctx = mp.get_context("spawn")
Process = ctx.Process
Queue = ctx.Queue
Value = ctx.Value


class ZarrCopier(AbstractCopier):
    _source: Path
    _destination: Path
    _queue: Queue
    _observer: ObserverType
    _stop = Value(
        "b", 0
    )  # Can't use a boolean explicitly - this is 0 / 1 for False / True. True if we want to immediately, forcefully terminate the copy!!!
    # TODO: I think observation_finished lives in one process now, so we can use a normal bool if we want to
    _observation_finished: Synchronized = Value(
        "b", 0
    )  # Same as above. True if the observer has finished (even so, the queue may have more files added by the integrity check step)
    _queue_draining: Synchronized = Value(
        "b", 0
    )  # This is set to true as a signal that the queue will never have new items added to it.
    _copy_procs: list[Process]
    _n_copy_procs: int
    _zarr_format: int
    _dimension_separator: Literal[".", "/"]
    _copy_count = Value("i", 0)  # The number of files that have been copied so far

    def __init__(self, source: Path, destination: Path, n_copy_procs: int = 1, log: Logger = LOG):
        super().__init__(source, destination, n_copy_procs, log)
        # self._source = source
        # self._destination = destination
        # self._log = log
        # self._n_copy_procs = nprocs

        self._zarr_format = zarr_utils.identify_zarr_format(source, log)
        if self._zarr_format is None:
            log.critical(f"Could not identify zarr version of source {source}.")
            exit(1)

        zarr_json_path = source / ("zarr.json" if self._zarr_format == 3 else ".zarray")

        # Compute the number of files in the zarr archive using shape and chunk size
        # files_nd = shape / chunk_size (ignores shard chunk size - those are all internal to a file)
        with open(zarr_json_path.expanduser()) as zarr_json_file:
            zarr_json = json.load(zarr_json_file)
            self._zarr_format = zarr_json["zarr_format"]
            shape = np.array(zarr_json["shape"])
            if self._zarr_format == 2:
                chunks = np.array(zarr_json["chunks"])
                self._dimension_separator = zarr_json["dimension_separator"]
                self._log.debug(f"Dimension separator: {self._dimension_separator}")
                if self._dimension_separator in (None, ""):
                    log.critical(
                        f"Could not determine dimension separator from zarr.json file {zarr_json_path}: {self._dimension_separator!r}"
                    )
                    exit(1)
            elif self._zarr_format == 3:
                chunks = np.array(zarr_json["chunk_grid"]["configuration"]["chunk_shape"])

                # This is convoluted but defined by the zarr json spec
                cke = zarr_json["chunk_key_encoding"]
                self._dimension_separator = cke.get("configuration", {}).get("separator")
                if self._dimension_separator is None:
                    if cke["name"] == "default":
                        self._dimension_separator = "/"
                    elif cke["name"] == "v2":
                        self._dimension_separator = "."
                    else:
                        log.critical(f"Unsupported chunk key encoding {cke['name']}")
                        exit(1)

            self._files_nd = np.ceil(shape / chunks).astype(int)
            log.debug(f"Shape: {shape}, chunks: {chunks}, files_nd: {self._files_nd}")
            log.debug(f"Number of files in zarr archive: {np.prod(self._files_nd)}")

        self._queue = Queue()
        self._observer = Observer()
        event_handler = ZarrFileEventHandler(
            self._zarr_format,
            self._dimension_separator,
            self._files_nd,
            self._observation_finished,
            self._queue,
            self._log,
        )
        self._observer.schedule(event_handler, source, recursive=True)
        self._copy_procs = []

    def start(self):
        self._log.debug("Creating zarr folder structure in destination.")
        ZarrCopier.create_zarr_folder_structure(
            self._destination, self._zarr_format, self._files_nd, self._dimension_separator, self._log
        )

        self._log.debug("Starting copy processes.")
        for _ in range(self._n_copy_procs):
            proc = Process(
                target=_copy_worker,
                args=(
                    self._queue,
                    self._stop,
                    self._queue_draining,
                    self._files_nd,
                    self._source,
                    self._destination,
                    self._dimension_separator,
                    self._zarr_format,
                    self._copy_count,
                ),
            )
            proc.start()
            self._copy_procs.append(proc)

        self._log.debug("Starting filesystem observer.")
        # Note: I think the order of these is important. It's possible that if we index first, a file could be created
        # before the watcher starts monitoring:
        self._observer.start()

        self._log.debug("Queueing existing files.")
        is_complete = self._queue_existing_files()
        # If the zarr archive is already complete, we can stop the observer now, safely knowing we didn't miss any files
        if is_complete:
            self._log.info("Copying started with complete zarr archive. Stopping observer.")
            self._observer.stop()
            self._observer.join()
            self._observation_finished.value = 1
        else:
            self._log.info("Copying started with incomplete zarr archive.")
            self._log.info(
                "If you did not expect the write to be incomplete, then this means the source is corrupted, and this copy will never terminate. If you touch a `complete` file in the source directory, then you can successfully copy the corrupted data. Otherwise, ctrl-c to stop the copy."
            )

    def stop(self):
        self._log.debug("Stopping zarr copier and observer. The zarr archive may not be fully copied!")
        self._stop.value = 1
        for proc in self._copy_procs:
            proc.terminate()
            proc.join()
            proc.close()

        if self._observer.is_alive():
            self._observer.stop()
            self._observer.join()

    def join(self):
        time_start = time.time()
        estimated_files = np.prod(self._files_nd)

        def print_copy_status():
            elapsed = str(timedelta(seconds=time.time() - time_start))
            self._log.info(
                f"Files copied: {self._copy_count.value}/{estimated_files} ({100 * self._copy_count.value / estimated_files:.1f}%) [{elapsed.split('.')[0]}]"
            )

        if self._observation_finished.value == 0:
            self._log.debug("Joining copier - waiting for observation to finish.")
            while self._observation_finished.value == 0 and self._stop.value == 0:
                time.sleep(2)
                print_copy_status()

            self._observer.stop()
            self._observer.join()

        # If the observer was terminated forcefully, stop immediately. In the current implementation this should never
        # be tripped, as stop is usually dispatched by a ctrl-c that would cause the observer loop to raise
        # KeyboardInterrupt - but in theory in a multithreaded program, stop could be called by something else
        # (i.e. a time limit...)
        if self._stop.value == 1:
            return

        # Wait for all the files added by the watcher to be copied
        self._log.debug("Waiting for files to be copied.")
        while not self._queue.empty():
            time.sleep(2)
            print_copy_status()

        self._log.debug("All detected files copied. Checking for missed files.")

        # Perform a final integrity check. The watcher can miss files, so we need to check that all files are copied
        missed_count = 0
        for chunk_index in range(np.prod(self._files_nd)):
            chunk_packed_name: PackedName = PackedName.from_index(chunk_index)
            chunk_path = chunk_packed_name.get_path(
                self._files_nd, self._destination, self._dimension_separator, self._zarr_format
            )

            if not chunk_path.exists():
                self._log.debug(f"File {chunk_path} was missed by the observer! Adding to queue for retry.")
                self._queue.put(chunk_packed_name)
                missed_count += 1

        missed_fraction = missed_count / np.prod(self._files_nd)

        if missed_count == 0:
            self._log.debug("Final integrity check complete. No files were missed.")
        else:
            self._log.debug(
                f"Final integrity check complete. {missed_count} files were missed ({100 * missed_fraction:.1f}%), but will be copied now."
            )
            self._log.debug("Waiting for files to be copied.")

        # We are now done adding files to the queue. This signals to the worker processes that, if the queue is empty,
        # they can safely stop running.
        self._queue_draining.value = 1

        for proc in self._copy_procs:
            proc.join()

        self._log.debug("All files copied. Zarr archive is complete.")

    def _queue_existing_files(self):
        is_complete = False
        chunk_count = 0
        found_lockfiles = False
        for dir_path, _, files in os.walk(self._source):
            for file in files:
                if file.endswith(".__lock"):
                    found_lockfiles = True
                    continue

                # PackedName only accepts paths relative to a source zarr archive, but `dir_path` is an absolute path.
                # We need to convert it to a relative path, then append the file.
                # i.e. we want to go from dir_path = /source/path/to/chunk/, file = chunkname to `path/to/chunk/chunkname`
                # and then pack the name
                relative_dir_path = os.path.relpath(dir_path, self._source)
                relative_filepath = os.path.join(relative_dir_path, file)
                packed_name = PackedName(
                    relative_filepath, self._files_nd, self._dimension_separator, self._zarr_format
                )
                if packed_name.is_zarr_chunk():
                    chunk_count += 1
                if Path(file).stem == "complete":
                    is_complete = True
                self._queue.put(packed_name)

        all_chunks_found = chunk_count == np.prod(self._files_nd)
        write_seems_finished = is_complete or all_chunks_found

        if found_lockfiles and write_seems_finished:
            reasons = []
            if is_complete:
                reasons.append("a 'complete' file is present")

            if all_chunks_found:
                reasons.append("all data chunks are present")

            self._log.warning(
                f"The source zarr archive is likely corrupted: Lockfiles are present, which should only exist in an in-progress write - but this write seems to be finished because {', '.join(reasons)}."
            )

        if is_complete and not all_chunks_found:
            self._log.warning(
                "The source zarr archive is corrupted: A 'complete' file is present, but not all data chunks are present."
            )
            exit(1)

        return write_seems_finished

    @staticmethod
    def create_nd_nested_folders(root_path, files_nd, log: Logger = LOG):
        for coord in itertools.product(*[range(n) for n in files_nd[:-1]]):
            terminal_folder = root_path / Path(*map(str, coord))
            terminal_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def create_zarr_folder_structure(
        zarr_archive_location, zarr_format, files_nd, dimension_separator, log: Logger = LOG
    ):
        if zarr_format == 2:
            # . delimited Zarr 2 has no nested folder structure. Just a top level folder full of files :)
            zarr_archive_location.mkdir(parents=True, exist_ok=True)
            log.info(f"Created zarr2 archive folder at {zarr_archive_location}")
            if dimension_separator == "/":
                ZarrCopier.create_nd_nested_folders(zarr_archive_location, files_nd, log)
                log.info(f"Created nested folder structure for /-delimited Zarr 2 archive in {zarr_archive_location}")
        elif zarr_format == 3:
            # Zarr 3 has a nested folder structure where file `c/1/2/3/4` corresponds to the chunk
            # indexed at (1, 2, 3, 4). There is likely a much smarter way to implement this rather
            # than mkdir p on every leaf folder, but this operation should not be happening on a write congested
            # disk so it's probably fine. (we have to be careful about using the source disk - not the dest
            # disk)
            ZarrCopier.create_nd_nested_folders(zarr_archive_location / "c", files_nd, log)
            log.info(f"Created zarr 3 archive skeleton at {zarr_archive_location}")
        else:
            log.critical(f"Unsupported zarr version {zarr_format}")
            exit(1)


def _copy_worker(
    queue: Queue,
    stop: Synchronized,
    queue_draining: Synchronized,
    files_nd: np.ndarray,
    source: Path,
    destination: Path,
    dimension_separator: Literal[".", "/"],
    zarr_format: Literal[2, 3],
    count: Synchronized,
):
    while stop.value == 0:
        try:
            data: PackedName = queue.get(timeout=1)
            srcfile = data.get_path(files_nd, source, dimension_separator, zarr_format)
            destfile = data.get_path(files_nd, destination, dimension_separator, zarr_format)
            print(f"Copying {srcfile} to {destfile}")
            # destfile = PackedName3.get_path(data, files_nd, destination)
            copyfile(srcfile, destfile)

            # Increment the copy count only if this is a data chunk
            if data._index is not None:
                with count.get_lock():
                    count.value += 1
        except Empty:
            # If we didn't get anything from the queue, and the queue is finished being added to, we can stop
            # this copy thread. We only check if the queue is empty to be sure that the timeout wasn't
            # a due to some other issue, although a 1s timeout is extremely unlikely if the queue is nonempty.
            if queue_draining.value == 1 and queue.empty():
                break


class ZarrFileEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        zarr_format: Literal[2, 3],
        dimension_separator: Literal[".", "/"],
        files_nd: np.ndarray,
        observation_finished: Synchronized,
        queue: Queue,
        log: Logger = LOG,
    ):
        super().__init__()
        self.zarr_format = zarr_format
        self._dimension_separator = dimension_separator
        self.files_nd = files_nd
        self.observation_finished = observation_finished
        self._log = log
        self.queue = queue

    def on_created(self, event: FileCreatedEvent):
        if isinstance(event, FileCreatedEvent):  # noqa: SIM102
            # This is probably pointless, but I am worried about path parsing overhead given how many file transactions
            # can occur - so only parse the path if we know the filepath ends with "complete"
            if event.src_path.endswith("complete"):
                if Path(event.src_path).stem == "complete":
                    self._log.info("Detected 'complete' file. Stopping observer.")
                    with self.observation_finished.get_lock():
                        self.observation_finished.value = 1
                else:
                    self._log.warning(
                        f"File {event.src_path} ends with 'complete' but is not a 'complete' file. - has stem {Path(event.src_path).stem}"
                    )

    def on_deleted(self, event):
        if isinstance(event, FileDeletedEvent):
            # remove .__lock suffix from right side of path
            lock_index = event.src_path.rfind(".__lock")
            if lock_index != -1:
                packed_name = PackedName(
                    event.src_path[:lock_index], self.files_nd, self._dimension_separator, self.zarr_format
                )
                if packed_name._index is None:
                    print(f"screwed up: {event.src_path}")
                self.queue.put(packed_name)

    def on_moved(self, event: FileMovedEvent):
        if isinstance(event, FileMovedEvent):
            src = Path(event.src_path)
            if src.suffix == ".__lock":
                packed_name = PackedName(event.dest_path, self.files_nd, self._dimension_separator, self.zarr_format)
                if packed_name._index is None:
                    print(f"screwed up: {event.src_path} -> {event.dest_path}")
                self.queue.put(packed_name)
