from pathlib import Path
from multiprocessing import Process, Queue, Value
from queue import Empty
from threading import Condition
import logging
from logging import Logger
import itertools
import json
from typing import Optional
import time
import os
from shutil import copyfile
from datetime import timedelta

from watchdog.events import FileCreatedEvent, FileMovedEvent, FileDeletedEvent, FileSystemEventHandler
from watchdog.observers import Observer
import numpy as np

LOG = logging.getLogger(__name__)

class ZarrCopier:
    _source: Path
    _destination: Path
    _queue: Queue
    _observer: Observer
    _stop = Value('b', 0) # Can't use a boolean explicitly - this is 0 / 1 for False / True. True if we want to immediately, forcefully terminate the copy!!!
    _observation_finished: bool = Value('b', 0) # Same as above. True if the queue will never have more items added to it
    _copy_procs: list[Process]
    _n_copy_procs: int
    _zarr_version: int
    _copy_count = Value('i', 0) # The number of files that have been copied so far

    def __init__(self, source: Path, destination: Path, nprocs: int = 1, log: Logger = LOG):
        self._source = source
        self._destination = destination
        self._log = log
        
        zarr_json_path = ZarrCopier.find_zarr_json(source, log)
        
        # Compute the number of files in the zarr archive using shape and chunk size
        # files_nd = shape / chunk_size (ignores shard chunk size - those are all internal to a file)
        with open(zarr_json_path.expanduser()) as zarr_json_file:
            zarr_json = json.load(zarr_json_file)
            self._zarr_version = zarr_json['zarr_format']
            shape = np.array(zarr_json['shape'])
            if self._zarr_version == 2:
                chunks = np.array(zarr_json['chunks'])
            elif self._zarr_version == 3:
                chunks = np.array(zarr_json['chunk_grid']['configuration']['chunk_shape'])
            
            self._files_nd = np.ceil(shape / chunks).astype(int)
            log.debug(f"Shape: {shape}, chunks: {chunks}, files_nd: {self._files_nd}")
            log.debug(f"Number of files in zarr archive: {np.prod(self._files_nd)}")

        self._queue = Queue()
        self._observer = Observer()
        event_handler = ZarrFileEventHandler(self._zarr_version, self._files_nd, self._observation_finished, self._queue, self._log)
        self._observer.schedule(event_handler, source, recursive=True)
        self._n_copy_procs = nprocs
        self._copy_procs = []
    
    def start(self):
        self._log.debug("Creating zarr folder structure in destination.")
        ZarrCopier.create_zarr_folder_structure(self._destination, self._zarr_version, self._files_nd, self._log)

        self._log.debug("Starting copy processes.")
        for _ in range(self._n_copy_procs):
            proc = Process(target=_copy_worker, args=(self._queue, self._stop, self._observation_finished, self._files_nd, self._source, self._destination, self._copy_count))
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
            self._log.info("If you did not expect the write to be incomplete, then this means the source is corrupted, and this copy will never terminate. If you touch a `complete` file in the source directory, then you can successfully copy the corrupted data. Otherwise, ctrl-c to stop the copy.")
    
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
            self._log.info(f"Files copied: {self._copy_count.value}/{estimated_files} ({100 * self._copy_count.value / estimated_files:.1f}%) [{elapsed.split('.')[0]}]")

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
            if self._zarr_version == 2:
                filename = ".".join(map(str, np.unravel_index(chunk_index, self._files_nd)))
                if not (self._destination / filename).exists():
                    self._queue.put(pack_name(filename, self._zarr_version, self._files_nd))
                    self._log.debug(f"File {filename} was missed by the observer! Adding to queue for retry.")
                    missed_count += 1
            elif self._zarr_version == 3:
                chunk_index_nd = np.unravel_index(chunk_index, self._files_nd)
                filename = Path("c") / Path(*map(str, chunk_index_nd))
                if not (self._destination / filename).exists():
                    self._queue.put(pack_name(filename, self._zarr_version, self._files_nd))
                    self._log.debug(f"File {filename} was missed by the observer! Adding to queue for retry.")
                    missed_count += 1
        missed_fraction = missed_count / np.prod(self._files_nd)
        
        if missed_count == 0:
            self._log.debug("Final integrity check complete. No files were missed.")
        else:
            self._log.debug(f"Final integrity check complete. {missed_count} files were missed ({100 * missed_fraction:.1f}%), but will be copied now.")
            self._log.debug("Waiting for files to be copied.")

        for proc in self._copy_procs:
            proc.join()
        
        self._log.debug("All files copied. Zarr archive is complete.")
    
    def _queue_existing_files(self):
        is_complete = False
        chunk_count = 0
        found_lockfiles = False
        for dir, _, files in os.walk(self._source):
            for file in files:
                if file.endswith(".__lock"):
                    found_lockfiles = True
                    continue
                rel_dir = os.path.relpath(dir, self._source)
                packed_name = pack_name(os.path.join(rel_dir, file), self._zarr_version, self._files_nd)
                if packed_name._index is not None:
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
            
            self._log.warning(f"The source zarr archive is likely corrupted: Lockfiles are present, which should only exist in an in-progress write - but this write seems to be finished because {', '.join(reasons)}.")
        
        if is_complete and not all_chunks_found:
            self._log.warning("The source zarr archive is corrupted: A 'complete' file is present, but not all data chunks are present.")
            exit(1)

        return write_seems_finished
    
    def find_zarr_json(archive_path: Path, log: Logger = LOG):
        # Find the zarr json file in the archive folder:
        zarr_json_path = None
        for possible_zarr_file in ("zarr.json", ".zarray"):
            candidate = archive_path / possible_zarr_file
            log.debug(f"Checking for zarr metadata file at {candidate}")
            if candidate.exists():
                log.debug(f"Found zarr metadata file at {candidate}")
                zarr_json_path = candidate
                break
        
        if zarr_json_path is None:
            log.critical(f"Could not find zarr metadata file in archive folder {archive_path}")
            exit(1)
        
        return zarr_json_path

    def create_zarr_folder_structure(zarr_archive_location, zarr_version, files_nd, log: Logger = LOG):
        if zarr_version == 2:
            # Zarr 2 has no nested folder structure. Just a top level folder full of files :)
            zarr_archive_location.mkdir(parents=True, exist_ok=True)
            log.info(f"Created zarr archive folder at {zarr_archive_location}")
        elif zarr_version == 3:
            # Zarr 3 has a nested folder structure where file `c/1/2/3/4` corresponds to the chunk
            # indexed at (1, 2, 3, 4). There is likely a much smarter way to implement this rather
            # than mkdir p on every leaf folder, but this operation should not be happening on a write congested
            # disk so it's probably fine. (we have to be careful about using the source disk - not the dest
            # disk)
            for coord in itertools.product(*[range(n) for n in files_nd[:-1]]):
                terminal_folder = zarr_archive_location / "c" / Path(*map(str, coord))
                terminal_folder.mkdir(parents=True, exist_ok=True)
            log.info(f"Created zarr 3 archive skeleton at {zarr_archive_location}")
        else:
            log.critical(f"Unsupported zarr version {zarr_version}")
            exit(1)

def _copy_worker(queue: Queue, stop: Value, observation_finished: Value, files_nd: np.ndarray, source: Path, destination: Path, count: Value):
    while stop.value == 0:
        try:
            data = queue.get(timeout=1)
            srcfile = data.get_path(files_nd, source)
            destfile = data.get_path(files_nd, destination)
            # destfile = PackedName3.get_path(data, files_nd, destination)
            copyfile(srcfile, destfile)

            # Increment the copy count only if this is a data chunk
            if data._index is not None:
                with count.get_lock():
                    count.value += 1
        except Empty:
            # If we didn't get anything from the queue, and the observation is finished, we can stop
            # this copy thread. We only check if the queue is empty to be sure that the timeout wasn't
            # a due to some other issue, although a 1s timeout is extremely unlikely if the queue is nonempty.
            if observation_finished.value == 1 and queue.empty():
                break

def pack_name(name: str, zarr_version: int, files_nd: np.ndarray):
    if zarr_version == 2:
        return PackedName2(name, files_nd)
    else:
        return PackedName3(name, files_nd)

class PackedName2:
    """
    A compressed path data type for data sets where most files are zarr 2 chunks. Rather than storing the long
    filepath to the chunk, a single ravelled integer index is stored when possible.
    """
    
    _path: Optional[Path] = None
    _index: Optional[int] = None

    def __init__(self, name: str, files_nd: np.ndarray):
        path = Path(name)
        parts = path.name.split(".")
        if len(parts) == files_nd.size:
            try:
                chunk_index_nd = list(map(int, parts))
                self._index = np.ravel_multi_index(chunk_index_nd, files_nd)
                return
            except ValueError:
                pass
        
        self._path = path
    
    def get_path(self, files_nd: np.ndarray, zarr_location: Path):
        if self._path is not None:
            return zarr_location / self._path
        elif self._index is not None:
            chunk_index_nd = np.unravel_index(self._index, files_nd)
            return zarr_location / ".".join(map(str, chunk_index_nd))

class PackedName3:
    """
    A compressed path data type for data sets where most files are zarr 3 chunks. Rather than storing the long
    filepath to the chunk, a single ravelled integer index is stored when possible.
    """

    _path: Optional[Path] = None
    _index: Optional[int] = None

    # IMPORTANT: `name` should be the name relative to the source directory. i.e. if the source is
    # /path/to/source, and you are packing the name /path/to/source/c/0/0/0, you should pass "c/0/0/0".
    # otherwise, this will break.
    def __init__(self, name: str, files_nd: np.ndarray):
        self._path = Path(name)
        parts = self._path.parts

        # If this is a chunk, it must have path parts that end like c, number, number, ..., number
        # if there ar fewer parts than expected, short circuit:
        needed_parts = 1 + files_nd.size
        if len(parts) < needed_parts:
            return
        
        # Crop the parts to the last needed_parts:
        parts = parts[-needed_parts:]

        # We have the right number of path components. Verify that the first is 'c':
        if parts[0] != 'c':
            return
        
        # Verify that all the other parts are integers:
        try:
            chunk_index_nd = list(map(int, parts[-files_nd.size:]))
        except ValueError:
            return

        # Bounds check the chunk index:
        if not all(0 <= coord < files_nd[i] for i, coord in enumerate(chunk_index_nd)):
            return
        
        # We made it! :) We don't keep the path, we just compress it to an index
        self._index = np.ravel_multi_index(chunk_index_nd, files_nd)
        self._path = None
    
    def get_path(self, files_nd: np.ndarray, zarr_location: Path):
        if self._path is not None:
            return zarr_location / self._path
        elif self._index is not None:
            chunk_index_nd = np.unravel_index(self._index, files_nd)
            return zarr_location / "c" / Path(*map(str, chunk_index_nd))

class ZarrFileEventHandler(FileSystemEventHandler):
    def __init__(self, zarr_version: int, files_nd: np.ndarray, observation_finished: Value, queue: Queue, log: Logger = LOG):
        super().__init__()
        self.zarr_version = zarr_version
        self.files_nd = files_nd
        self.observation_finished = observation_finished
        self._log = log
        self.queue = queue

    def on_created(self, event: FileCreatedEvent):
        if isinstance(event, FileCreatedEvent):
            # This is probably pointless, but I am worried about path parsing overhead given how many file transactions
            # can occur - so only parse the path if we know the filepath ends with "complete"
            if event.src_path.endswith("complete"):
                if Path(event.src_path).stem == "complete":
                    self._log.info("Detected 'complete' file. Stopping observer.")
                    with self.observation_finished.get_lock():
                        self.observation_finished.value = 1
    
    def on_deleted(self, event):
        if isinstance(event, FileDeletedEvent):
            # remove .__lock suffix from right side of path
            lock_index = event.src_path.rfind(".__lock")
            if lock_index != -1:
                packed_name = pack_name(event.src_path[:lock_index], self.zarr_version, self.files_nd)
                if packed_name._index is None:
                    print(f"screwed up: {event.src_path}")
                self.queue.put(packed_name)

    def on_moved(self, event: FileMovedEvent):
        if isinstance(event, FileMovedEvent):
            src = Path(event.src_path)
            if src.suffix == ".__lock":
                packed_name = pack_name(event.dest_path, self.zarr_version, self.files_nd)
                if packed_name._index is None:
                    print(f"screwed up: {event.src_path} -> {event.dest_path}")
                self.queue.put(packed_name)