from pathlib import Path
from multiprocessing import Process, Queue, Value
from queue import Empty
from threading import Condition
import logging
from logging import Logger
import itertools
import json
from typing import Optional
import os
from shutil import copyfile

from watchdog.events import FileCreatedEvent, FileMovedEvent, FileDeletedEvent, FileSystemEventHandler
from watchdog.observers import Observer
import numpy as np

LOG = logging.getLogger(__name__)

class ZarrCopier:
    _source: Path
    _destination: Path
    _queue: Queue
    _observer: Observer
    _cv: Condition
    _stop = Value('b', 0) # Can't use a boolean explicitly - this is 0 / 1 for False / True. True if we want to immediately, forcefully terminate the copy!!!
    _observation_finished: bool = Value('b', 0) # Same as above. True if the queue will never have more items added to it
    _copy_procs: list[Process]
    _n_copy_procs: int

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
        self._cv = Condition()
        self._observer = Observer()
        event_handler = ZarrFileEventHandler(self._zarr_version, self._files_nd, self._cv, self._queue, self._log)
        self._observer.schedule(event_handler, source, recursive=True)
        self._n_copy_procs = nprocs
        self._copy_procs = []
    
    def start(self):
        ZarrCopier.create_zarr_folder_structure(self._destination, 3, self._files_nd, self._log)

        for _ in range(self._n_copy_procs):
            proc = Process(target=_copy_worker, args=(self._queue, self._stop, self._observation_finished, self._files_nd, self._source, self._destination, self._log))
            proc.start()
            self._copy_procs.append(proc)

        # Note: I think the order of these is important. It's possible that if we index first, a file could be created
        # before the watcher starts monitoring:
        self._observer.start()
        is_complete = self._index_existing_files()
        # If the zarr archive is already complete, we can stop the observer now, safely knowing we didn't miss any files
        if is_complete:
            self._observer.stop()
            self._observation_finished.value = 1
    
    def stop(self):
        self._stop.value = 1
        for proc in self._copy_procs:
            proc.terminate()
            proc.join()
            proc.close()
        with self._cv:
            self._cv.notify_all()
            if self._observer.is_alive():
                self._observer.stop()
                self._observer.join()
    
    def join(self):
        if self._observation_finished.value == 0:
            with self._cv:
                self._cv.wait()
            
            self._observer.stop()
            self._observer.join()
            self._observation_finished.value = 1

        for proc in self._copy_procs:
            print("joining")
            proc.join()
            print("joined")
    
    def _index_existing_files(self):
        is_complete = False
        chunk_count = 0
        for dir, _, files in os.walk(self._source):
            for file in files:
                rel_dir = os.path.relpath(dir, self._source)
                packed_name = pack_name(os.path.join(rel_dir, file), self._zarr_version, self._files_nd)
                if packed_name._index is not None:
                    chunk_count += 1
                if Path(file).stem == "complete":
                    is_complete = True
                self._queue.put(packed_name)
        
        return is_complete or chunk_count == self._files_nd.prod()
    
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

def _copy_worker(queue: Queue, stop, observation_finished, files_nd, source, destination, log):
    while stop.value == 0:
        try:
            data = queue.get(timeout=1)
            srcfile = data.get_path(files_nd, source)
            # destfile = data.get_path(files_nd, destination)
            destfile = PackedName3.get_path(data, files_nd, destination)
            copyfile(srcfile, destfile)
        except Empty as e:
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
        print(name)
        self._path = Path(name)
        parts = self._path.parts

        # If this is a chunk, it must have path parts that end like c, number, number, ..., number
        # if there ar fewer parts than expected, short circuit:
        if len(parts) < files_nd.size + 1:
            return
        
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
    def __init__(self, zarr_version: int, files_nd: np.ndarray, cv: Condition, queue: Queue, log: Logger = LOG):
        super().__init__()
        self.zarr_version = zarr_version
        self.files_nd = files_nd
        self._cv = cv
        self._log = log
        self.queue = queue

    def on_created(self, event: FileCreatedEvent):
        if isinstance(event, FileCreatedEvent):
            # This is probably pointless, but I am worried about path parsing overhead given how many file transactions
            # can occur - so only parse the path if we know the filepath ends with "complete"
            if event.src_path.endswith("complete"):
                if Path(event.src_path).stem == "complete":
                    print("complete")
                    with self._cv:
                        self._cv.notify_all()
    
    def on_deleted(self, event):
        if isinstance(event, FileDeletedEvent):
            # remove .__lock suffix from right side of path
            lock_index = event.src_path.rfind(".__lock")
            if lock_index != -1:
                packed_name = pack_name(event.src_path[:lock_index], self.zarr_version, self.files_nd)
                self.queue.put(packed_name)

    def on_moved(self, event: FileMovedEvent):
        if isinstance(event, FileMovedEvent):
            src = Path(event.src_path)
            if src.suffix == ".__lock":
                packed_name = pack_name(event.dest_path, self.zarr_version, self.files_nd)
                self.queue.put(packed_name)