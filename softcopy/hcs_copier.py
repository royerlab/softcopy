import logging
import shutil
import time
from pathlib import Path
from threading import Event, Thread

import iohub

from .copier import AbstractCopier
from .zarr_copier import ZarrCopier

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s : %(levelname)s from %(name)s] %(message)s")
LOG = logging.getLogger(__name__)


def get_all_zarr_paths(plate_path: Path) -> list[Path]:
    plate: iohub.ngff.Plate = iohub.open_ome_zarr(plate_path)
    wells = plate.wells()
    zarr_paths = []

    for _, well in wells:
        for _, pos in well.positions():
            for _, image in pos.images():
                zarr_paths.append(Path(image._attrs.store.path) / Path(image._attrs.key).parent)

    return zarr_paths


def create_metadata_path_mapping(source: Path, destination: Path) -> list[tuple[Path, Path]]:
    path_mappings = []
    plate: iohub.ngff.Plate = iohub.open_ome_zarr(source)

    for metadata_file in [".zattrs", ".zgroup"]:
        src_metadata_path = source / metadata_file
        dest_metadata_path = destination / metadata_file
        path_mappings.append((src_metadata_path, dest_metadata_path))

    for row, _ in plate.rows():
        src = source / row
        dest = destination / row
        for metadata_file in [".zattrs", ".zgroup"]:
            src_metadata_path = src / metadata_file
            dest_metadata_path = dest / metadata_file
            path_mappings.append((src_metadata_path, dest_metadata_path))

    for well_path_component, well in plate.wells():
        well_src = source / well_path_component
        well_dest = destination / well_path_component
        for metadata_file in [".zattrs", ".zgroup"]:
            src_metadata_path = well_src / metadata_file
            dest_metadata_path = well_dest / metadata_file
            path_mappings.append((src_metadata_path, dest_metadata_path))

        for pos_path_component, _ in well.positions():
            src = well_src / pos_path_component
            dest = well_dest / pos_path_component
            for metadata_file in [".zattrs", ".zgroup"]:
                src_metadata_path = src / metadata_file
                dest_metadata_path = dest / metadata_file
                path_mappings.append((src_metadata_path, dest_metadata_path))

    return path_mappings

def wait_for_metadata(source: Path, log: logging.Logger):
    while True:
        log.info(f"Waiting for metadata files in {source} to be created.")
        if all((source / f).exists() for f in [".zattrs", ".zgroup"]):
            log.info("Metadata files found.")
            break
        time.sleep(1)

class SlowCopier(Thread):
    def __init__(self, path_mappings: list[tuple[Path, Path]]):
        super().__init__()
        self.path_mappings = path_mappings
        self.stop_event = Event()
        self.last_copy_times = {dest: None for _, dest in path_mappings}

    def _copy_pass(self):
        for source, destination in self.path_mappings:
            try:
                if source.exists():
                    source_mtime = source.stat().st_mtime
                    last_copy_time = self.last_copy_times[destination]

                    if last_copy_time is None or source_mtime > last_copy_time:
                        # Ensure the destination directory exists:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)
                        self.last_copy_times[destination] = source.stat().st_mtime
                        LOG.debug(f"Copied {source} to {destination}")
                else:
                    LOG.debug(f"Source file {source} does not exist yet.")
            except Exception as e:
                LOG.exception(f"Error copying {source} to {destination}: {e}")  # noqa: TRY401

    def run(self):
        while not self.stop_event.is_set():
            self._copy_pass()
            self.stop_event.wait(4)  # Sleep for 4 seconds or until stop_event is set

    def join(self):
        self.stop_event.set()
        super().join()  # Wait for the thread to finish
        self._copy_pass()  # Final copy pass to ensure all files are copied


class HCSCopier(AbstractCopier):
    _copiers: list[ZarrCopier]
    _metadata_copier: SlowCopier

    def __init__(
        self,
        source: Path,
        destination: Path,
        n_copy_procs: int,
        sleep_time: float = 0,
        wait_for_source: bool = True,
        log: logging.Logger = LOG,
    ):
        super().__init__(source, destination, n_copy_procs, sleep_time, log)

        if wait_for_source:
            wait_for_metadata(source, log)
            time.sleep(60) # Make sure that the rest of the metadata files are written - we can't check for them
                            # explicitly because they are dynamically defined by the OME-Zarr structure.
        image_paths = get_all_zarr_paths(source)
        print(image_paths)
        self._copiers = []
        for image_path in image_paths:
            image_destination = destination / image_path.relative_to(source)
            print(image_destination)
            image_destination.mkdir(parents=True, exist_ok=True)
            zarr_copier = ZarrCopier(image_path, image_destination, n_copy_procs, sleep_time, wait_for_source, log)
            self._copiers.append(zarr_copier)

        path_map = create_metadata_path_mapping(source, destination)
        self._metadata_copier = SlowCopier(path_map)

    def start(self):
        # Before starting, we make the parent directory that the image will be copied into.
        self._destination.mkdir(parents=True, exist_ok=True)
        for copier in self._copiers:
            copier.start()
        self._metadata_copier.start()

    def join(self):
        for copier in self._copiers:
            copier.join()
        self._metadata_copier.join()

    def stop(self):
        for copier in self._copiers:
            copier.stop()

        self._metadata_copier.join()
