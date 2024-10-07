import hashlib
import logging
import shutil
from pathlib import Path
from typing import ClassVar

from .copier import AbstractCopier
from .zarr_copier import ZarrCopier

LOG = logging.getLogger(__name__)


class OMEZarrCopier(AbstractCopier):
    """
    Wrapper around a ZarrCopier that also copies the metadata files for an OME-Zarr archive.
    """

    # Files in the ome zarr archive that, if present, should be copied to the destination.
    # Importantly, note that these files are all small - we assume they can be loaded into memory.
    METADATA_FILES: ClassVar[list] = [
        ".zattrs",
        ".zgroup",
        ".zarray",
        "zarr.json",
        "daxi.json",
    ]

    _zarr_copier: ZarrCopier
    _metadata_hashes: dict[str, str]

    def __init__(self, source: Path, destination: Path, n_copy_procs: int, log: logging.Logger = LOG):
        super().__init__(source, destination, n_copy_procs, log)
        image_0_source = source / "0"
        image_0_destination = destination / "0"
        self._zarr_copier = ZarrCopier(image_0_source, image_0_destination, n_copy_procs, log)
        self._metadata_hashes = {}

    def start(self):
        # Before starting, we make the parent directory that the image will be copied into.
        self._destination.mkdir(parents=True, exist_ok=True)
        self._zarr_copier.start()

    def join(self):
        self._zarr_copier.join()
        self.copy_metadata_files()

    def stop(self):
        self._zarr_copier.stop()
        self.copy_metadata_files()

    def copy_metadata_files(self):
        for metadata_file in self.METADATA_FILES:
            source_file = self._source / metadata_file
            destination_file = self._destination / metadata_file
            if source_file.exists():
                file_hash = hashlib.md5(source_file.read_bytes()).hexdigest()  # noqa: S324 (this hash is not cryptographically relevant)

                if metadata_file in self._metadata_hashes and self._metadata_hashes[metadata_file] == file_hash:
                    self._log.debug(f"Metadata file {source_file} (hash {file_hash}) is unchanged, skipping")
                    continue

                self._log.debug(f"Copying metadata file {source_file} (hash {file_hash}) to {destination_file}")

                # The line below is not needed in this specific use case but it's forseeable that the directory might
                # not exist for certain files... leaving this out for now.
                # destination_file.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(source_file, destination_file)
            else:
                self._log.debug(f"Metadata file {source_file} does not exist, skipping")
