import functools
import json
import logging
import operator
import time
from logging import Logger
from pathlib import Path
from typing import Literal, Optional

LOG = logging.getLogger(__name__)
METADATA_FILES_BY_VERSION = {
    2: [".zarray", ".zattrs", ".zgroup"],
    3: ["zarr.json"],
}
ALL_METADATA_FILES = set(functools.reduce(operator.iadd, METADATA_FILES_BY_VERSION.values(), []))
KNOWN_VERSIONS = set(METADATA_FILES_BY_VERSION.keys())


def identify_zarr_format(archive_path: Path, log: Logger = LOG) -> Optional[Literal[2, 3]]:
    """
    Identify the zarr version of the archive by identifying a metadata file and reading its zarr_format key.
    If the metadata file is missing, the zarr_format key is missing, or the specified version is not "2" or "3",
    returns None.
    """

    for candidate_file in ALL_METADATA_FILES:
        metadata_file = archive_path / candidate_file

        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            zarr_format = metadata.get("zarr_format")
            if zarr_format in KNOWN_VERSIONS:
                log.debug(f"Identified zarr version {zarr_format} from metadata file {metadata_file}")
                return zarr_format
            else:
                log.debug(f"Invalid zarr version {zarr_format} in metadata file {metadata_file}")
                return None

    log.debug(f"Could not identify zarr version from metadata files in archive folder {archive_path}")
    return None


def dtype_string_zarr2(dtype):
    endianness = dtype.byteorder
    if endianness == "=":
        endianness = "<"
    bytesize = dtype.itemsize
    dtype_kind = dtype.kind

    if dtype_kind not in ("i", "f", "u"):
        raise ValueError(f"Unsupported dtype kind: {dtype_kind}")

    dtype_str = f"{endianness}{dtype_kind}{bytesize}"
    return dtype_str


def wait_for_source(source: Path, log: Logger = LOG):
    """
    Wait for the source to be created. This is useful if you want to start softcopy and a writer at similar times and don't know
    how long the writer will take to create the source. This looks for the presence of a metadata file in the source.
    """
    zarr_format = identify_zarr_format(source, log)
    while zarr_format is None:
        time.sleep(1.0)
        zarr_format = identify_zarr_format(source, log)
        log.debug(f"Waiting for source {source} to be created")

    log.debug(f"Source {source} is available")
