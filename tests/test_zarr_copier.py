import json
import shutil
import tempfile
import time
from logging import getLogger
from pathlib import Path

import pytest

from softcopy.zarr_copier import ZarrCopier

def create_zarr_2_archive(path: Path, shape, chunks):
    zarr_json = {"zarr_format": 2, "shape": shape, "chunks": chunks}
    (path / ".zarray").write_text(json.dumps(zarr_json))


def test_zarr_copier(workspace):
    source = workspace / "source"
    destination = workspace / "destination"
    source.mkdir()
    destination.mkdir()

    shape = [100, 100]
    chunks = [10, 10]
    create_zarr_2_archive(source, shape, chunks)

    log = getLogger(__name__)
    copier = ZarrCopier(source, destination, nprocs=1, log=log)
    copier.start()

    time.sleep(1)

    # Create a file in the source directory
    chunk_file = source / "0.0.__lock"
    chunk_file.write_text("test data")
    import shutil

    shutil.move(str(chunk_file), str(source / "0.0"))

    # Wait for a few seconds to allow the copier to copy the file
    time.sleep(1)

    # Check if the file was copied successfully
    copied_file = destination / "0.0"
    assert copied_file.exists()
    assert copied_file.read_text() == "test data"

    copier.stop()
    copier.join()
