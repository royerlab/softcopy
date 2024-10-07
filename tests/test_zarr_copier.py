import json
import shutil
import time

from softcopy.zarr_copier import ZarrCopier

TEST_DATA = "test data"


def test_zarr_copier(workspace, dummy_zarr_path):
    destination = workspace / "destination"
    destination.mkdir()
    dummy_zarr_path.mkdir()

    shape = [100, 100]
    chunks = [10, 10]

    zarr_json = {"zarr_format": 2, "shape": shape, "chunks": chunks, "dimension_separator": "."}
    (dummy_zarr_path / ".zarray").write_text(json.dumps(zarr_json))

    copier = ZarrCopier(dummy_zarr_path, destination, n_copy_procs=1)
    copier.start()

    time.sleep(1)

    # Create a file in the source directory
    chunk_file = dummy_zarr_path / "0.0.__lock"
    chunk_file.write_text(TEST_DATA)

    shutil.move(str(chunk_file), str(dummy_zarr_path / "0.0"))

    # Wait for a few seconds to allow the copier to copy the file
    time.sleep(1)

    # Check if the file was copied successfully
    copied_file = destination / "0.0"
    assert copied_file.exists()
    assert copied_file.read_text() == TEST_DATA

    copier.stop()
    copier.join()
