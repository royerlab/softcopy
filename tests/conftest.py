import shutil
import tempfile
from pathlib import Path

import pytest
import numpy as np
import tensorstore as ts

from softcopy import zarr_utils


@pytest.fixture
def workspace():
    """
    Creates a temporary directory to safely test filesystem operations inside of. Deletes the directory
    after the test finishes.
    """
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


@pytest.fixture
def dummy_zarr_path(workspace):
    path = workspace / "dummy.zarr"
    return path


@pytest.fixture(
    params=[
        {"dtype": np.int8},
        {"dtype": np.uint16},
        {"dtype": np.float32},
    ]
)
def create_zarr2_archive(dummy_zarr_path, request):
    dtype = np.dtype(request.param["dtype"])
    dtype_str = zarr_utils.dtype_string_zarr2(dtype)

    shape = [1, 10, 100, 100, 100]
    chunks = [1, 1, 10, 10, 10]

    dataset = ts.open({
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(dummy_zarr_path)},
        "metadata": {"dtype": dtype_str, "shape": shape, "chunks": chunks},
        "create": True,
    }).result()

    scale = np.iinfo(dtype).max if dtype.kind == "i" else 1000
    random_data = (np.random.random(shape) * scale).astype(dtype)
    dataset[:].write(random_data).result()

    return {
        "data": random_data,
        "chunks": chunks,
    }