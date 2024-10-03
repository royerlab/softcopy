import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts
import yaml

import softcopy
import softcopy.main
import softcopy.slow_write


def run_slow_write(dummy_path: Path, input_path: Path):
    softcopy.slow_write.main([str(dummy_path), str(input_path), "--method", "v2"], standalone_mode=False)


def run_softcopy(targets_file_path: Path):
    softcopy.main.main([str(targets_file_path)], standalone_mode=False)


def create_targets_yaml(targets_file_path: Path, input_path: Path, output_path: Path):
    targets = {
        "targets": [
            {
                "source": str(input_path),
                "destination": str(output_path),
            }
        ]
    }
    with open(targets_file_path, "w") as f:
        f.write(yaml.dump(targets))


def wait_for_file(file_path: Path, timeout: int = 120):
    start_time = time.time()
    while not file_path.exists():
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for {file_path} to be created")
        time.sleep(0.1)


def test_slow_write_and_softcopy(workspace, dummy_zarr_path, create_zarr2_archive):
    input_path = workspace / "input"
    output_path = workspace / "output"
    targets_file_path = workspace / "targets.yaml"

    create_targets_yaml(targets_file_path, input_path, output_path)

    slow_write_process = Process(target=run_slow_write, args=(dummy_zarr_path, input_path))
    slow_write_process.start()

    try:
        wait_for_file(input_path / ".zarray")
    except TimeoutError:
        slow_write_process.terminate()
        pytest.fail("slow_write process did not start in time")

    softcopy_process = Process(target=run_softcopy, args=(targets_file_path,))
    softcopy_process.start()

    slow_write_process.join(timeout=120)
    softcopy_process.join(timeout=120)

    if slow_write_process.is_alive():
        slow_write_process.terminate()
        pytest.fail("slow_write process did not complete in 2 minutes")

    if softcopy_process.is_alive():
        softcopy_process.terminate()
        pytest.fail("softcopy process did not complete in 2 minutes")

    slow_write_output = ts.open({
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(input_path)},
    }).result()

    assert (output_path / ".zarray").exists()
    copied_dataset = ts.open({
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(output_path)},
    }).result()

    assert np.all(np.equal(slow_write_output[:].read().result(), copied_dataset[:].read().result()))