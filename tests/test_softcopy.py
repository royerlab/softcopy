import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts
import yaml

import softcopy
import softcopy.main
import softcopy.slow_write


def run_slow_write(dummy_path: Path, input_path: Path, no_complete_file: bool = False):
    softcopy.slow_write.main(
        [str(dummy_path), str(input_path), "--method", "v2", "--no-complete-file" if no_complete_file else "--"],
        standalone_mode=False,
    )


def run_softcopy(targets_file_path: Path):
    softcopy.main.main(
        [
            str(targets_file_path),
        ],
        standalone_mode=False,
    )


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

    ctx = mp.get_context("spawn")
    slow_write_process = ctx.Process(target=run_slow_write, args=(dummy_zarr_path, input_path))
    # use spawn instead:
    slow_write_process.start()

    try:
        wait_for_file(input_path / ".zarray")
    except TimeoutError:
        slow_write_process.terminate()
        pytest.fail("slow_write process did not start in time")

    softcopy_process = ctx.Process(target=run_softcopy, args=(targets_file_path,))
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


# In the tests above, we just test copying a regular zarr archive. However, we also need limited support for ome-zarr FOV data.
# Specifically, the ome-zarr subset that is written by DaXi. This looks like the following:
# aquisition.ome.zarr/
# - .zgroup # .zgroup and .zattrs are modified only at the acquisition start
# - .zattrs
# - daxi.json # daxi.json is appended to at each timepoint
# - daxi.json.temp # Sometimes exists, doesn't need to be copied
# - 0/ # this is a normal, /-delimited zarr 2 archive.
#   - .zarray
#   - 0/
#     - 0/
#     - ...
# This test is thus almost the same as the previous one, but with a more complicated skeleton. Additionally, as slow_write can only
# output zarr (not ome zarr), we just use slow write to output to the 0/ folder, and then invoke softcopy on the entire .ome.zarr
# folder.
def test_slow_write_and_softcopy_ome_zarr(workspace, dummy_zarr_path, create_zarr2_archive):
    # Construct a dummy acquisition
    input_path = workspace / "input.ome.zarr"
    input_path.mkdir()
    (input_path / ".zgroup").write_text(".zgroup")
    (input_path / ".zattrs").write_text(".zattrs")

    output_path = workspace / "output.ome.zarr"
    targets_file_path = workspace / "targets.yaml"

    create_targets_yaml(targets_file_path, input_path, output_path)

    ctx = mp.get_context("spawn")
    slow_write_process = ctx.Process(target=run_slow_write, args=(dummy_zarr_path, input_path / "0", True))
    slow_write_process.start()

    try:
        wait_for_file(input_path / "0" / ".zarray")
    except TimeoutError:
        slow_write_process.terminate()
        pytest.fail("slow_write process did not start in time")

    # Start softcopy and disable the complete file so that we can tamper with daxi.json
    # before the write is complete
    softcopy_process = ctx.Process(target=run_softcopy, args=(targets_file_path,))
    softcopy_process.start()

    # Wait for all of the data to be done writing
    slow_write_process.join(timeout=120)
    if slow_write_process.is_alive():
        slow_write_process.terminate()
        pytest.fail("slow_write process did not complete in 2 minutes")

    # Modify daxi.json via daxi.json.temp and shutil.move
    daxi_json_temp = input_path / "daxi.json.temp"
    daxi_json = input_path / "daxi.json"
    daxi_json_temp.write_text("daxi.json")
    shutil.move(daxi_json_temp, daxi_json)

    # Create the complete file to signal that the write is done
    (input_path / "0" / "complete").touch()

    softcopy_process.join(timeout=120)

    if softcopy_process.is_alive():
        softcopy_process.terminate()
        pytest.fail("softcopy process did not complete in 2 minutes")

    slow_write_output = ts.open({
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(input_path / "0")},
    }).result()

    assert (output_path / ".zgroup").exists() and (output_path / ".zgroup").read_text() == ".zgroup"
    assert (output_path / ".zattrs").exists() and (output_path / ".zattrs").read_text() == ".zattrs"
    assert (output_path / "daxi.json").exists() and (output_path / "daxi.json").read_text() == "daxi.json"
    assert not (output_path / "daxi.json.temp").exists()

    copied_dataset = ts.open({
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(output_path / "0")},
    }).result()

    assert np.all(np.equal(slow_write_output[:].read().result(), copied_dataset[:].read().result()))
