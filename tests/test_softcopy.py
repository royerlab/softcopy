import tempfile
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


# Wrapper functions to allow piclking of the slow_write and softcopy processes
def run_slow_write(dummy_path, input_path):
    softcopy.slow_write.main([dummy_path, input_path], standalone_mode=False)


def run_softcopy(temp_dir):
    softcopy.main.main([temp_dir], standalone_mode=False)


def test_slow_write_and_softcopy():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create random data file for slow_write
        temp_path = Path(temp_dir)
        dummy_path = temp_path / "dummy"  # This is where the garbage data that slow_write reads from is stored
        input_path = temp_path / "input"
        output_path = temp_path / "output"

        # Create a dummy zarr 2 archive
        shape = [100, 50, 50, 100]
        chunks = [1, 10, 10, 100]
        data = (np.random.random(shape) * 2**16).astype(np.uint16)
        dataset = ts.open({
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(dummy_path)},
            "metadata": {"dtype": ">u2", "shape": shape, "chunks": chunks},
            "create": True,
        }).result()
        dataset[:].write(data).result()

        # Create a targets.yaml file in tmpdir
        targets = {
            "targets": [
                {
                    "source": str(input_path),
                    "destination": str(output_path),
                }
            ]
        }
        targets_file_path = temp_path / "targets.yaml"
        with open(targets_file_path, "w") as f:
            f.write(yaml.dump(targets))

        # Start slow_write process
        slow_write_process = Process(
            target=run_slow_write,
            args=(
                str(dummy_path),
                str(input_path),
            ),
        )
        slow_write_process.start()

        # Wait for the slow write to have started
        while not (input_path / "zarr.json").exists():
            time.sleep(0.1)

        # Start softcopy process
        softcopy_process = Process(target=run_softcopy, args=(str(targets_file_path),))
        softcopy_process.start()

        # Wait for processes to complete or timeout after 2 minutes
        timeout = 120
        slow_write_process.join(timeout=timeout)
        softcopy_process.join(timeout=timeout)

        if slow_write_process.is_alive():
            slow_write_process.terminate()
            pytest.fail("slow_write process did not complete in 2 minutes")

        if softcopy_process.is_alive():
            softcopy_process.terminate()
            pytest.fail("softcopy process did not complete in 2 minutes")

        # Add assertions here to verify the expected behavior
        # For example, check if the output files are created correctly
        assert (output_path / "zarr.json").exists()
        result_dataset = ts.open({
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(output_path)},
        }).result()

        # Note: We use np.equal and not np.allclose because we expect exact bit for bit equality
        assert np.all(np.equal(result_dataset[:].read().result(), data))


if __name__ == "__main__":
    pytest.main([__file__])
