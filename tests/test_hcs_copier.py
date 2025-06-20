import multiprocessing as mp
import random
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Event, Thread

import iohub
import iohub.ngff
import numpy as np
import pytest
import tensorstore as ts
from test_softcopy import run_softcopy


def slow_write(target: Path):
    shape = [3, 3, 100, 100, 100]
    z_chunk = 20
    archive = ts.open({
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": str(target),
        },
        "metadata": {
            "compressor": {
                "id": "blosc",
                "cname": "blosclz",
                "shuffle": 1,
                "clevel": 3,
            },
            "dtype": "<u2",
            "shape": shape,
            "chunks": [1, 1, z_chunk, 30, 40],
            "dimension_separator": "/",
        },
        "create": True,
        "delete_existing": True,
    }).result()

    time.sleep(random.uniform(1, 2))  # noqa: S311

    random_slab = np.random.randint(0, 65536, size=(z_chunk, *shape[-2:]), dtype="<u2")
    for t in range(shape[0]):
        for v in range(shape[1]):
            for z_index in range(0, shape[2], z_chunk):
                archive[t, v, z_index : z_index + z_chunk, ...].write(random_slab).result()
                time.sleep(random.uniform(0.1, 0.5))  # noqa: S311


def slow_write_hcs(target: Path, start_barrier: Event):
    plate = iohub.open_ome_zarr(
        target,
        "hcs",
        "w",
        channel_names=["a", "b"],
    )

    start_barrier.wait()

    targets: list[Path] = []
    for camera in ("cam1", "cam2"):
        for color in ("488nm", "561nm", "640nm"):
            for pos in ("pos1", "pos2", "pos3"):
                pos: iohub.ngff.Position = plate.create_position(camera, color, pos)

                for image, scale in enumerate([[1] * 5, [1, 1, 0.25, 0.25, 0.25]]):
                    pos._create_image_meta(str(image), [iohub.ngff.TransformationMeta(type="scale", scale=scale)])

                    targets.append(Path(pos.zattrs.store.path) / Path(pos.zattrs.key).parent / str(image))
    plate.wells()

    threads = []
    for target in targets:
        thread = Thread(target=slow_write, args=(target,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def test_hcs_copier():
    root_path = Path(tempfile.mkdtemp())
    plate_path = root_path / "plate.ome.zarr"
    print(plate_path)
    start_barrier = Event()
    thread = Thread(target=slow_write_hcs, args=(plate_path, start_barrier))
    thread.start()
    start_barrier.set()
    time.sleep(5)

    ctx = mp.get_context("spawn")
    softcopy_process = ctx.Process(target=run_softcopy, args=(plate_path, root_path / "copy.ome.zarr"))
    softcopy_process.start()

    # from softcopy.hcs_copier import HCSCopier

    # copier = HCSCopier(plate_path, root_path / "copy.ome.zarr", 4)

    # copier.start()

    thread.join()

    limit = 4
    softcopy_process.join(timeout=limit * 60)

    if softcopy_process.is_alive():
        softcopy_process.terminate()
        pytest.fail(f"softcopy process did not complete in {limit} minutes")

    # copier.stop()
    # copier.join()

    def compare_directories(dir1: Path, dir2: Path):
        result = subprocess.run(  # noqa: S602
            f"diff -qr {dir1} {dir2} | grep -E 'Only in|Files .* differ'", shell=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    # compare_directories(plate_path, root_path / "copy.ome.zarr")
    assert not compare_directories(plate_path, root_path / "copy.ome.zarr"), "Directories differ after copying"


if __name__ == "__main__":
    test_hcs_copier()
