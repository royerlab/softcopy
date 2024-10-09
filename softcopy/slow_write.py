"""
Uses tensorstore to write chunks to a zarr archive (either zarr 3 or zarr 2) very slowly, one stack at a time, with a sleep
between each write. This is useful for testing the ability of the main program to read from the archive while it is being
written to.
"""

import sys
import time
from pathlib import Path

import click
import numpy as np
import psutil
import tensorstore as ts
from psutil import AccessDenied

from . import zarr_utils


@click.command()
@click.argument("source", type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path))
@click.argument("destination", type=click.Path(exists=False, file_okay=True, dir_okay=True))
@click.option("--method", type=click.Choice(["v2", "v3", "v3_shard"]), default="v3")
@click.option("--sleep", type=float, default=1.0)
@click.option("--timepoints", type=int, default=3)
@click.option(
    "--no-complete-file",
    is_flag=True,
    help="Disable using a file called 'complete' to signal that the write is done",
    default=False,
)
def main(source, destination, method, sleep, timepoints, no_complete_file):
    ensure_high_io_priority()

    if source.is_file():
        if source.suffix == ".raw":
            data: np.ndarray = np.fromfile(source, dtype=np.uint16)
            data = data.reshape((1, 1, 1, *data.shape))
        else:
            raise ValueError(f"Unsupported file type {source.suffix}")
    else:
        zarr_format = zarr_utils.identify_zarr_format(source)
        if zarr_format is None:
            raise ValueError(f"Could not find zarr metadata file in archive folder {source}")

        dataset = ts.open({
            "driver": "zarr" if zarr_format == 2 else "zarr3",
            "kvstore": {"driver": "file", "path": str(source)},
        }).result()

        # Load the data into memory
        data: np.ndarray = dataset.read().result()
    print(f"Data loaded into memory ({data.nbytes / 1e9:.2f}GB)")

    if data.size == 0:
        raise ValueError("Input data is empty")

    # If the data is 4 dimensional infer (CZYX), expand it to 5 dimensions (TCZYX). We use 5D for ome zarr
    # compatibility
    if data.ndim == 4:
        data = np.expand_dims(data, axis=0)

    if data.ndim != 5:
        raise ValueError("Input data must be 4 or 5 dimensional.")

    # We want a good number of chunks so that we can test the ability of softcopy to copy data.
    # Let's make `1000 * timepoints`` files, if we can, by starting with the files_nd and computing
    # chunks - note that we ignore the first dimension because we do weird modular stuff, but know
    # that we will guarantee that target_files_nd[0] == timepoints later on in the way we write
    target_files_nd = np.array([1, 10, 10, 10])
    # files_nd = shape // chunks
    # chunks = shape // files_nd
    chunks = np.ones_like(data.shape, dtype=np.uint32)
    # We know that data.shape[i] > 0 for all i because data.size > 0 from earlier. Thus data.shape[i] / target_files_nd[i] > 0,
    # so ceil will always yield a nonzero positive integer. Thus we always have a valid chunk size.
    chunks[1:] = np.ceil(data.shape[1:] / target_files_nd)
    # Let's compute the actual number of files we will write - if the number of files is radically less than our
    # target of 1000 * timepoints, we will warn the user:
    num_files = np.prod(data.shape[1:] // chunks[1:])
    if num_files / np.prod(target_files_nd) < 0.1:
        print(
            f"Warning: the number of files being written is very low ({num_files} / timepoint). In real acquisitions, softcopy moves much more data than this, and so this may not be a good test of its performance."
        )

    preparation_methods = {"v2": prepare_zarr_v2, "v3": prepare_zarr_v3, "v3_shard": prepare_zarr_v3_shard}

    prepare_zarr = preparation_methods[method]
    dataset = prepare_zarr(destination, data, timepoints, chunks)

    for t in range(timepoints):
        print()
        print(f"Writing stack {t + 1}/{timepoints}")
        now = time.time()
        dataset[t].write(data[t % data.shape[0]]).result()
        print(f"Wrote stack {t + 1}/{timepoints} in {time.time() - now:.2f}s")
        time.sleep(sleep)

    if no_complete_file:
        return

    # touch an empty file called "complete" to signal that the write is done to other processes
    print("writing complete file")
    (Path(destination) / "complete").touch()


def prepare_zarr_v2(destination, data, timepoints, chunks):
    # Write the data to the destination, one stack at a time
    dataset = ts.open({
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": destination,
        },
        "metadata": {
            "compressor": {
                "id": "blosc",
                "cname": "blosclz",
                "shuffle": 1,
                "clevel": 3,
            },
            "dtype": zarr_utils.dtype_string_zarr2(data.dtype),
            "shape": [timepoints, *data.shape[1:]],
            "chunks": chunks,
        },
        "create": True,
        "delete_existing": True,
    }).result()

    return dataset


def prepare_zarr_v3(destination, data, timepoints, chunks):
    codecs = ts.CodecSpec({
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {
                "name": "blosc",
                "configuration": {
                    "cname": "blosclz",
                    "shuffle": "shuffle",
                    "clevel": 3,
                },
            },
        ],
        "driver": "zarr3",
    })

    dataset = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": destination,
            },
            "metadata": {
                "shape": [timepoints, *data.shape[1:]],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": chunks},
                },
                "chunk_key_encoding": {"name": "default"},
                "data_type": data.dtype.name,
            },
            "create": True,
            "delete_existing": True,
        },
        codec=codecs,
    ).result()

    return dataset


def prepare_zarr_v3_shard(destination, data, timepoints, chunks):
    codecs = ts.CodecSpec({
        "driver": "zarr3",
        "codecs": [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": 2 * chunks,
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {
                            "name": "blosc",
                            "configuration": {
                                "cname": "blosclz",
                                "shuffle": "shuffle",
                                "clevel": 3,
                            },
                        },
                    ],
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }
        ],
    })

    dataset = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": destination,
            },
            "metadata": {
                "shape": [timepoints, *data.shape[1:]],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": chunks},
                },
                "chunk_key_encoding": {"name": "default"},
                "data_type": data.dtype.name,
            },
            "create": True,
            "delete_existing": True,
        },
        codec=codecs,
    ).result()

    return dataset


def ensure_high_io_priority():
    try:
        if sys.platform == "linux":
            # On linux, 0 is the lowest niceness => highest io priority. RT is the highest priority
            # class
            psutil.Process().ionice(psutil.IOPRIO_CLASS_RT, 0)
        elif sys.platform == "win32":
            # On windows, 2 is "high" io priority
            psutil.Process().ionice(2)
        else:
            print("Cannot set high io priority on this platform")
    except (PermissionError, AccessDenied):
        print("Warning: Could not set high io priority, you may need to run as root to do this")


if __name__ == "__main__":
    main()
