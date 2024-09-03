"""
Uses tensorstore to write chunks to a zarr archive (either zarr 3 or zarr 2) very slowly, one stack at a time, with a sleep
between each write. This is useful for testing the ability of the main program to read from the archive while it is being
written to.
"""

import time
from pathlib import Path
import sys
import json
import time

import psutil
import tensorstore as ts
import click


@click.command()
@click.argument("source", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument(
    "destination", type=click.Path(exists=False, file_okay=True, dir_okay=True)
)
@click.option("--method", type=click.Choice(["v2", "v3", "v3_shard"]), default="3")
@click.option("--sleep", type=float, default=1.0)
@click.option("--timepoints", type=int, default=3)
def main(source, destination, method, sleep, timepoints):
    ensure_high_io_priority()

    # Extract the zarr version - I don't think tensorstore is smart enough to do this :(
    zarr_version = None
    for possible_zarr_file in ("zarr.json", ".zarray"):
        candidate = Path(source) / possible_zarr_file
        if candidate.exists():
            with open(candidate) as zarr_json_file:
                zarr_json = json.load(zarr_json_file)
                zarr_version = zarr_json["zarr_format"]
                break
    if zarr_version is None:
        raise ValueError(
            f"Could not find zarr metadata file in archive folder {source}"
        )

    dataset = ts.open(
        {
            "driver": "zarr" if zarr_version == 2 else "zarr3",
            "kvstore": {"driver": "file", "path": source},
        }
    ).result()

    # Load the data into memory
    data = dataset.read().result()
    print(f"Data loaded into memory ({data.nbytes / 1e9:.2f}GB)")

    if method == "v2":
        dataset = write_v2(destination, data, timepoints)
    elif method == "v3":
        dataset = write_v3(destination, data, timepoints)
    elif method == "v3_shard":
        dataset = write_v3_shard(destination, data, timepoints)
    
    for t in range(timepoints):
        print()
        print(f"Writing stack {t + 1}/{timepoints}")
        now = time.time()
        dataset[t].write(data).result()
        print(f"Wrote stack {t + 1}/{timepoints} in {time.time() - now:.2f}s")
        time.sleep(sleep)
    
    # touch an empty file called "complete" to signal that the write is done to other processes
    print("writing complete file")
    (Path(destination) / "complete").touch()


def write_v2(destination, data, timepoints):
    # Write the data to the destination, one stack at a time
    dataset = ts.open(
        {
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
                "dtype": "<u2",
                "shape": [timepoints, *data.shape],
                "chunks": [1, 600, 600, 600],
            },
            "create": True,
            "delete_existing": True,
        }
    ).result()

    return dataset


def write_v3(destination, data, timepoints):
    codecs = ts.CodecSpec(
        {
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
        }
    )

    dataset = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": destination,
            },
            "metadata": {
                "shape": [timepoints, *data.shape],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 100, 100, 100]},
                },
                "chunk_key_encoding": {"name": "default"},
                "data_type": "uint16",
            },
            "create": True,
            "delete_existing": True,
        },
        codec=codecs,
    ).result()

    return dataset


def write_v3_shard(destination, data, timepoints):
    codecs = ts.CodecSpec(
        {
            "driver": "zarr3",
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": [1, 100, 100, 100],
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
        }
    )

    dataset = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": destination,
            },
            "metadata": {
                "shape": [timepoints, *data.shape],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 200, 600, 600]},
                },
                "chunk_key_encoding": {"name": "default"},
                "data_type": "uint16",
            },
            "create": True,
            "delete_existing": True,
        },
        codec=codecs,
    ).result()

    return dataset


def ensure_high_io_priority():
    if sys.platform == "linux":
        # On linux, 0 is the lowest niceness => highest io priority. RT is the highest priority
        # class
        psutil.Process().ionice(psutil.IOPRIO_CLASS_RT, 0)
    elif sys.platform == "win32":
        # On windows, 2 is "high" io priority
        psutil.Process().ionice(2)
    else:
        print("Cannot set high io priority on this platform")


if __name__ == "__main__":
    main()
