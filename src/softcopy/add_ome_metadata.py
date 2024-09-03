# A zarr 3 archive is like this:
# root/
# - zarr.yaml
# - c/
#   - 0/
#     - 0/
#       - 0/
#         - 0
#         - 1
#         - ...
#       - 1/
#         - 0
#         - 1
#         - ...
#     - 1/
#     - ...
#   - 1/
#   - ...
# - ...
#
# An ome-zarr archive is like this:
# root/
# - .zarray
# - .zgroup
# - 0/ (multiscale size 0)
#   - 0/
#     - ...
# - 1/ (multiscale size 1)
# etc
#
# so to convert a zarr 3 archive to an ome-zarr archive, we need to:
# 1. convert the zarr yaml to .zarray and .zgroup
# 2. move the c/ folder to be the 0/ folder
# 3. create downsamples in the 1, 2, etc folders if desired

import json
from pathlib import Path
import itertools
from shutil import move

import tensorstore as ts
import click
import numpy as np

from .zarr_copier import ZarrCopier

# scope just for today: convert zarr 2 archives to ome zarr

@click.command()
@click.argument("source", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.argument("daxi-metadata", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), required=False)
def main(source, daxi_metadata):
    zarr_json_path = ZarrCopier.find_zarr_json(source)
    zarr_version = None

    with open(zarr_json_path.expanduser()) as zarr_json_file:
        zarr_json = json.load(zarr_json_file)
        zarr_version = zarr_json['zarr_format']
        shape = np.array(zarr_json['shape'])

        if zarr_version == 2:
            chunks = np.array(zarr_json['chunks'])
        elif zarr_version == 3:
            chunks = np.array(zarr_json['chunk_grid']['configuration']['chunk_shape'])
        
        files_nd = np.ceil(shape / chunks).astype(int)

    create_ome_zarr_folder_structure(source, files_nd)

    for chunk in itertools.product(*[range(n) for n in files_nd]):
        chunk_file = source / ".".join(map(str, chunk))
        if not chunk_file.exists():
            print(f"Missing chunk file {chunk_file}")
            break
        move(chunk_file, source / "0" / Path(*map(str, chunk)))

def create_ome_zarr_folder_structure(destination, files_nd):
    for coord in itertools.product(*[range(n) for n in files_nd[:-1]]):
        terminal_folder = destination / "0" / Path(*map(str, coord))
        terminal_folder.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    main()