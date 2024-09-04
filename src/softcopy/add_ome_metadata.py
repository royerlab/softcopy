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
import sys
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml
import tensorstore as ts
import click
import numpy as np

from .zarr_copier import ZarrCopier

# scope just for today: convert zarr 2 archives to ome zarr
# dimension separator is '.' for this dataset

@click.command()
@click.argument("source", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.argument("daxi-metadata", type=click.File(mode="r"), required=False)
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

    # integrity check:
    for chunk in itertools.product(*[range(n) for n in files_nd]):
        chunk_file = source / ".".join(map(str, chunk))
        if not chunk_file.exists():
            print(f"Missing chunk file {chunk_file}")
            exit(1)
    
    # Ensure we have all the metadata that we'll need:
    if daxi_metadata:
        dm = yaml.safe_load(daxi_metadata)
        # print(dm)
    
    ome_metadata = {
        "history": [
            " ".join(sys.argv)
        ],
        "multiscales": [
            {
                "axes": [
                    {
                        "name": "T",
                        "type": "time",
                        "unit": "second"
                    },
                    {
                        "name": "C",
                        "type": "channel"
                    },
                    {
                        "name": "Z",
                        "type": "space",
                        "unit": "micrometer"
                    },
                    {
                        "name": "Y",
                        "type": "space",
                        "unit": "micrometer"
                    },
                    {
                        "name": "X",
                        "type": "space",
                        "unit": "micrometer"
                    }
                ],
                "coordinateTransformations": [
                    {
                        "type": "identity"
                    }
                ],
                "datasets": [
                    {
                        "coordinateTransformations": [
                            {
                                "scale": [
                                    1.0,
                                    1.0,
                                    dm["Z step size (um)"] if daxi_metadata else 1.24,
                                    0.439,
                                    0.439
                                ],
                                "type": "scale"
                            }
                        ],
                        "path": "0"
                    }
                ],
                "name": "0",
                "version": "0.4"
            }
        ],
        "omero": {
            "channels": [
                {
                    "active": True,
                    "coefficient": 1.0,
                    "color": "FFFFFF",
                    "family": "linear",
                    "inverted": False,
                    "label": "v0_c488",
                    "window": {
                        "end": 65535.0,
                        "max": 65535.0,
                        "min": 0.0,
                        "start": 0.0
                    }
                },
                {
                    "active": True,
                    "coefficient": 1.0,
                    "color": "FFFFFF",
                    "family": "linear",
                    "inverted": False,
                    "label": "v1_c488",
                    "window": {
                        "end": 65535.0,
                        "max": 65535.0,
                        "min": 0.0,
                        "start": 0.0
                    }
                }
            ],
            "id": 0,
            "name": "",
            "rdefs": {
                "defaultT": 0,
                "defaultZ": 0,
                "model": "color",
                "projection": "normal"
            },
            "version": "0.4",
        },
        "daxi": {
            "version": "0.0"
        }
    }

    if daxi_metadata:
        key_pattern = re.compile("T_(\d+).V_(\d+)")
        tz = ZoneInfo('America/Los_Angeles')
        time_windows = {}
        # Collect all of the timestamps and append them to the ome-zarr metadata:
        for (key, value) in dm.items():
            match = key_pattern.match(key)
            if match:
                timepoint = int(match.group(1))
                view = int(match.group(2))
                start_dt, end_dt = [datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz) for date_str in (value["start"], value["end"])]
                if timepoint not in time_windows:
                    time_windows[timepoint] = {}
                
                time_windows[timepoint][view] = {
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat()
                }

        ome_metadata["daxi"]["timing_detail"] = time_windows

    zattrs = source / ".zattrs"
    if zattrs.exists():
        print(".zattrs file exists - aborting to avoid corruption.")
        exit(1)
    
    with open(zattrs, "w") as zattrs_file:
        zattrs_file.write(
            json.dumps(ome_metadata, indent=4)
        )
    print(ome_metadata)
    exit(1)
    
    # Create the folder for this multiscale:
    (source / "0").mkdir(exist_ok=True)

    # Execute the transformation:
    # Move the .zarray file:
    print(source / ".zarray", source / "0" / ".zarray")

    for chunk in itertools.product(*[range(n) for n in files_nd]):
        filename = ".".join(map(str, chunk))
        chunk_file = source / filename
        print(chunk_file, source / "0" / filename)

        # Sheng's code uses a format given by Time x View x Color x Z x Y x X
        # The acquisitions we use are only ever single color at the moment so we can probably just ignore
        # that key
        # move(chunk_file, source / "0" / Path(*map(str, chunk)))
        # parts = chunk[:2] + chunk[3:]
        # print(chunk_file, source / "0" / Path(*map(str, parts)))

def create_ome_zarr_folder_structure(destination, files_nd):
    for coord in itertools.product(*[range(n) for n in files_nd[:-1]]):
        terminal_folder = destination / "0" / Path(*map(str, coord))
        terminal_folder.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    main()