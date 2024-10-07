# This was a script needed before we had the ability to write directly to ome-zarr - hopefully we won't need it
# anymore, hence disabled qa.
# ruff: noqa

"""
A CLI tool to efficiently convert a zarr 2 archive into an ome-zarr 0.4 archive. This is useful internally at biohub,
as we are trying to pivot our analysis pipelines to ome-zarr but still have legacy zarr 2 data.

The reason this tool is fast compared to a simpler approach (i.e. load the zarr array with dask and then save it using
ome-zarr-py) is that we only do filesystem manipulation - to_ome just moves the existing files into the ome-zarr structure,
which is perfectly functional as ome-zarr images are just zarr archives. Then, we just create some additional metadata.

This tool does not create image pyramids - for a "good" ome-zarr archive, you should probably run `impp pyramid <path.zarr>`
after converting from zarr to ome zarr.
"""

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

import itertools
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import click
import numpy as np
import yaml

from . import zarr_utils

# While the program is running, the multiscale "0" will actually be named this. This is to
# avoid conflics where a zarr 2 array with the / dimension separator is being converted - as the
# folder "0" will already exist in the root directory in that case.
# At the end of conversion, the WIP prefix folder is renamed to "0".
WIP_PREFIX = "wip"

# scope just for today: convert zarr 2 archives to ome zarr
# dimension separator is '.' for this dataset


@click.command()
@click.argument("source", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option("--daxi_metadata", type=click.File(mode="r"), default=None)
@click.option("--run", default=False, is_flag=True)
@click.option(
    "--resume",
    default=False,
    is_flag=True,
    help="If you have partially completed a run, this will continue it (i.e. no folder creation, no )",
)
@click.option(
    "--remove-dim",
    type=int,
    default=None,
    help="If you have acquired to a 6D array, you can specify the 0 indexed axis you'd like to drop to permit conversion to OME-Zarr",
)
@click.option(
    "--partial",
    default=False,
    is_flag=True,
    help="If the archive is not fully complete but you'd like to move it anyway, use this flag to skip the integrity check.",
)
def main(source, daxi_metadata, run, resume, remove_dim, partial):
    if resume and run:
        raise ValueError("Resume is not currently supported due to the increasing complexity of this program")

    import logging

    logging.basicConfig(level=logging.DEBUG)
    if zarr_utils.identify_zarr_format(source) != 2:
        raise ValueError("This script only supports zarr 2 archives.")

    # As a dry run, we just print the move commands instead of executing them.
    move = shutil.move if run else print

    zarr_json_path = source / ".zarray"

    # TODO: reimplement resume
    # if resume:
    #     zarr_json_path = ZarrCopier.find_zarr_json(source / '0')
    # else:
    #     zarr_json_path = ZarrCopier.find_zarr_json(source)

    with open(zarr_json_path.expanduser()) as zarr_json_file:
        zarr_json = json.load(zarr_json_file)
        shape = np.array(zarr_json["shape"])

        # elif zarr_format == 3:
        #     chunks = np.array(zarr_json['chunk_grid']['configuration']['chunk_shape'])

        # Lots of error handling is needed for different edge cases about the shape
        if remove_dim is None:
            if shape.size != 5:
                raise ValueError(
                    "The input array must be 5 dimensional to be converted to ome zarr - did you forget to specify --remove-dim?"
                )
        else:
            if shape.size != 6:
                raise ValueError(
                    f"remove-dim can only be used on a six dimensional input array, but the input has {shape.size} dimensions."
                )

            if remove_dim < 0 or remove_dim >= 5:
                raise ValueError("remove-dim must be 0, 1, 2, 3, or 4.")

            if shape[remove_dim] != 0:
                raise ValueError(
                    f"remove-dim can only be used to delete a dimension that has shape 1 - the axis you chose has {shape[remove_dim]} elements."
                )

        chunks = np.array(zarr_json["chunks"])
        files_nd = np.ceil(shape / chunks).astype(int)
        dimension_separator = zarr_json["dimension_separator"]

    can_do_quick_copy = dimension_separator == "/" and remove_dim is None

    if not resume:
        # undo:
        # for chunk in itertools.product(*[range(n) for n in files_nd]):
        #     chunk2 = chunk[:2] + chunk[3:]
        #     chunk_file = source / "0" / ".".join(map(str, chunk2))
        #     move(chunk_file, source / ".".join(map(str, chunk)))
        # exit(1)

        # integrity check:
        if not partial:
            file_count = np.prod(files_nd)
            report_interval = min(400, file_count // 20)  # How many files should be checked between status updates.
            existing_count = 0
            for chunk in itertools.product(*[range(n) for n in files_nd]):
                chunk_file = source / dimension_separator.join(map(str, chunk))
                if not chunk_file.exists():
                    print(f"Missing chunk file {chunk_file}")
                    exit(1)
                existing_count += 1
                if existing_count == file_count or existing_count % report_interval == 0:
                    print(f"Pre-copy integrity check {existing_count / file_count * 100:.1f}% complete")

        print("Converting metadata")

        # Ensure we have all the metadata that we'll need:
        if daxi_metadata:
            dm = yaml.safe_load(daxi_metadata)
            # print(dm)

        ome_metadata = {
            "history": [" ".join(sys.argv)],
            "multiscales": [
                {
                    "axes": [
                        {"name": "T", "type": "time", "unit": "second"},
                        {"name": "C", "type": "channel"},
                        {"name": "Z", "type": "space", "unit": "micrometer"},
                        {"name": "Y", "type": "space", "unit": "micrometer"},
                        {"name": "X", "type": "space", "unit": "micrometer"},
                    ],
                    "coordinateTransformations": [{"type": "identity"}],
                    "datasets": [
                        {
                            "coordinateTransformations": [
                                {
                                    "scale": [
                                        1.0,
                                        1.0,
                                        dm["Z step size (um)"]
                                        if daxi_metadata and "Z step size (um)" in dm
                                        else 1.24427,
                                        0.439,
                                        0.439,
                                    ],
                                    "type": "scale",
                                }
                            ],
                            "path": "0",
                        }
                    ],
                    "name": "0",
                    "version": "0.4",
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
                        "label": f"v{view_idx}_c488",
                        "window": {"end": 65535.0, "max": 65535.0, "min": 0.0, "start": 0.0},
                    }
                    for view_idx in range(shape[1])
                ],
                "id": 0,
                "name": "",
                "rdefs": {"defaultT": 0, "defaultZ": 0, "model": "color", "projection": "normal"},
                "version": "0.4",
            },
            "daxi": {"version": "0.0"},
        }

        daxi_json = {"version": "0.0"}

        if daxi_metadata:
            if "Timepoint 0.View 0" in dm.keys():
                key_pattern = re.compile("Timepoint (\\d+).View (\\d+)")
            else:
                raise ValueError("The daxi metadata is not in a known layout")
            tz = ZoneInfo("America/Los_Angeles")
            time_windows = {}

            def localized_iso_timestamp(time_str):
                seconds_since_epoch, microseconds_since_epoch = (int(part) for part in time_str.split("."))
                # Combine seconds and microseconds into a total timestamp
                total_microseconds = seconds_since_epoch * 1_000_000 + microseconds_since_epoch

                # Create a datetime object from the total microseconds
                dt = datetime.fromtimestamp(total_microseconds / 1_000_000, tz=timezone.utc)

                # Localize to America/Los_Angeles timezone
                la_tz = ZoneInfo("America/Los_Angeles")
                localized_dt = dt.astimezone(la_tz)

                # Return the ISO format string
                return localized_dt.isoformat()

            # Collect all of the timestamps and append them to the ome-zarr metadata:
            for key, value in dm.items():
                match = key_pattern.match(key)
                if match:
                    timepoint = int(match.group(1))
                    view = int(match.group(2))

                    start_dt, end_dt = (
                        localized_iso_timestamp(time_str)
                        for time_str in (value["start timestamp"], value["end   timestamp"])
                    )
                    if timepoint not in time_windows:
                        time_windows[timepoint] = {}

                    time_windows[timepoint][view] = {"start": start_dt, "end": end_dt}

            daxi_json["timing_detail"] = time_windows
            daxi_json["framerate_hz"] = float(dm["Frame rate"]) if "Frame rate" in dm else 60

        zattrs = source / ".zattrs"
        if zattrs.exists():
            print(".zattrs file exists - aborting to avoid corruption.")
            exit(1)

        if run:
            # Execute the transformation:
            with open(zattrs, "w") as zattrs_file:
                zattrs_file.write(json.dumps(ome_metadata, indent=4))

            if daxi_metadata:
                with open(source / "daxi.json", "w") as daxi_file:
                    daxi_file.write(json.dumps(daxi_json, indent=4))

            zgroup = source / ".zgroup"
            with open(zgroup, "w") as zgroup_file:
                zgroup_file.write(json.dumps({"zarr_format": 2}, indent=4))

            if can_do_quick_copy:
                (source / WIP_PREFIX).mkdir(parents=True, exist_ok=True)
            else:
                # Create the folder for this multiscale (assuming we can't do a quick copy):
                if remove_dim:
                    files_nd_reduced = list(files_nd[:remove_dim]) + list(files_nd[(remove_dim + 1) :])
                    create_ome_zarr_folder_structure(source, files_nd_reduced)
                else:
                    create_ome_zarr_folder_structure(source, files_nd)

            # Fix the dimension separator and shape of the array:
            zarray_filepath = source / ".zarray"
            with open(zarray_filepath) as zarray_file:
                zarray_json = json.load(zarray_file)

            zarray_json["dimension_separator"] = "/"

            if remove_dim is not None:
                for key in ("shape", "chunks"):
                    arr = zarray_json[key]
                    zarray_json[key] = arr[:remove_dim] + arr[(remove_dim + 1) :]

            with open(zarray_filepath, "w") as zarray_file:
                zarray_file.write(json.dumps(zarray_json, indent=4))

            # Move the .zarray file:
            move(zarray_filepath, source / WIP_PREFIX / ".zarray")

    if resume and remove_dim:
        # resume is mega broken don't use resume
        files_nd = list(files_nd[:2]) + [1] + list(files_nd[2:])
        files_nd = np.array(files_nd)

    print("Starting rearrangement")

    # print(*[range(n) for n in files_nd])

    file_count = np.prod(files_nd)
    report_interval = min(400, file_count // 20)  # How many files should be checked between status updates.
    existing_count = 0

    if can_do_quick_copy:
        # If the files are already slash separated and the correct number of dimensions, we just have to
        # move the top level folders into wip
        for i in range(files_nd[0]):
            target = source / str(i)
            if target.exists():
                move(target, source / WIP_PREFIX / str(i))
    else:
        for chunk in itertools.product(*[range(n) for n in files_nd]):
            # This line of code is brittle if resume is happening!
            src_filename = dimension_separator.join(map(str, chunk))
            if remove_dim is not None:
                dest_chunk = chunk[:remove_dim] + chunk[(remove_dim + 1) :]
            else:
                dest_chunk = chunk
            dest_path = source / WIP_PREFIX / Path(*map(str, dest_chunk))
            # dest_filename = ".".join(map(str, dest_chunk))

            # print((source/src_filename).is_file())
            # print(source / src_filename)
            if (source / src_filename).exists():
                move(source / src_filename, dest_path)
            else:
                print(f"WARNING: file {src_filename} was not found, and could not be moved to {dest_path}")

            existing_count += 1
            if existing_count == file_count or existing_count % report_interval == 0:
                print(f"Renaming: {existing_count / file_count * 100:.1f}% complete")
            # Sheng's code uses a format given by Time x View x Color x Z x Y x X
            # The acquisitions we use are only ever single color at the moment so we can probably just ignore
            # that key
            # move(chunk_file, source / "0" / Path(*map(str, chunk)))
            # parts = chunk[:2] + chunk[3:]
            # print(chunk_file, source / "0" / Path(*map(str, parts)))

        if run and dimension_separator == "/":
            # Delete the folder mess
            for i in range(files_nd[0]):
                deletion_target = source / str(i)
                if deletion_target.exists():
                    shutil.rmtree(deletion_target)

    move(source / WIP_PREFIX, source / "0")


def create_ome_zarr_folder_structure(destination, files_nd):
    for coord in itertools.product(*[range(n) for n in files_nd[:-1]]):
        terminal_folder = destination / WIP_PREFIX / Path(*map(str, coord))
        terminal_folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
