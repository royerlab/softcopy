import logging
from pathlib import Path
import json
import sys
import time
import itertools

import click
import yaml
import numpy as np
import psutil

from .zarr_copier import ZarrCopier

BOLD_SEQ = "\033[1m"
RESET_SEQ = "\033[0m"
LOG = logging.getLogger(__name__)

@click.command()
@click.option("--verbose", default=False, is_flag=True, help="print debug information while running")
@click.option("--nprocs", default=1, type=int, help="number of processes to use for copying")
@click.argument("targets_file", type=click.File("r"))
def main(verbose, nprocs, targets_file):
    """Tranfer data from source to destination as described in a yaml TARGETS_FILE. Uses low priority io to allow
    data to be moved while the microscope is acquiring. The program is zarr-aware and can safely copy an archive
    before it is finished being written to."""
    
    log_level = logging.INFO if not verbose else logging.DEBUG
    LOG.setLevel(log_level)
    logging.basicConfig(format="[%(asctime)s : %(levelname)s from %(name)s] " + BOLD_SEQ + "%(message)s" + RESET_SEQ)
    
    # Load the yaml at a normal io priority because it is small and likely not on
    # the target disk
    all_yaml = yaml.safe_load(targets_file)
    targets = all_yaml['targets']
    LOG.debug(f"Number of targets: {len(targets)}")

    # Now that we have the yaml, we floor our io priority. We are about to read zarr metadata, and even doing that
    # at the wrong time could slow down the writer process!
    ensure_low_io_priority()

    copiers = []

    try:
        for target_id, target in enumerate(targets):
            source = Path(target['source']).expanduser().absolute()
            destination = Path(target['destination']).expanduser().absolute()
            copier = ZarrCopier(source, destination, nprocs, LOG.getChild(f"Target {target_id}"))
            copiers.append(copier)
            copier.start()
            copier.join()
    except KeyboardInterrupt:
        LOG.info("Keyboard interrupt recieved, stopping all copiers")
        for copier in copiers:
            copier.stop()
        # log = LOG.getChild(f"Target {target_id}")
        # log.info(f"Target {target_id}: from {target['source']} to {target['destination']}")
        # zarr_archive_location = Path(target['source']).expanduser()
        
        # zarr_json_path = find_zarr_json(zarr_archive_location, log)
        
        # # Compute the number of files in the zarr archive using shape and chunk size
        # # files_nd = shape / chunk_size (ignores shard chunk size - those are all internal to a file)
        # with open(zarr_json_path.expanduser()) as zarr_json_file:
        #     zarr_json = json.load(zarr_json_file)
        #     zarr_version = zarr_json['zarr_format']
        #     shape = np.array(zarr_json['shape'])
        #     if zarr_version == 2:
        #         chunks = np.array(zarr_json['chunks'])
        #     elif zarr_version == 3:
        #         chunks = np.array(zarr_json['chunk_grid']['configuration']['chunk_shape'])
            
        #     files_nd = np.ceil(shape / chunks).astype(int)
        #     log.debug(f"Shape: {shape}, chunks: {chunks}, files_nd: {files_nd}")
        #     log.debug(f"Number of files in zarr archive: {np.prod(files_nd)}")
    
        # create_zarr_folder_structure(Path(target['destination']).absolute(), zarr_version, files_nd, log)
    
    # How can we know if a file is safe to start copying?
    # 1. The `complete` file exists
    # 2. Its successor file exists (i.e. 0/0/0 is done being written if 0/0/1 exists)
    # 3. We recieved a filesystem event that it closed
    #
    # We don't want to use the disk more than we have to. 3 is free from the windows filesystem watching
    # APIs, but we need to poll for 1 and 2.
    #
    # So, we set up watchdog to poll for events immediately, and every time it gets a closed event, it adds
    # that filename to the list of files that are safe to copy.
    # Then, and only once at the beginning of the program, we poll the completed file and get a complete
    # list of all the files that exist in this zarr archive. Together, this is enough information to
    # know what's ready to be copied at any point in time.

    # As a test, predict the filepaths of each file in the zarr archive.
    # expected_files = set()
    # for coord in itertools.product(*[range(n) for n in files_nd]):
    #     if zarr_version == 2:
    #         file_path = zarr_archive_location / ".".join(map(str, coord))
    #     else:
    #         file_path = zarr_archive_location / "c" / Path(*map(str, coord))
    #     expected_files.add(file_path.absolute())

    # prepared_files = set()

    # missed_files = set(expected_files) - set(prepared_files)
    # log.info(f"Missed files: {len(missed_files)}")
    # import pdb; pdb.set_trace()

def ensure_low_io_priority():
    if sys.platform == "linux":
        # On linux, 7 is the highest niceness => lowest io priority. IDLE is the lowest priority
        # class
        psutil.Process().ionice(psutil.IOPRIO_CLASS_IDLE, 7)
    elif sys.platform == "win32":
        # On windows, 0 is "very low" io priority
        psutil.Process().ionice(0)
    else:
        LOG.warning("Cannot set low io priority on this platform")

if __name__ == "__main__":
    main()