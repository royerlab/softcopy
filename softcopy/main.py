import logging
import sys
from pathlib import Path

import click
import psutil
import yaml

from .zarr_copier import ZarrCopier

BOLD_SEQ = "\033[1m"
RESET_SEQ = "\033[0m"
LOG = logging.getLogger(__name__)


@click.command()
@click.argument("targets_file", type=click.File("r"))
@click.option("--verbose", default=False, is_flag=True, help="print debug information while running")
@click.option("--nprocs", default=1, type=int, help="number of processes to use for copying")
def main(targets_file, verbose, nprocs):
    """Tranfer data from source to destination as described in a yaml TARGETS_FILE. Uses low priority io to allow
    data to be moved while the microscope is acquiring. The program is zarr-aware and can safely copy an archive
    before it is finished being written to."""

    log_level = logging.INFO if not verbose else logging.DEBUG
    LOG.setLevel(logging.DEBUG)
    logging.basicConfig(format="[%(asctime)s : %(levelname)s from %(name)s] " + BOLD_SEQ + "%(message)s" + RESET_SEQ)

    # Load the yaml at a normal io priority because it is small and likely not on
    # the target disk
    all_yaml = yaml.safe_load(targets_file)
    targets = all_yaml["targets"]
    LOG.debug(f"Number of targets: {len(targets)}")

    # Now that we have the yaml, we floor our io priority. We are about to read zarr metadata, and even doing that
    # at the wrong time could slow down the writer process!
    ensure_low_io_priority()

    copiers = []

    try:
        for target_id, target in enumerate(targets):
            source = Path(target["source"]).expanduser().absolute()
            destination = Path(target["destination"]).expanduser().absolute()
            copier = ZarrCopier(source, destination, nprocs, LOG.getChild(f"Target {target_id}"))
            copiers.append(copier)
            copier.start()
            copier.join()
    except KeyboardInterrupt:
        LOG.info("Keyboard interrupt recieved, stopping all copiers")
        for copier in copiers:
            copier.stop()

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
