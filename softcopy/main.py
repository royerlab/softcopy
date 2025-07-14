import logging
import sys
from pathlib import Path

import click

from .hcs_copier import HCSCopier
from .priority import set_low_io_priority
from .zarr_copier import ZarrCopier

BOLD_SEQ = "\033[1m"
RESET_SEQ = "\033[0m"
LOG = logging.getLogger(__name__)


@click.command()
@click.argument("source-path", type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.argument("dest-path", type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.option("--verbose", default=False, is_flag=True, help="print debug information while running")
@click.option("--nprocs", default=8, type=int, help="number of processes to use for copying")
@click.option(
    "--sleep-time",
    default=5.0,
    type=float,
    help="time to sleep in each copy process between copies. Can help mitigate down an overwhelemed system",
)
@click.option(
    "--wait-for-source",
    default=True,
    help="If the source does not exist when softcopy is started, wait for it to appear. If false, softcopy will crash if the source does not exist",
)
def main(source_path, dest_path, verbose, nprocs, sleep_time, wait_for_source):
    """Tranfer data from source to destination as described in a yaml TARGETS_FILE. Uses low priority io to allow
    data to be moved while the microscope is acquiring. The program is zarr-aware and can safely copy an archive
    before it is finished being written to."""

    log_level = logging.INFO if not verbose else logging.DEBUG
    LOG.setLevel(log_level)
    logging.basicConfig(format="[%(asctime)s : %(levelname)s from %(name)s] " + BOLD_SEQ + "%(message)s" + RESET_SEQ)

    set_low_io_priority()

    try:
        source_path = source_path.expanduser().absolute()
        dest_path = dest_path.expanduser().absolute()
    except Exception:
        LOG.exception("Error expanding paths")
        sys.exit(1)

    filetype_map = {
        ".ome.zarr": HCSCopier,
        ".zarr": ZarrCopier,
    }
    # We are going to find the longeest (most specific) suffix match - so we sort the keys by length
    # in descending order. This way, if a file has multiple suffixes (e.g., .ome.zarr and .zarr), the more specific
    # one will be chosen first.
    suffix_choices = sorted(filetype_map.keys(), key=lambda suffix: len(suffix), reverse=True)
    best_suffix = next((
        suffix for suffix in suffix_choices if source_path.name.lower().endswith(suffix)
    ), None)
    if best_suffix is None:
        LOG.error(f"Source path {source_path} does not have a recognized suffix. Supported suffixes are: {', '.join(suffix_choices)}")
        sys.exit(1)

    print(f"Best suffix found: {best_suffix}")
    copier_class = filetype_map[best_suffix]

    copier = copier_class(
        source_path,
        dest_path,
        n_copy_procs=nprocs,
        sleep_time=sleep_time,
        wait_for_source=wait_for_source,
        log=LOG.getChild("Copier"),
    )

    try:
        LOG.info(f"Starting copy from {source_path} to {dest_path} using {copier_class.__name__}")
        copier.start()
        copier.join()
        LOG.info("Copy completed successfully.")
    except KeyboardInterrupt:
        LOG.info("Keyboard interrupt received, stopping copier.")
        copier.stop()
    except Exception:
        LOG.exception("An error occurred during copying")
        copier.stop()
        sys.exit(1)
    finally:
        copier.join()
        LOG.info("Copier shut down successfully.")


if __name__ == "__main__":
    main()
