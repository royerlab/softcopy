import click
from pathlib import Path
import json

import tensorstore as ts

@click.command()
@click.argument('target', type=click.Path(exists=True))
def main(target: Path):
    zattrs_path = target / ".zattrs"
    with open(zattrs_path, "r") as f:
        zattrs = json.load(f)

if __name__ == '__main__':
    main()