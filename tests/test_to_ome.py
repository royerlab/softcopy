import tempfile
from pathlib import Path

import numpy as np
import tensorstore as ts

from softcopy.to_ome import main

def test_full_run():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        # Create a dummy zarr 2 archive
        shape = [10, 10, 10, 10, 10]
        chunks = [1, 1, 10, 10, 10]
        files_nd = [10, 10, 1, 1, 1]
        data = np.random.random(shape)
        dataset = ts.open({
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": tmpdirname},
            "metadata": {"dtype": ">f8", "shape": shape, "chunks": chunks},
            "create": True,
        }).result()
        dataset[:].write(data).result()

        # Invoke the `main` function (i.e. the cli) to convert to ome zarr
        main([tmpdirname, "--run"], standalone_mode=False)

        # Check that the ome zarr archive was created
        for chunk_id in np.ndindex(*files_nd):
            assert (tmpdir / "0" / "/".join(map(str, chunk_id))).exists()

        # Ensure that the contents of the ome zarr archive (non recursive) is just
        # .zattrs, .zgroup, and the data folder, "0/":
        top_level_contents = {node.name for node in tmpdir.iterdir()}
        expected_contents = {".zattrs", ".zgroup", "0"}
        assert top_level_contents == expected_contents

# TODO: test for / dimension separator input, remove dim