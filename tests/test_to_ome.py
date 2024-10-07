import numpy as np

from softcopy.to_ome import main


def test_full_run(dummy_zarr_path, create_zarr2_archive):
    # Invoke the `main` function (i.e. the cli) to convert to ome zarr
    main([str(dummy_zarr_path), "--run"], standalone_mode=False)

    data = create_zarr2_archive["data"]
    chunks = create_zarr2_archive["chunks"]
    files_nd = np.floor_divide(data.shape, chunks)

    # Check that the ome zarr archive was created
    for chunk_id in np.ndindex(*files_nd):
        assert (dummy_zarr_path / "0" / "/".join(map(str, chunk_id))).exists()

    # Ensure that the contents of the ome zarr archive (non recursive) is just
    # .zattrs, .zgroup, and the data folder, "0/":
    top_level_contents = {node.name for node in dummy_zarr_path.iterdir()}
    expected_contents = {".zattrs", ".zgroup", "0"}
    assert top_level_contents == expected_contents


# TODO: test for / dimension separator input, remove dim
