import os
from pathlib import Path
from typing import Literal

import numpy as np


class PackedName:
    """
    In applications like Softcopy where millions of filepaths within a zarr archive are being processed, the memory
    overhead of using pathlib.Path objects for each filepath can be significant. This class is a memory efficient
    wrapper which compresses filenames into a single integer index when possible.

    To optimize performance, this class is ambiguous about *which* zarr archive the file is in. For example,
    /archive1.zarr/0/1/2 and /archive2.zarr/0/1/2 will both compress to the same PackedName instance. This is
    unambiguous in Softcopy, as there is a one-to-one mapping between ZarrCopier classes and zarr archive, but
    for general use, some additional context may be needed to disambiguate.
    """

    # __slots__ is a special attribute that tells Python to not use a dict, and only allocate space for a fixed set of
    # attributes. This is a performance optimization which saves memory.
    __slots__ = ("_path", "_index")

    def __init__(self, name: str, files_nd: np.ndarray, dim_separator: Literal["/", "."], zarr_format: Literal[2, 3]):
        if dim_separator == "/":
            parts = name.split(os.sep)
        else:
            last_slash = name.rfind(os.sep)
            parts = name[last_slash + 1 :].split(dim_separator)

        require_c_prefix = zarr_format == 3
        needed_parts = files_nd.size + (1 if require_c_prefix else 0)

        if len(parts) < needed_parts:
            self._path = name
            self._index = None
            return

        if require_c_prefix and parts[-needed_parts][0] != "c":
            self._path = name
            self._index = None
            return

        # Think this is unneeded
        # if require_c_prefix:
        #     if parts[-needed_parts][0] != "c":
        #         self._path = name
        #         self._index = None
        #         return
        #     parts = parts[-needed_parts:]

        try:
            chunk_index_nd = tuple(int(p) for p in parts[-files_nd.size :])
        except ValueError:
            self._path = name
            self._index = None
            return

        # To make things faster, we just assume validity in Softcopy. Someday we might change that
        # if not all(0 <= coord < files_nd[i] for i, coord in enumerate(chunk_index_nd)):
        #     self._path = name
        #     self._index = None
        #     return

        self._index = np.ravel_multi_index(chunk_index_nd, files_nd)
        self._path = None

    def from_index(index: int):
        """
        Create a PackedName instance from an index. This is useful when you know the index of a file but not the path.
        """
        ret = PackedName.__new__(PackedName)
        ret._index = index
        ret._path = None
        return ret

    def path_from_index(
        index: int,
        files_nd: np.ndarray,
        zarr_location: Path,
        dim_separator: Literal["/", "."],
        zarr_format: Literal[2, 3],
    ) -> Path:
        chunk_index_nd = np.unravel_index(index, files_nd)

        # We don't need to worry about using `/` here, because on Windows, `Path` will automatically convert `/` to `\`
        prefixless_chunk_key = dim_separator.join(map(str, chunk_index_nd))

        if zarr_format == 3:
            return zarr_location / "c" / prefixless_chunk_key
        return zarr_location / prefixless_chunk_key

    def get_path(
        self, files_nd: np.ndarray, zarr_location: Path, dim_separator: Literal["/", "."], zarr_format: Literal[2, 3]
    ) -> Path:
        if self._path is not None:
            return zarr_location / self._path
        elif self._index is not None:
            return PackedName.path_from_index(self._index, files_nd, zarr_location, dim_separator, zarr_format)

    def is_zarr_chunk(self):
        return self._index is not None
