import numpy as np
import timeit
from pathlib import Path

# Original Implementation (for comparison)
class PackedNameOriginal:
    _path: Path = None
    _index: int = None

    def __init__(self, name: str, files_nd: np.ndarray, version: int = 2, dim_separator: str = "."):
        path = Path(name)

        if version == 3:
            parts = path.parts
        else:
            parts = path.name.split(dim_separator)

        needed_parts = (1 + files_nd.size) if version == 3 else files_nd.size

        if len(parts) < needed_parts:
            self._path = path
            return

        if version == 3:
            if parts[-needed_parts][0] != "c":
                self._path = path
                return
            parts = parts[-needed_parts:]

        try:
            chunk_index_nd = list(map(int, parts[-files_nd.size:]))
            if all(0 <= coord < files_nd[i] for i, coord in enumerate(chunk_index_nd)):
                self._index = np.ravel_multi_index(chunk_index_nd, files_nd)
                self._path = None
        except ValueError:
            self._path = path

    def get_path(self, files_nd: np.ndarray, zarr_location: Path, dim_separator: str = "."):
        if self._path is not None:
            return zarr_location / self._path
        elif self._index is not None:
            chunk_index_nd = np.unravel_index(self._index, files_nd)
            if dim_separator == "/":
                return zarr_location / Path(*map(str, chunk_index_nd))
            return zarr_location / dim_separator.join(map(str, chunk_index_nd))


# Optimized Implementation
class PackedNameOptimized:
    __slots__ = ('_path', '_index')

    def __init__(self, name: str, files_nd: np.ndarray, version: int = 2, dim_separator: str = "."):
        if version == 3:
            parts = name.split('/')
        else:
            parts = name.split(dim_separator)

        needed_parts = (1 + files_nd.size) if version == 3 else files_nd.size

        if len(parts) < needed_parts:
            self._path = name
            self._index = None
            return

        if version == 3:
            if parts[-needed_parts][0] != "c":
                self._path = name
                self._index = None
                return
            parts = parts[-needed_parts:]

        try:
            chunk_index_nd = tuple(int(p) for p in parts[-files_nd.size:])
        except ValueError:
            self._path = name
            self._index = None
            return

        if not all(0 <= coord < files_nd[i] for i, coord in enumerate(chunk_index_nd)):
            self._path = name
            self._index = None
            return

        self._index = np.ravel_multi_index(chunk_index_nd, files_nd)
        self._path = None

    def get_path(self, files_nd: np.ndarray, zarr_location: Path, dim_separator: str = "."):
        if self._path is not None:
            return zarr_location / self._path
        elif self._index is not None:
            chunk_index_nd = np.unravel_index(self._index, files_nd)
            if dim_separator == "/":
                return zarr_location / "/".join(map(str, chunk_index_nd))
            return zarr_location / dim_separator.join(map(str, chunk_index_nd))


# Set up the test environment
files_nd = np.array([100, 100, 100])  # Example 3D file array
zarr_location = Path("/zarr_location")
name = "c/50/50/50"
dim_separator = "/"

# Benchmark both implementations
def benchmark_original():
    packed = PackedNameOriginal(name, files_nd, version=3, dim_separator=dim_separator)
    return packed.get_path(files_nd, zarr_location, dim_separator=dim_separator)

def benchmark_optimized():
    packed = PackedNameOptimized(name, files_nd, version=3, dim_separator=dim_separator)
    return packed.get_path(files_nd, zarr_location, dim_separator=dim_separator)

# Run benchmarks
original_time = timeit.timeit(benchmark_original, number=100000)
optimized_time = timeit.timeit(benchmark_optimized, number=100000)

print(f"Original Implementation Time: {original_time} seconds")
print(f"Optimized Implementation Time: {optimized_time} seconds")

