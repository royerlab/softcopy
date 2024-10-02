import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def workspace():
    """
    Creates a temporary directory to safely test filesystem operations inside of. Deletes the directory
    after the test finishes.
    """
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)