import logging
from abc import ABC, abstractmethod
from pathlib import Path

LOG = logging.getLogger(__name__)


class AbstractCopier(ABC):
    def __init__(
        self, source: Path, destination: Path, n_copy_procs: int, sleep_time: float = 1.0, log: logging.Logger = LOG
    ):
        self._source = source
        self._destination = destination
        self._n_copy_procs = n_copy_procs
        self._log = log
        self._sleep_time = sleep_time

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def join(self):
        pass

    @abstractmethod
    def stop(self):
        pass
