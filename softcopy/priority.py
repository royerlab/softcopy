import logging
import sys

import psutil
from psutil import AccessDenied

LOG = logging.getLogger(__name__)

def set_low_io_priority():
    try:
        if sys.platform == "linux":
            # On linux, 7 is the highest niceness => lowest io priority. IDLE is the lowest priority
            # class
            psutil.Process().ionice(psutil.IOPRIO_CLASS_IDLE)
        elif sys.platform == "win32":
            # On windows, 0 is "very low" io priority
            psutil.Process().ionice(0)
        else:
            LOG.warning("Cannot set low io priority on this platform")
    except (PermissionError, AccessDenied):
        LOG.warning("Could not set low io priority, you may need to run as root to do this")