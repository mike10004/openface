from __future__ import print_function

import os
import sys
import logging


_LOGGING_CONFIGURED = False


def configure_logging():
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    level_str = os.getenv('UNIT_TESTS_LOG_LEVEL') or 'INFO'
    try:
        level = logging.__dict__[level_str]
    except KeyError:
        print("log level invalid:", level_str, file=sys.stderr)
        level = logging.INFO
    logging.basicConfig(level=level)
    _LOGGING_CONFIGURED = True
