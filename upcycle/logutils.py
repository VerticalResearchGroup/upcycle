import logging

from .common import *

# Barrowed from https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

CGREY    = '\33[90m'
GREY = "\x1b[38;20m"
YELLOW = "\x1b[33;20m"
RED = "\x1b[31;20m"

RESET_SEQ = "\x1b[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

class CustomFormatter(logging.Formatter):
    format = "[%(levelname)s:%(name)s][@%(asctime)s] %(message)s"

    FORMATS = {
        logging.DEBUG: CGREY + format + RESET_SEQ,
        logging.INFO: format,
        logging.WARNING: YELLOW + format + RESET_SEQ,
        logging.ERROR: RED + format + RESET_SEQ,
        logging.CRITICAL: BOLD_SEQ + RED + format + RESET_SEQ
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
