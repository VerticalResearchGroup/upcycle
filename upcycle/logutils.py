import logging

from .common import *

# Barrowed from https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

CYAN = '\x1b[38;5;39m'
CGREY    = '\33[90m'
GREY = "\x1b[38;20m"
YELLOW = "\x1b[33;20m"
RED = "\x1b[31;20m"

RESET_SEQ = "\x1b[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

class CustomFormatter(logging.Formatter):
    format = "[%(levelname)s:%(name)s][@%(asctime)s] {msg_fmt_start}%(message)s{msg_fmt_end}"

    FORMATS = {
        logging.DEBUG: CGREY + format + RESET_SEQ,
        logging.INFO: format,
        logging.WARNING: YELLOW + format + RESET_SEQ,
        logging.ERROR: RED + format + RESET_SEQ,
        logging.CRITICAL: BOLD_SEQ + RED + format + RESET_SEQ
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keywords = []

    def format(self, record):
        bold = False
        for kw in self.keywords:
            if kw in record.getMessage(): bold = True

        log_fmt = self.FORMATS.get(record.levelno)

        if bold:
            log_fmt = log_fmt.format(msg_fmt_start=CYAN, msg_fmt_end=RESET_SEQ)
        else:
            log_fmt = log_fmt.format(msg_fmt_start='', msg_fmt_end='')

        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
