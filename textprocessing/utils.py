import logging
import sys

def init_log(log_level="WARNING", log_target=sys.stderr):
    log = logging.getLogger("textauger")
    log.setLevel(getattr(logging, log_level.upper()))
    ch = logging.StreamHandler(log_target)
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(
        ' [%(levelname)s] %(asctime)s - %(module)s.%(funcName)s(l.%(lineno)s): %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    log.handlers = [ch]
    return log

log = init_log()
