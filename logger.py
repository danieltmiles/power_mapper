import logging
import datetime

_RESET = "\033[0m"
_LEVEL_COLORS = {
    logging.INFO: "\033[97m",     # white
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",    # red
    logging.CRITICAL: "\033[31m", # red
}


class _RFC3339Formatter(logging.Formatter):
    def __init__(self, fmt, service_name: str):
        super().__init__(fmt)
        self.service_name = service_name

    def formatTime(self, record, datefmt=None):
        return datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat()

    def format(self, record):
        color = _LEVEL_COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{_RESET}"


def get_logger(service_name: str) -> logging.Logger:
    logger = logging.getLogger(service_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_RFC3339Formatter(f"%(asctime)s [{service_name}] %(levelname)s %(filename)s:%(lineno)d %(message)s", service_name))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger
