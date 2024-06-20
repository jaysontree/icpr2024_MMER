
import os
import logging

PROJECT_ROOT_PATH = os.path.abspath(os.path.join(__file__, '../../'))

DATA_ROOT = os.path.join(PROJECT_ROOT_PATH, "data")
CACHE_ROOT = os.path.join(DATA_ROOT, "cache")
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)

logger = logging.getLogger()
stream_handler = logging.StreamHandler()

log_formatter = logging.Formatter(fmt='%(asctime)s\t%(levelname)s\t%(name)s %(filename)s:%(lineno)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(log_formatter)

logger.addHandler(stream_handler)

logger.setLevel(logging.INFO)