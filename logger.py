import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(created)f:%(levelname)s:%(message)s"))
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
