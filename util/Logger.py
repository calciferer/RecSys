import logging
import sys


logger = logging.getLogger('consoleLogger')

logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(levelname)s %(asctime)s %(filename)s %(lineno)d行]:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

if __name__ == '__main__':
    logger.debug("asd")
