import logging
import sys


def consoleLogger(level=logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger('consoleLogger')
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s %(filename)s %(lineno)d行]:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def fileLogger(filePath: str, level=logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger('fileLogger')
    logger.setLevel(level)
    handler = logging.FileHandler(filePath)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s %(filename)s %(lineno)d行]:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def fileAndConsoleLogger(filePath: str, level=logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger('fileLogger')
    logger.setLevel(level)
    fileHandler = logging.FileHandler(filePath)
    consoleHandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s %(filename)s %(lineno)d行]:%(message)s")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    return logger


# if __name__ == '__main__':
#     logger = getFileAndConsoleLogger(filePath='test.log')
#     logger.debug("asd")
