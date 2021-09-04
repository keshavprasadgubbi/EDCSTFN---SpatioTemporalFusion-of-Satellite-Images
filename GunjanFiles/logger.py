import logging
import os

def get_logger(name, level=logging.INFO, out_dir=os.getcwd()):
    logger = logging.Logger(name)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s ', datefmt='%Y/%m/%d %H:%M:%S')
    handler = logging.FileHandler(os.path.join(out_dir, name + '.log'), mode='a', encoding=None, delay=False)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger