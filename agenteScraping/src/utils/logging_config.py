import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_file='logs/scraping_agent.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('scraping_agent')
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    console_handler = logging.StreamHandler()

    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(file_format)
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger