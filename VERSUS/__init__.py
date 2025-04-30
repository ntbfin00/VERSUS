import logging 

logging.basicConfig(level=logging.CRITICAL, 
        format='%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | Ln%(lineno)d | %(message)s',
        datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
