import logging

logger = logging.getLogger(__name__)

def configure_logging(debug_file: str, info_file: str):
    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all logs at DEBUG and higher

    # Handler for DEBUG and higher (DEBUG log file)
    debug_handler = logging.FileHandler(debug_file, mode='w')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter('%(message)s'))  # No prefix, just message
    logger.addHandler(debug_handler)

    # Handler for INFO and higher (INFO log file)
    info_handler = logging.FileHandler(info_file, mode='w')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(logging.Formatter('%(message)s'))  # No prefix, just message
    logger.addHandler(info_handler)

    # Suppress logs from boto3 and botocore at DEBUG level
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)