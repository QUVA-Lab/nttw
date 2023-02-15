"""Project wide logger."""
import datetime
import logging
import random


# Create the Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create the Handler for logging data to a file
now = datetime.datetime.now()
prefix_time = now.strftime("%Y%m%d_%H%M%S")
filename = f'log/{prefix_time}_{random.randint(0, 99999)}.log'
logger_handler = logging.FileHandler(filename=filename)
logger_handler.setLevel(logging.DEBUG)

# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# Add the Formatter to the Handler
logger_handler.setFormatter(logger_formatter)

# Add the Handler to the Logger
logger.addHandler(logger_handler)

# Add stream handler in same format
logger_handler = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)s - %(message)s')
logger_handler.setFormatter(logger_formatter)
logger_handler.setLevel(logging.DEBUG)
logger.addHandler(logger_handler)

# Make globally available
LOGGER_FILENAME = filename

logger.info(
  f'Completed configuring logger; writing from {__name__} to {filename}')
