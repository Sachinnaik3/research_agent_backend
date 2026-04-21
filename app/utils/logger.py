from loguru import logger
import sys
import os

# Create logs folder if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Remove default logger
logger.remove()

# Console logging (colored)
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)

# File logging (rotating logs)
logger.add(
    f"{LOG_DIR}/app.log",
    rotation="10 MB",       # rotate after 10MB
    retention="10 days",    # keep logs for 10 days
    compression="zip",
    level="DEBUG",
)

# Error-specific log file
logger.add(
    f"{LOG_DIR}/error.log",
    level="ERROR",
)
