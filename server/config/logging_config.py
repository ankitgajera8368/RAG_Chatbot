import logging
import logging.handlers
import os

# Example: log to both console and a rotating file
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_LEVEL = logging.INFO

LOG_FILE_PATH = os.path.join(LOG_DIR, "app.log")

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_FILE_PATH, maxBytes=1_000_000, backupCount=5
        ),
    ],
)
