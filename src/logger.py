import logging
import os
from datetime import datetime

# Create a log file name based on the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%f')}.log"

# Define the directory where logs will be saved
logs_dir = os.path.join(os.getcwd(), "logs")  # Log directory path

# Create the directory if it does not exist
os.makedirs(logs_dir, exist_ok=True)

# Full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Test logging
logging.info("Logging setup complete.")
