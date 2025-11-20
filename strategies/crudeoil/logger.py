# logger.py
import logging
import os
from datetime import datetime
import sys

def setup_logger(name, log_dir="logs"):
    """
    Sets up a logger that writes to a file and the console.
    
    Args:
        name (str): Name of the logger (usually the strategy name).
        log_dir (str): Directory to save log files.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # Capture everything
    
    # Prevent adding multiple handlers if logger is already set up
    if logger.hasHandlers():
        return logger

    # Create handlers
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{name}_{today_str}.log")
    
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO) # Console shows INFO and above
    
    # Create formatters and add it to handlers
    # File: Detailed with timestamp
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Console: Simpler
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    
    file_handler.setFormatter(file_format)
    console_handler.setFormatter(console_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
