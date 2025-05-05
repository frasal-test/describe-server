import logging
import os
import time
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure API logger
api_logger = logging.getLogger("api_interactions")
api_logger.setLevel(logging.INFO)

# Create a file handler for the API logger
log_file = logs_dir / f"api_interactions_{datetime.now().strftime('%Y-%m-%d')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
api_logger.addHandler(file_handler)

class APILogger:
    @staticmethod
    def log_request(client_ip, endpoint, request_data):
        """Log API request information"""
        api_logger.info(f"REQUEST - IP: {client_ip} - ENDPOINT: {endpoint} - DATA: {request_data}")
    
    @staticmethod
    def log_response(client_ip, endpoint, response_data, execution_time):
        """Log API response information"""
        # Truncate response if too long
        if isinstance(response_data, str) and len(response_data) > 500:
            response_data = response_data[:500] + "... [truncated]"
        
        api_logger.info(
            f"RESPONSE - IP: {client_ip} - ENDPOINT: {endpoint} - "
            f"EXECUTION TIME: {execution_time:.4f}s - RESPONSE: {response_data}"
        )
    
    @staticmethod
    def log_error(client_ip, endpoint, error):
        """Log API error information"""
        api_logger.error(f"ERROR - IP: {client_ip} - ENDPOINT: {endpoint} - ERROR: {str(error)}")