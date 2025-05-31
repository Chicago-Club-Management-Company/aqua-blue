"""
Configuration template for stock data pipeline module.

Copy this file to 'config.py' and fill in your polygon.io API key.
"""

# Polygon.io API Configuration
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY_HERE"  # Get from https://polygon.io/

# Historical Data Settings
HISTORICAL_DATA_SETTINGS = {
    "min_historical_years": 5,  # Always load at least 5 years
    "default_historical_days": 1825,  # 5 years (5 * 365)
    "max_historical_years": 10  # Maximum years to search back
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_prefix": "data_pipeline"
}

# Data Processing Settings
DATA_PROCESSING = {
    "window_size": 1000,
    "time_interval_seconds": 1,
    "outlier_detection": True,
    "smoothing": False
}

# Stream Management
STREAM_SETTINGS = {
    "max_concurrent_streams": 5,
    "reconnect_attempts": 3,
    "heartbeat_interval": 30
}
