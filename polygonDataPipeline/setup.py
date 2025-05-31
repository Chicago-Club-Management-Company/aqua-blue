#!/usr/bin/env python3
"""
Setup script for the aqua-blue stock data pipeline module.

This script installs the necessary dependencies for real-time stock data
streaming and historical data loading using polygon.io websockets.

Usage:
    python setup.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a shell command with error handling."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    dependencies = [
        "websockets>=13.0",
        "aiohttp>=3.11.0", 
        "numpy>=2.0.0",
        "python-dateutil>=2.9.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    return True


def verify_installation():
    """Verify that key dependencies can be imported."""
    print("🔍 Verifying installation...")
    
    imports_to_test = [
        ("websockets", "WebSocket support"),
        ("aiohttp", "HTTP async client"),
        ("numpy", "Numerical computing"),
        ("dateutil", "Date utilities")
    ]
    
    all_passed = True
    for module, description in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {description} - OK")
        except ImportError:
            print(f"❌ {description} - FAILED")
            all_passed = False
    
    return all_passed


def create_config_template():
    """Create a configuration template file."""
    config_path = Path("config_template.py")
    
    config_content = '''"""
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
'''
    
    try:
        with open(config_path, "w") as f:
            f.write(config_content)
        print(f"✅ Created configuration template at {config_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create config template: {e}")
        return False


def main():
    """Main setup function."""
    print("📊 aqua-blue Stock Data Pipeline Module Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Create helper files
    create_config_template()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Copy config_template.py to config.py")
    print("2. Add your polygon.io API key to config.py")
    print("3. Run: python examples/data_pipeline_example.py --symbol AAPL")
    print("\n📚 Available examples:")
    print("• python examples/data_pipeline_example.py --symbol AAPL --historical-days 1825")
    print("• python examples/flat_files_example.py --symbol AAPL --days 365")
    print("\n🔗 Get your API key: https://polygon.io/")


if __name__ == "__main__":
    main() 