"""
Stock data pipeline module for aqua-blue reservoir computing.

This module provides real-time stock data streaming and historical data loading:
- polygon.io real-time market data streams
- Historical data loading and caching
- S3 Flat Files support for large datasets
- Data processing and buffering

Main components:
- StockDataPipeline: Main data pipeline engine for loading and streaming stock data
- PolygonWebsocketClient: Handles polygon.io websocket connections
- PolygonHistoricalDataFetcher: Fetches historical market data
- PolygonFlatFilesClient: Handles polygon.io flat files
- MarketDataProcessor: Processes and buffers market data
"""

__version__ = "0.1.0"
__author__ = "Ramez Karim"

from .stock_data_pipeline import StockDataPipeline
from .polygon_client import (
    PolygonWebsocketClient, 
    PolygonHistoricalDataFetcher, 
    PolygonFlatFilesClient,
    HistoricalDataPoint
)
from .data_processor import MarketDataProcessor

__all__ = [
    "StockDataPipeline", 
    "PolygonWebsocketClient",
    "PolygonHistoricalDataFetcher",
    "PolygonFlatFilesClient",
    "HistoricalDataPoint",
    "MarketDataProcessor",
] 