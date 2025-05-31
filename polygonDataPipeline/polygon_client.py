"""
Polygon.io WebSocket client for real-time market data streaming.

This module implements a WebSocket client that connects to polygon.io's real-time
market data streams, handling authentication, subscriptions, and data processing.
It also includes historical data fetching for model training.
"""

import asyncio
import websockets
import json
import logging
import aiohttp
import boto3
import gzip
import csv
import io
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import ssl
from botocore.config import Config

logger = logging.getLogger(__name__)


@dataclass
class PolygonMessage:
    """Represents a message received from polygon.io websocket."""
    event_type: str
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]


@dataclass
class PolygonWebsocketClient:
    """
    WebSocket client for polygon.io real-time market data.
    
    Connects to polygon.io's websocket streams and handles:
    - Authentication with API key
    - Subscription management for different symbols
    - Real-time data processing and callbacks
    
    Attributes:
        api_key (str): Your polygon.io API key
        base_url (str): The websocket endpoint URL
        subscriptions (List[str]): List of subscribed symbols
        message_handlers (Dict): Callbacks for different message types
        is_connected (bool): Connection status
    """
    
    api_key: str
    base_url: str = "wss://socket.polygon.io/stocks"
    subscriptions: List[str] = field(default_factory=list)
    message_handlers: Dict[str, Callable] = field(default_factory=dict)
    is_connected: bool = field(default=False, init=False)
    websocket: Optional[websockets.WebSocketServerProtocol] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize default message handlers."""
        self.message_handlers = {
            "T": self._handle_trade,
            "Q": self._handle_quote,
            "A": self._handle_aggregate,
            "status": self._handle_status,
            "error": self._handle_error
        }
    
    async def connect(self) -> bool:
        """
        Establish websocket connection to polygon.io.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create SSL context for secure connection
            ssl_context = ssl.create_default_context()
            
            # Connect to polygon.io websocket
            self.websocket = await websockets.connect(
                self.base_url,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10
            )
            
            # Wait for initial connection confirmation
            response = await self.websocket.recv()
            initial_response = json.loads(response)
            
            # Handle initial connection response (could be array or single message)
            connected = False
            if isinstance(initial_response, list):
                for msg in initial_response:
                    if (isinstance(msg, dict) and 
                        msg.get("ev") == "status" and 
                        msg.get("status") == "connected"):
                        connected = True
                        logger.info("Received connection confirmation from polygon.io")
                        break
            elif (isinstance(initial_response, dict) and 
                  initial_response.get("ev") == "status" and 
                  initial_response.get("status") == "connected"):
                connected = True
                logger.info("Received connection confirmation from polygon.io")
            
            if not connected:
                logger.error(f"Expected 'connected' status, got: {initial_response}")
                return False
            
            # Now send authentication message
            auth_message = {
                "action": "auth",
                "params": self.api_key
            }
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for authentication response
            auth_response_raw = await self.websocket.recv()
            auth_response = json.loads(auth_response_raw)
            
            # Handle authentication response (could be array or single message)
            authenticated = False
            if isinstance(auth_response, list):
                for msg in auth_response:
                    if (isinstance(msg, dict) and 
                        msg.get("ev") == "status" and 
                        msg.get("status") == "auth_success"):
                        authenticated = True
                        logger.info("Successfully authenticated to polygon.io")
                        break
                    elif (isinstance(msg, dict) and 
                          msg.get("ev") == "status" and 
                          "status" in msg):
                        logger.error(f"Authentication failed: {msg}")
                        return False
            elif (isinstance(auth_response, dict) and 
                  auth_response.get("ev") == "status" and 
                  auth_response.get("status") == "auth_success"):
                authenticated = True
                logger.info("Successfully authenticated to polygon.io")
            elif (isinstance(auth_response, dict) and 
                  auth_response.get("ev") == "status"):
                logger.error(f"Authentication failed: {auth_response}")
                return False
            
            if authenticated:
                self.is_connected = True
                return True
            else:
                logger.error(f"No authentication success in response: {auth_response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to polygon.io: {e}")
            return False
    
    async def disconnect(self):
        """Close the websocket connection."""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
        self.is_connected = False
        logger.info("Disconnected from polygon.io")
    
    async def subscribe(self, symbols: List[str], stream_types: List[str] = None) -> bool:
        """
        Subscribe to real-time data for specified symbols.
        
        Args:
            symbols (List[str]): List of stock symbols (e.g., ['AAPL', 'TSLA'])
            stream_types (List[str]): Types of data streams ('T'=trades, 'Q'=quotes, 'A'=aggregates)
            
        Returns:
            bool: True if subscription successful
        """
        if not self.is_connected:
            logger.error("Must be connected before subscribing")
            return False
        
        if stream_types is None:
            stream_types = ["T", "Q"]  # Default to trades and quotes
        
        try:
            # Build subscription message
            subscriptions = []
            for stream_type in stream_types:
                for symbol in symbols:
                    subscriptions.append(f"{stream_type}.{symbol}")
            
            subscribe_message = {
                "action": "subscribe",
                "params": ",".join(subscriptions)
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            
            # Wait for subscription confirmation
            response = await self.websocket.recv()
            sub_response = json.loads(response)
            
            # Handle subscription response - polygon.io uses "ev":"status" format
            subscribed = False
            if isinstance(sub_response, list):
                for msg in sub_response:
                    if (isinstance(msg, dict) and 
                        msg.get("ev") == "status" and 
                        msg.get("status") == "success"):
                        subscribed = True
                        logger.info(f"Successfully subscribed to: {subscriptions}")
                        break
                    elif (isinstance(msg, dict) and 
                          msg.get("ev") == "status" and 
                          "status" in msg):
                        logger.error(f"Subscription failed: {msg}")
                        return False
            elif (isinstance(sub_response, dict) and 
                  sub_response.get("ev") == "status" and 
                  sub_response.get("status") == "success"):
                subscribed = True
                logger.info(f"Successfully subscribed to: {subscriptions}")
            elif (isinstance(sub_response, dict) and 
                  sub_response.get("ev") == "status"):
                logger.error(f"Subscription failed: {sub_response}")
                return False
            
            if subscribed:
                self.subscriptions.extend(subscriptions)
                return True
            else:
                logger.error(f"No subscription success in response: {sub_response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
            return False
    
    async def unsubscribe(self, symbols: List[str], stream_types: List[str] = None):
        """Unsubscribe from specified symbols and stream types."""
        if not self.is_connected:
            return
        
        if stream_types is None:
            stream_types = ["T", "Q"]
        
        try:
            subscriptions = []
            for stream_type in stream_types:
                for symbol in symbols:
                    subscriptions.append(f"{stream_type}.{symbol}")
            
            unsubscribe_message = {
                "action": "unsubscribe", 
                "params": ",".join(subscriptions)
            }
            
            await self.websocket.send(json.dumps(unsubscribe_message))
            
            # Remove from local subscriptions list
            for sub in subscriptions:
                if sub in self.subscriptions:
                    self.subscriptions.remove(sub)
                    
            logger.info(f"Unsubscribed from: {subscriptions}")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe: {e}")
    
    async def start_streaming(self):
        """
        Start the main message processing loop.
        
        This method will continuously listen for messages from polygon.io
        and route them to appropriate handlers.
        """
        if not self.is_connected:
            logger.error("Must be connected before starting stream")
            return
        
        logger.info("Starting message processing loop")
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    
                    # Handle different message formats
                    if isinstance(data, list):
                        # Multiple messages in array
                        for msg in data:
                            await self._process_message(msg)
                    else:
                        # Single message
                        await self._process_message(data)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
            self.is_connected = False
    
    async def _process_message(self, message: Dict):
        """Process a single message from polygon.io."""
        # Ensure message is a dictionary
        if not isinstance(message, dict):
            logger.warning(f"Skipping non-dict message: {message}")
            return
            
        event_type = message.get("ev", "unknown")
        
        # Route message to appropriate handler
        if event_type in self.message_handlers:
            await self.message_handlers[event_type](message)
        else:
            logger.debug(f"No handler for message type: {event_type}")
    
    async def _handle_trade(self, message: Dict):
        """Handle trade messages (T)."""
        try:
            polygon_msg = PolygonMessage(
                event_type="trade",
                symbol=message.get("sym", ""),
                timestamp=datetime.fromtimestamp(message.get("t", 0) / 1000),
                data={
                    "price": message.get("p", 0.0),
                    "size": message.get("s", 0),
                    "exchange": message.get("x", ""),
                    "conditions": message.get("c", [])
                }
            )
            
            # Call custom trade handler if registered
            if hasattr(self, 'on_trade') and callable(self.on_trade):
                await self.on_trade(polygon_msg)
                
        except Exception as e:
            logger.error(f"Error handling trade message: {e}")
    
    async def _handle_quote(self, message: Dict):
        """Handle quote messages (Q)."""
        try:
            polygon_msg = PolygonMessage(
                event_type="quote",
                symbol=message.get("sym", ""),
                timestamp=datetime.fromtimestamp(message.get("t", 0) / 1000),
                data={
                    "bid_price": message.get("bp", 0.0),
                    "bid_size": message.get("bs", 0),
                    "ask_price": message.get("ap", 0.0),
                    "ask_size": message.get("as", 0),
                    "exchange": message.get("x", "")
                }
            )
            
            # Call custom quote handler if registered
            if hasattr(self, 'on_quote') and callable(self.on_quote):
                await self.on_quote(polygon_msg)
                
        except Exception as e:
            logger.error(f"Error handling quote message: {e}")
    
    async def _handle_aggregate(self, message: Dict):
        """Handle aggregate/bar messages (A)."""
        try:
            polygon_msg = PolygonMessage(
                event_type="aggregate",
                symbol=message.get("sym", ""),
                timestamp=datetime.fromtimestamp(message.get("s", 0) / 1000),
                data={
                    "open": message.get("o", 0.0),
                    "high": message.get("h", 0.0),
                    "low": message.get("l", 0.0),
                    "close": message.get("c", 0.0),
                    "volume": message.get("v", 0),
                    "vwap": message.get("vw", 0.0)
                }
            )
            
            # Call custom aggregate handler if registered
            if hasattr(self, 'on_aggregate') and callable(self.on_aggregate):
                await self.on_aggregate(polygon_msg)
                
        except Exception as e:
            logger.error(f"Error handling aggregate message: {e}")
    
    async def _handle_status(self, message: Dict):
        """Handle status messages."""
        logger.info(f"Status message: {message}")
    
    async def _handle_error(self, message: Dict):
        """Handle error messages."""
        logger.error(f"Error message from polygon.io: {message}")
    
    def add_handler(self, event_type: str, handler: Callable):
        """Add a custom message handler for specific event types."""
        self.message_handlers[event_type] = handler
    
    def set_trade_handler(self, handler: Callable):
        """Set a custom handler for trade messages."""
        self.on_trade = handler
    
    def set_quote_handler(self, handler: Callable):
        """Set a custom handler for quote messages."""
        self.on_quote = handler
    
    def set_aggregate_handler(self, handler: Callable):
        """Set a custom handler for aggregate messages."""
        self.on_aggregate = handler


@dataclass
class HistoricalDataPoint:
    """Represents a historical market data point."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    transactions: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "transactions": self.transactions
        }


@dataclass
class PolygonHistoricalDataFetcher:
    """
    Fetches historical market data from polygon.io REST API.
    
    This class handles fetching historical aggregates (bars) for stocks
    to provide better training data for reservoir computing models.
    """
    
    api_key: str
    base_url: str = "https://api.polygon.io"
    
    async def fetch_historical_aggregates(
        self,
        symbol: str,
        timespan: str = "minute",
        multiplier: int = 1,
        from_date: datetime = None,
        to_date: datetime = None,
        limit: int = 50000
    ) -> List[HistoricalDataPoint]:
        """
        Fetch historical aggregate data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timespan: Time unit ('minute', 'hour', 'day', 'week', 'month', 'quarter', 'year')
            multiplier: Size of timespan multiplier
            from_date: Start date (defaults to 30 days ago)
            to_date: End date (defaults to today)
            limit: Maximum number of data points to fetch
            
        Returns:
            List of HistoricalDataPoint objects
        """
        if from_date is None:
            from_date = datetime.now() - timedelta(days=30)
        if to_date is None:
            to_date = datetime.now()
        
        # Format dates for API
        from_str = from_date.strftime("%Y-%m-%d")
        to_str = to_date.strftime("%Y-%m-%d")
        
        url = (f"{self.base_url}/v2/aggs/ticker/{symbol}/range/"
               f"{multiplier}/{timespan}/{from_str}/{to_str}")
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": limit,
            "apikey": self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"API request failed: {response.status} - {await response.text()}")
                        return []
                    
                    data = await response.json()
                    
                    if data.get("status") != "OK":
                        logger.error(f"API returned error: {data}")
                        return []
                    
                    results = data.get("results", [])
                    logger.info(f"Fetched {len(results)} historical data points for {symbol}")
                    
                    # Convert to HistoricalDataPoint objects
                    historical_data = []
                    for item in results:
                        point = HistoricalDataPoint(
                            timestamp=datetime.fromtimestamp(item["t"] / 1000),
                            open=item["o"],
                            high=item["h"], 
                            low=item["l"],
                            close=item["c"],
                            volume=item["v"],
                            vwap=item.get("vw"),
                            transactions=item.get("n")
                        )
                        historical_data.append(point)
                    
                    return historical_data
                    
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    async def fetch_max_historical_data(
        self,
        symbol: str,
        timespan: str = "minute",
        multiplier: int = 1
    ) -> List[HistoricalDataPoint]:
        """
        Fetch maximum available historical data by making multiple API calls.
        
        This method attempts to fetch as much historical data as possible
        by working backwards from today in chunks.
        """
        all_data = []
        current_date = datetime.now()
        
        # Start with recent data and work backwards
        for days_back in [7, 30, 90, 365, 730]:  # 1 week, 1 month, 3 months, 1 year, 2 years
            try:
                from_date = current_date - timedelta(days=days_back)
                to_date = current_date - timedelta(days=days_back-7) if days_back > 7 else current_date
                
                logger.info(f"Fetching {symbol} data from {from_date.date()} to {to_date.date()}")
                
                chunk_data = await self.fetch_historical_aggregates(
                    symbol=symbol,
                    timespan=timespan,
                    multiplier=multiplier,
                    from_date=from_date,
                    to_date=to_date,
                    limit=50000
                )
                
                if chunk_data:
                    # Add to beginning of list to maintain chronological order
                    all_data = chunk_data + all_data
                    logger.info(f"Added {len(chunk_data)} points, total: {len(all_data)}")
                else:
                    logger.warning(f"No data returned for {symbol} from {from_date.date()}")
                
                # Rate limiting - don't overwhelm the API
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching chunk for {symbol}: {e}")
                continue
        
        # Remove duplicates and sort by timestamp
        seen_timestamps = set()
        unique_data = []
        for point in sorted(all_data, key=lambda x: x.timestamp):
            if point.timestamp not in seen_timestamps:
                unique_data.append(point)
                seen_timestamps.add(point.timestamp)
        
        logger.info(f"Fetched total of {len(unique_data)} unique historical data points for {symbol}")
        return unique_data


@dataclass
class PolygonFlatFilesClient:
    """
    Client for accessing Polygon.io Flat Files via S3.
    
    This provides efficient access to large amounts of historical data
    using Polygon.io's S3-compatible flat files endpoint.
    
    Requires separate S3 credentials from your Polygon.io dashboard.
    """
    
    s3_access_key: str
    s3_secret_key: str
    endpoint_url: str = "https://files.polygon.io"
    bucket_name: str = "flatfiles"
    
    def __post_init__(self):
        """Initialize the S3 client."""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.s3_access_key,
            aws_secret_access_key=self.s3_secret_key,
            endpoint_url=self.endpoint_url,
            config=Config(signature_version='s3v4')
        )
    
    async def list_available_files(self, prefix: str = "us_stocks_sip") -> List[str]:
        """
        List available flat files for a given prefix.
        
        Args:
            prefix: Data type prefix (e.g., 'us_stocks_sip', 'us_options_opra')
            
        Returns:
            List of available file keys
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            files = []
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        files.append(obj['Key'])
            
            logger.info(f"Found {len(files)} files with prefix {prefix}")
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    async def download_and_parse_file(self, file_key: str, symbol_filter: Optional[str] = None) -> List[HistoricalDataPoint]:
        """
        Download and parse a specific flat file.
        
        Args:
            file_key: S3 key of the file to download
            symbol_filter: Optional symbol to filter data (e.g., 'AAPL')
            
        Returns:
            List of HistoricalDataPoint objects
        """
        try:
            # Download file content
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            
            # Handle gzipped files
            if file_key.endswith('.gz'):
                content = gzip.decompress(response['Body'].read()).decode('utf-8')
            else:
                content = response['Body'].read().decode('utf-8')
            
            # Parse CSV content
            csv_reader = csv.DictReader(io.StringIO(content))
            data_points = []
            
            for row in csv_reader:
                # Filter by symbol if specified
                if symbol_filter and row.get('ticker') != symbol_filter:
                    continue
                
                # Parse the data according to flat files format
                # Format: ticker,volume,open,close,high,low,window_start,transactions
                try:
                    point = HistoricalDataPoint(
                        timestamp=datetime.fromtimestamp(int(row['window_start']) / 1_000_000_000),  # nanoseconds to seconds
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=int(row['volume']),
                        transactions=int(row.get('transactions', 0))
                    )
                    data_points.append(point)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid row: {row}, error: {e}")
                    continue
            
            logger.info(f"Parsed {len(data_points)} data points from {file_key}")
            return data_points
            
        except Exception as e:
            logger.error(f"Error downloading/parsing file {file_key}: {e}")
            return []
    
    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        data_type: str = "minute_aggs_v1"
    ) -> List[HistoricalDataPoint]:
        """
        Fetch historical data for a symbol across a date range.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date
            end_date: End date
            data_type: Type of data ('minute_aggs_v1', 'day_aggs_v1', etc.)
            
        Returns:
            List of HistoricalDataPoint objects
        """
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            # Build file path for this date
            year = current_date.year
            month = current_date.month
            date_str = current_date.strftime("%Y-%m-%d")
            
            file_key = f"us_stocks_sip/{data_type}/{year}/{month:02d}/{date_str}.csv.gz"
            
            logger.info(f"Fetching data from {file_key} for {symbol}")
            
            # Download and parse the file
            daily_data = await self.download_and_parse_file(file_key, symbol)
            all_data.extend(daily_data)
            
            # Move to next day
            current_date += timedelta(days=1)
            
            # Rate limiting to be respectful
            await asyncio.sleep(0.1)
        
        # Sort by timestamp
        all_data.sort(key=lambda x: x.timestamp)
        
        logger.info(f"Fetched total of {len(all_data)} data points for {symbol}")
        return all_data
    
    async def fetch_maximum_historical_data(
        self,
        symbol: str,
        data_type: str = "minute_aggs_v1",
        start_year: int = None,
        max_days_back: int = 3650  # 10 years default limit
    ) -> List[HistoricalDataPoint]:
        """
        Fetch maximum available historical data for a symbol.
        
        This method intelligently discovers available data by:
        1. Listing available files for the data type
        2. Finding files that contain the symbol
        3. Fetching data from the earliest available date
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            data_type: Type of data ('minute_aggs_v1', 'day_aggs_v1', etc.)
            start_year: Earliest year to search (defaults to 3 years ago)
            max_days_back: Maximum days to go back (safety limit)
            
        Returns:
            List of HistoricalDataPoint objects sorted by timestamp
        """
        if start_year is None:
            start_year = datetime.now().year - 10  # Default to 10 years ago
        
        logger.info(f"Discovering maximum available historical data for {symbol}")
        logger.info(f"Searching from {start_year} onwards with {data_type}")
        
        # List available files for this data type
        prefix = f"us_stocks_sip/{data_type}"
        available_files = await self.list_available_files(prefix)
        
        if not available_files:
            logger.warning(f"No files found for prefix {prefix}")
            return []
        
        # Filter files by symbol and date range
        symbol_files = []
        current_year = datetime.now().year
        
        for file_path in available_files:
            # Parse file path: us_stocks_sip/minute_aggs_v1/2024/01/2024-01-15.csv.gz
            parts = file_path.split('/')
            if len(parts) >= 5:
                try:
                    year = int(parts[2])
                    if start_year <= year <= current_year:
                        # Check if this file might contain our symbol
                        # We'll download and check rather than guess
                        symbol_files.append(file_path)
                except ValueError:
                    continue
        
        if not symbol_files:
            logger.warning(f"No files found for {symbol} in date range")
            return []
        
        # Sort files by date to process chronologically
        symbol_files.sort()
        
        logger.info(f"Found {len(symbol_files)} potential files to check")
        logger.info(f"Date range: {symbol_files[0]} to {symbol_files[-1]}")
        
        all_data = []
        files_processed = 0
        files_with_data = 0
        
        # Process files in batches to avoid overwhelming the system
        batch_size = 50  # Process 50 files at a time
        total_files = len(symbol_files)
        
        for i in range(0, len(symbol_files), batch_size):
            batch = symbol_files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}: files {i+1}-{min(i+batch_size, total_files)}")
            
            for file_path in batch:
                try:
                    # Download and parse file for this symbol
                    daily_data = await self.download_and_parse_file(file_path, symbol)
                    
                    if daily_data:
                        all_data.extend(daily_data)
                        files_with_data += 1
                        logger.debug(f"Found {len(daily_data)} points in {file_path}")
                    
                    files_processed += 1
                    
                    # Progress update every 10 files
                    if files_processed % 10 == 0:
                        logger.info(f"Progress: {files_processed}/{total_files} files processed, {len(all_data)} data points collected")
                    
                    # Rate limiting
                    await asyncio.sleep(0.05)  # 50ms between requests
                    
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    continue
            
            # Safety check - don't go back too far
            if len(all_data) > 0:
                earliest_date = min(point.timestamp for point in all_data)
                days_back = (datetime.now() - earliest_date).days
                if days_back > max_days_back:
                    logger.info(f"Reached maximum days limit ({max_days_back} days), stopping")
                    break
        
        # Sort by timestamp and remove duplicates
        seen_timestamps = set()
        unique_data = []
        for point in sorted(all_data, key=lambda x: x.timestamp):
            if point.timestamp not in seen_timestamps:
                unique_data.append(point)
                seen_timestamps.add(point.timestamp)
        
        if unique_data:
            earliest = unique_data[0].timestamp
            latest = unique_data[-1].timestamp
            total_days = (latest - earliest).days
            
            logger.info(f"✅ Successfully fetched maximum historical data for {symbol}")
            logger.info(f"📊 Total data points: {len(unique_data):,}")
            logger.info(f"📅 Date range: {earliest.date()} to {latest.date()} ({total_days} days)")
            logger.info(f"📁 Files processed: {files_processed}, Files with data: {files_with_data}")
            logger.info(f"📈 Average points per day: {len(unique_data) / max(1, total_days):.1f}")
        else:
            logger.warning(f"No data found for {symbol} in any files")
        
        return unique_data 