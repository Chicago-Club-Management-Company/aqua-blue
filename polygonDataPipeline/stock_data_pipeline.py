"""
Stock Data Pipeline for aqua-blue reservoir computing.

This module provides efficient stock data loading, streaming, and caching
using polygon.io websockets and REST API with support for S3 Flat Files.

Key features:
- Real-time websocket data streaming
- Historical data loading with intelligent caching
- S3 Flat Files support for large datasets
- Bulk data processing for efficiency
- Configurable data quality controls
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any

# Third-party imports
import numpy as np

# Local imports
from .polygon_client import (
    PolygonWebsocketClient, 
    PolygonHistoricalDataFetcher,
    PolygonFlatFilesClient,
    PolygonMessage,
    HistoricalDataPoint
)
from .data_processor import MarketDataProcessor, MarketDataPoint
from .data_cache import HistoricalDataCache

logger = logging.getLogger(__name__)


@dataclass
class StockDataPipeline:
    """
    Real-time stock data pipeline for data loading and streaming.
    
    This class provides stock data loading, caching, and real-time streaming
    without any prediction functionality. Focused purely on data preparation.
    
    Attributes:
        symbol (str): Stock symbol to process
        api_key (str): Polygon.io API key
        training_window (int): Number of data points for data buffers
        s3_access_key (Optional[str]): S3 access key for Flat Files (for large datasets)
        s3_secret_key (Optional[str]): S3 secret key for Flat Files (for large datasets)
    """
    
    symbol: str
    api_key: str
    training_window: int = 500
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    
    # Internal components
    polygon_client: Optional[PolygonWebsocketClient] = field(default=None, init=False)
    data_processor: Optional[MarketDataProcessor] = field(default=None, init=False)
    historical_fetcher: Optional[PolygonHistoricalDataFetcher] = field(default=None, init=False)
    flat_files_client: Optional[PolygonFlatFilesClient] = field(default=None, init=False)
    data_cache: Optional[HistoricalDataCache] = field(default=None, init=False)
    
    # State tracking
    data_count: int = field(default=0, init=False)
    is_running: bool = field(default=False, init=False)
    historical_data_loaded: bool = field(default=False, init=False)
    historical_data_range: Optional[Dict[str, datetime]] = field(default=None, init=False)
    
    # Callbacks
    on_data_update: Optional[Callable[[Dict], None]] = field(default=None, init=False)
    on_model_data_callback: Optional[Callable[[Dict], None]] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize the components."""
        self.polygon_client = PolygonWebsocketClient(api_key=self.api_key)
        self.data_processor = MarketDataProcessor(
            symbol=self.symbol,
            window_size=self.training_window * 2  # Keep extra data for stability
        )
        self.historical_fetcher = PolygonHistoricalDataFetcher(api_key=self.api_key)
        
        # Initialize flat files client if S3 credentials are provided
        if self.s3_access_key and self.s3_secret_key:
            self.flat_files_client = PolygonFlatFilesClient(
                s3_access_key=self.s3_access_key,
                s3_secret_key=self.s3_secret_key
            )
        
        # Initialize data cache
        self.data_cache = HistoricalDataCache()
        
        # Set up polygon client handlers
        self.polygon_client.set_trade_handler(self._handle_trade_message)
        self.polygon_client.set_quote_handler(self._handle_quote_message)
        self.polygon_client.set_aggregate_handler(self._handle_aggregate_message)
    
    async def start_data_stream(self):
        """Start the real-time data streaming."""
        if self.is_running:
            logger.warning("Data stream is already running")
            return
        
        logger.info(f"Starting data stream for {self.symbol}")
        
        try:
            # Connect to polygon.io
            if not await self.polygon_client.connect():
                raise ConnectionError("Failed to connect to polygon.io")
            
            # Subscribe to real-time data
            if not await self.polygon_client.subscribe([self.symbol], ["T", "Q"]):
                raise ConnectionError(f"Failed to subscribe to {self.symbol}")
            
            self.is_running = True
            logger.info(f"Successfully started data stream for {self.symbol}")
            
            # Start the message processing loop
            await self.polygon_client.start_streaming()
            
        except Exception as e:
            logger.error(f"Error starting data stream: {e}")
            await self.stop_data_stream()
            raise
    
    async def stop_data_stream(self):
        """Stop the data stream and cleanup."""
        if not self.is_running:
            return
        
        logger.info(f"Stopping data stream for {self.symbol}")
        
        try:
            if self.polygon_client:
                await self.polygon_client.disconnect()
            
            self.is_running = False
            logger.info(f"Stopped data stream for {self.symbol}")
            
        except Exception as e:
            logger.error(f"Error stopping data stream: {e}")
    
    async def _handle_trade_message(self, message: PolygonMessage):
        """Handle incoming trade messages."""
        try:
            self.data_processor.process_trade_message(message)
            self.data_count += 1
            
            # Call data update callback
            if self.on_data_update:
                stats = self.data_processor.get_buffer_stats()
                stats['data_count'] = self.data_count
                self.on_data_update(stats)
            
            # Check if we have enough data for model training
            if self.data_count >= self.training_window and self.on_model_data_callback:
                self._notify_model_data_ready()
                
        except Exception as e:
            logger.error(f"Error handling trade message: {e}")
    
    async def _handle_quote_message(self, message: PolygonMessage):
        """Handle incoming quote messages."""
        try:
            self.data_processor.process_quote_message(message)
            self.data_count += 1
            
            # Similar processing as trade messages but with quotes
            if self.on_data_update:
                stats = self.data_processor.get_buffer_stats()
                stats['data_count'] = self.data_count
                self.on_data_update(stats)
            
            # Check if we have enough data for model training
            if self.data_count >= self.training_window and self.on_model_data_callback:
                self._notify_model_data_ready()
                
        except Exception as e:
            logger.error(f"Error handling quote message: {e}")
    
    async def _handle_aggregate_message(self, message: PolygonMessage):
        """Handle incoming aggregate messages."""
        try:
            self.data_processor.process_aggregate_message(message)
            self.data_count += 1
            
            if self.on_data_update:
                stats = self.data_processor.get_buffer_stats()
                stats['data_count'] = self.data_count
                self.on_data_update(stats)
            
            # Check if we have enough data for model training
            if self.data_count >= self.training_window and self.on_model_data_callback:
                self._notify_model_data_ready()
                
        except Exception as e:
            logger.error(f"Error handling aggregate message: {e}")
    
    def _notify_model_data_ready(self):
        """Notify that model training data is ready."""
        if self.historical_data_range:
            total_days = (self.historical_data_range['end'] - self.historical_data_range['start']).days
            time_span = f"{total_days} days ({total_days/365:.1f} years)"
            data_quality = "High" if self.data_count > 100000 else "Medium" if self.data_count > 10000 else "Basic"
        else:
            time_span = "Real-time only"
            data_quality = "Basic"
        
        training_data_info = {
            'data_count': self.data_count,
            'time_span': time_span,
            'data_quality': data_quality,
            'training_window': self.training_window,
            'has_historical': self.historical_data_loaded
        }
        
        self.on_model_data_callback(training_data_info)
    
    def set_data_update_callback(self, callback: Callable[[Dict], None]):
        """Set callback function for when new data is received."""
        self.on_data_update = callback
    
    def set_model_data_callback(self, callback: Callable[[Dict], None]):
        """Set callback function for when model training data is ready."""
        self.on_model_data_callback = callback
    
    def get_model_status(self) -> Dict:
        """Get current status of the data pipeline."""
        status = {
            "is_running": self.is_running,
            "model_trained": self.data_count >= self.training_window,  # Data ready for training
            "data_count": self.data_count,
            "buffer_stats": self.data_processor.get_buffer_stats() if self.data_processor else {}
        }
        
        # Add historical data information
        if self.historical_data_loaded and self.historical_data_range:
            status["historical_data"] = {
                "loaded": True,
                "start_date": self.historical_data_range['start'].isoformat(),
                "end_date": self.historical_data_range['end'].isoformat(),
                "total_days": (self.historical_data_range['end'] - self.historical_data_range['start']).days,
                "total_years": (self.historical_data_range['end'] - self.historical_data_range['start']).days / 365,
                "data_points": self.historical_data_range['count'],
                "timespan": self.historical_data_range.get('timespan', 'unknown'),
                "points_per_day": self.historical_data_range['count'] / max(1, (self.historical_data_range['end'] - self.historical_data_range['start']).days)
            }
        else:
            status["historical_data"] = {"loaded": False}
        
        return status
    
    def get_historical_data_info(self) -> Dict:
        """
        Get detailed information about loaded historical data.
        
        Returns:
            Dictionary with historical data information or empty if no data loaded
        """
        if not self.historical_data_loaded or not self.historical_data_range:
            return {
                "status": "No historical data loaded",
                "recommendation": "Call load_historical_data() or load_maximum_historical_data() to load data"
            }
        
        start_date = self.historical_data_range['start']
        end_date = self.historical_data_range['end']
        total_days = (end_date - start_date).days
        total_years = total_days / 365
        
        return {
            "status": "Historical data loaded",
            "symbol": self.symbol,
            "date_range": {
                "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S"),
                "total_days": total_days,
                "total_years": round(total_years, 2)
            },
            "data_metrics": {
                "total_points": self.historical_data_range['count'],
                "timespan": self.historical_data_range.get('timespan', 'unknown'),
                "points_per_day": round(self.historical_data_range['count'] / max(1, total_days), 1),
                "data_density": "High" if self.historical_data_range['count'] / max(1, total_days) > 100 else "Medium" if self.historical_data_range['count'] / max(1, total_days) > 10 else "Low"
            },
            "coverage_analysis": {
                "years_back_from_now": round((datetime.now() - start_date).days / 365, 2),
                "is_recent": (datetime.now() - end_date).days < 7,
                "data_freshness": "Recent" if (datetime.now() - end_date).days < 7 else f"{(datetime.now() - end_date).days} days old"
            },
            "model_readiness": {
                "sufficient_for_training": self.historical_data_range['count'] >= self.training_window,
                "training_window_size": self.training_window,
                "data_ready": self.data_count >= self.training_window
            },
            "cache_info": {
                "cache_status": "Available" if self.data_cache else "Not available"
            }
        }
    
    def get_cache_info(self, timespan: str = "minute") -> Dict:
        """
        Get information about cached data for this symbol.
        
        Args:
            timespan: Data timespan to check
            
        Returns:
            Dictionary with cache information
        """
        cache_info = self.data_cache.get_cache_info(self.symbol, timespan)
        
        if not cache_info:
            return {
                "status": "No cached data",
                "symbol": self.symbol,
                "timespan": timespan,
                "recommendation": "Run load_historical_data() to fetch and cache data"
            }
        
        total_days = (cache_info.end_date - cache_info.start_date).days
        
        return {
            "status": "Cached data available",
            "symbol": self.symbol,
            "timespan": timespan,
            "cache_file": cache_info.file_path,
            "date_range": {
                "start": cache_info.start_date.strftime("%Y-%m-%d %H:%M:%S"),
                "end": cache_info.end_date.strftime("%Y-%m-%d %H:%M:%S"),
                "total_days": total_days,
                "total_years": round(total_days / 365, 2)
            },
            "data_metrics": {
                "total_points": cache_info.total_points,
                "file_size_mb": cache_info.file_size_mb,
                "points_per_day": round(cache_info.total_points / max(1, total_days), 1)
            },
            "file_info": {
                "last_modified": cache_info.last_modified.strftime("%Y-%m-%d %H:%M:%S"),
                "age_days": (datetime.now() - cache_info.last_modified).days
            }
        }
    
    def clear_cache(self, timespan: str = "minute") -> bool:
        """
        Clear cached data for this symbol.
        
        Args:
            timespan: Data timespan to clear
            
        Returns:
            True if cleared successfully
        """
        return self.data_cache.clear_cache(self.symbol, timespan)
    
    def get_all_cache_stats(self) -> Dict:
        """Get statistics about all cached data."""
        return self.data_cache.get_cache_stats()
    
    async def load_historical_data(self, days_back: int = 1825, timespan: str = "minute") -> bool:
        """
        Load historical data to improve data quality.
        
        Default is now 5 years (1825 days) minimum for optimal data coverage.
        Uses intelligent caching to avoid re-downloading existing data.
        
        Automatically chooses between REST API and Flat Files based on:
        - Amount of data requested (large datasets use Flat Files if available)
        - Availability of S3 credentials
        - What data we already have cached (avoids duplicates)
        
        Args:
            days_back: Number of days of historical data to fetch (default: 1825 = 5 years, minimum enforced)
            timespan: Timespan for historical data ('minute', 'hour', 'day')
            
        Returns:
            True if historical data was loaded successfully
        """
        # Enforce minimum of 5 years for optimal data coverage
        min_days = 1825  # 5 years
        if days_back < min_days:
            logger.warning(f"⚠️  Requested {days_back} days, enforcing minimum of {min_days} days (5 years) for optimal data coverage")
            days_back = min_days
        
        logger.info(f"📈 Loading historical data for {self.symbol} ({days_back} days = ~{days_back/365:.1f} years)")
        logger.info(f"🎯 Using minimum 5-year requirement for optimal data quality")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Check cache first
        cache_info = self.data_cache.get_cache_info(self.symbol, timespan)
        
        if cache_info:
            logger.info(f"📂 Found cached data for {self.symbol}: "
                       f"{cache_info.total_points:,} points from {cache_info.start_date.date()} to {cache_info.end_date.date()}")
            
            # Check if cached data covers our requested range
            if (cache_info.start_date <= start_date and 
                cache_info.end_date >= end_date - timedelta(days=1)):  # Allow 1 day buffer for recent data
                
                logger.info(f"✅ Cached data covers requested range - loading from cache")
                cached_data = self.data_cache.load_historical_data(
                    self.symbol, timespan, start_date, end_date
                )
                
                if cached_data:
                    return await self._process_cached_data(cached_data, timespan)
        
        # Determine what data we need to fetch
        missing_ranges = self.data_cache.get_missing_date_ranges(
            self.symbol, timespan, start_date, end_date
        )
        
        if not missing_ranges:
            logger.info(f"✅ All requested data already cached for {self.symbol}")
            # Load from cache
            cached_data = self.data_cache.load_historical_data(
                self.symbol, timespan, start_date, end_date
            )
            if cached_data:
                return await self._process_cached_data(cached_data, timespan)
        
        # Need to fetch missing data
        logger.info(f"📡 Need to fetch {len(missing_ranges)} missing date range(s) for {self.symbol}")
        
        all_new_data = []
        
        for range_start, range_end in missing_ranges:
            range_days = (range_end - range_start).days
            logger.info(f"📊 Fetching data from {range_start.date()} to {range_end.date()} ({range_days} days)")
            
            # Choose method based on data size and availability
            use_flat_files = (
                self.flat_files_client is not None and  # S3 credentials available
                range_days > 90 and  # Large dataset
                timespan == "minute"  # Minute data works well with flat files
            )
            
            try:
                if use_flat_files:
                    logger.info("🚀 Using Flat Files for large dataset - much more efficient!")
                    range_data = await self._load_range_via_flat_files(range_start, range_end, timespan)
                else:
                    logger.info("📡 Using REST API for historical data")
                    range_data = await self._load_range_via_rest_api(range_start, range_end, timespan)
                
                if range_data:
                    all_new_data.extend(range_data)
                    logger.info(f"✅ Fetched {len(range_data):,} data points for range")
                else:
                    logger.warning(f"❌ No data returned for range {range_start.date()} to {range_end.date()}")
                
            except Exception as e:
                logger.error(f"❌ Error fetching data for range {range_start.date()} to {range_end.date()}: {e}")
                continue
        
        # Update cache with new data
        if all_new_data:
            logger.info(f"💾 Saving {len(all_new_data):,} new data points to cache")
            success = self.data_cache.update_cache(self.symbol, all_new_data, timespan)
            if not success:
                logger.warning("⚠️  Failed to save data to cache, but continuing with processing")
        
        # Load all data (cached + new) for processing
        all_data = self.data_cache.load_historical_data(
            self.symbol, timespan, start_date, end_date
        )
        
        if not all_data:
            logger.error(f"❌ No historical data available for {self.symbol}")
            return False
        
        # Process the data
        return await self._process_cached_data(all_data, timespan)
    
    async def _process_cached_data(self, historical_data: List[HistoricalDataPoint], timespan: str) -> bool:
        """Process historical data that was loaded from cache or API."""
        try:
            # Convert historical data to market data format and process in bulk  
            logger.info(f"📊 Processing {len(historical_data):,} historical data points efficiently...")
            
            # Convert to MarketDataPoint format
            market_data_points = []
            for point in historical_data:
                market_point = MarketDataPoint(
                    timestamp=point.timestamp,
                    symbol=self.symbol,
                    price=point.close,
                    volume=point.volume,
                    open=point.open,
                    high=point.high,
                    low=point.low
                )
                market_data_points.append(market_point)
            
            # Use bulk processing for efficiency
            target_interval = 60.0 if timespan == "minute" else 3600.0  # 1 min or 1 hour
            success = self.data_processor.process_historical_data_bulk(
                market_data_points, 
                target_interval_seconds=target_interval
            )
            
            if not success:
                logger.error("Failed to process historical data in bulk")
                return False
            
            # Update data count
            self.data_count = len(market_data_points)
            
            # Track what historical data we now have
            all_timestamps = [point.timestamp for point in historical_data]
            self.historical_data_range = {
                'start': min(all_timestamps),
                'end': max(all_timestamps),
                'count': len(historical_data),
                'timespan': timespan
            }
            
            self.historical_data_loaded = True
            
            # Show statistics
            total_days = (self.historical_data_range['end'] - self.historical_data_range['start']).days
            logger.info(f"✅ Loaded {len(historical_data):,} historical data points for {self.symbol}")
            logger.info(f"📅 Date range: {self.historical_data_range['start'].date()} to {self.historical_data_range['end'].date()} ({total_days} days)")
            logger.info(f"📊 Average points per day: {len(historical_data) / max(1, total_days):.1f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing cached data: {e}")
            return False
    
    async def _load_range_via_rest_api(self, start_date: datetime, end_date: datetime, timespan: str) -> List[HistoricalDataPoint]:
        """Load a specific date range via REST API."""
        days_back = (end_date - start_date).days
        
        if days_back > 365:
            # For large date ranges, use the max fetcher
            return await self.historical_fetcher.fetch_max_historical_data(
                symbol=self.symbol,
                timespan=timespan
            )
        else:
            # For smaller ranges, use standard fetcher
            return await self.historical_fetcher.fetch_historical_aggregates(
                symbol=self.symbol,
                timespan=timespan,
                from_date=start_date,
                to_date=end_date
            )
    
    async def _load_range_via_flat_files(self, start_date: datetime, end_date: datetime, timespan: str) -> List[HistoricalDataPoint]:
        """Load a specific date range via Flat Files."""
        if not self.flat_files_client:
            raise ValueError("Flat Files client not initialized - need S3 credentials")
        
        # Map timespan to flat files data type
        data_type_map = {
            "minute": "minute_aggs_v1",
            "hour": "hour_aggs_v1", 
            "day": "day_aggs_v1"
        }
        
        data_type = data_type_map.get(timespan, "minute_aggs_v1")
        
        logger.info(f"📁 Fetching {self.symbol} via Flat Files from {start_date.date()} to {end_date.date()}")
        
        return await self.flat_files_client.fetch_historical_data(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            data_type=data_type
        )
    
    async def load_maximum_historical_data(self, timespan: str = "minute", max_years_back: int = 10) -> bool:
        """
        Load maximum available historical data for the symbol.
        
        This method will attempt to fetch as much historical data as possible
        using the most efficient method available (prioritizes Flat Files for large datasets).
        Enforces a minimum of 5 years for optimal data coverage.
        
        Args:
            timespan: Timespan for historical data ('minute', 'hour', 'day')
            max_years_back: Maximum years to search back (default: 10 years, minimum: 5 years)
            
        Returns:
            True if historical data was loaded successfully
        """
        # Enforce minimum of 5 years
        min_years = 5
        if max_years_back < min_years:
            logger.warning(f"⚠️  Requested {max_years_back} years, enforcing minimum of {min_years} years for optimal data coverage")
            max_years_back = min_years
        
        # Check if we already have sufficient historical data
        if self.historical_data_loaded and self.historical_data_range:
            existing_days = (datetime.now() - self.historical_data_range['start']).days
            requested_days = max_years_back * 365
            
            if existing_days >= max(requested_days * 0.9, min_years * 365):  # 90% coverage or minimum 5 years
                logger.info(f"✅ Sufficient historical data already loaded for {self.symbol}")
                logger.info(f"📊 Current coverage: {existing_days} days (~{existing_days/365:.1f} years)")
                if existing_days >= min_years * 365:
                    logger.info(f"🎯 Meets minimum 5-year requirement")
                return True
        
        logger.info(f"🚀 Loading MAXIMUM available historical data for {self.symbol}")
        logger.info(f"📊 Timespan: {timespan}, Max years back: {max_years_back} (minimum: {min_years})")
        logger.info(f"🎯 Enforcing minimum 5-year requirement for optimal data quality")
        
        # Prefer Flat Files for maximum data if available
        use_flat_files = (
            self.flat_files_client is not None and 
            timespan == "minute"  # Flat files work best with minute data
        )
        
        try:
            if use_flat_files:
                logger.info("🎯 Using Flat Files for maximum historical data - much more efficient!")
                historical_data = await self._load_maximum_via_flat_files(timespan, max_years_back)
            else:
                logger.info("📡 Using REST API for maximum historical data")
                # For REST API, use our existing max fetcher with extended range
                historical_data = await self._load_maximum_via_rest_api(timespan, max_years_back)
            
            if not historical_data:
                logger.warning(f"No historical data returned for {self.symbol}")
                return False
            
            # Filter out data we already have to avoid duplicates
            if self.historical_data_range and self.historical_data_range.get('start'):
                existing_start = self.historical_data_range['start']
                # Only keep data older than what we already have, plus a small overlap for continuity
                overlap_buffer = timedelta(hours=1)  # 1 hour overlap
                cutoff_time = existing_start - overlap_buffer
                
                new_data = [point for point in historical_data if point.timestamp < cutoff_time]
                if new_data:
                    logger.info(f"📊 Found {len(new_data):,} new historical data points "
                              f"(filtered {len(historical_data) - len(new_data):,} duplicates)")
                    # Sort to maintain chronological order: new data + existing data
                    historical_data = sorted(new_data, key=lambda x: x.timestamp)
                else:
                    logger.info(f"✅ All maximum historical data already exists - no new data to add")
                    return True
            
            # Convert historical data to market data format and process in bulk
            logger.info(f"📊 Processing {len(historical_data):,} historical data points efficiently...")
            
            # Convert to MarketDataPoint format
            market_data_points = []
            for point in historical_data:
                market_point = MarketDataPoint(
                    timestamp=point.timestamp,
                    symbol=self.symbol,
                    price=point.close,
                    volume=point.volume,
                    open=point.open,
                    high=point.high,
                    low=point.low
                )
                market_data_points.append(market_point)
            
            # Use bulk processing for efficiency
            target_interval = 60.0 if timespan == "minute" else 3600.0  # 1 min or 1 hour
            success = self.data_processor.process_historical_data_bulk(
                market_data_points, 
                target_interval_seconds=target_interval
            )
            
            if not success:
                logger.error("Failed to process historical data in bulk")
                return False
            
            logger.info(f"💾 Saving {len(historical_data):,} maximum historical data points to cache")
            cache_success = self.data_cache.update_cache(self.symbol, historical_data, timespan)
            if cache_success:
                logger.info(f"✅ Successfully saved maximum historical data to cache")
            else:
                logger.warning("⚠️  Failed to save maximum data to cache, but continuing with processing")
            
            # Update data count
            self.data_count = len(market_data_points)
            
            # Update historical data range tracking
            all_timestamps = [point.timestamp for point in historical_data]
            if self.historical_data_range:
                # Extend existing range
                self.historical_data_range['start'] = min(self.historical_data_range['start'], min(all_timestamps))
                self.historical_data_range['end'] = max(self.historical_data_range['end'], max(all_timestamps))
                self.historical_data_range['count'] += len(historical_data)
            else:
                # Create new range
                self.historical_data_range = {
                    'start': min(all_timestamps),
                    'end': max(all_timestamps),
                    'count': len(historical_data),
                    'timespan': timespan
                }
            
            self.historical_data_loaded = True
            
            # Show final statistics
            earliest = self.historical_data_range['start']
            latest = self.historical_data_range['end']
            total_days = (latest - earliest).days
            
            logger.info(f"✅ Successfully loaded MAXIMUM historical data for {self.symbol}")
            logger.info(f"📊 Total data points: {self.historical_data_range['count']:,}")
            logger.info(f"📅 Date range: {earliest.date()} to {latest.date()} ({total_days} days = {total_days/365:.1f} years)")
            logger.info(f"📈 Average points per day: {self.historical_data_range['count'] / max(1, total_days):.1f}")
            logger.info(f"🎯 Data ready for timeseries model training!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading maximum historical data: {e}")
            return False
    
    async def _load_maximum_via_flat_files(self, timespan: str, max_years_back: int) -> List[HistoricalDataPoint]:
        """Load maximum historical data via Flat Files."""
        if not self.flat_files_client:
            raise ValueError("Flat Files client not initialized - need S3 credentials")
        
        # Map timespan to flat files data type
        data_type_map = {
            "minute": "minute_aggs_v1",
            "hour": "hour_aggs_v1", 
            "day": "day_aggs_v1"
        }
        
        data_type = data_type_map.get(timespan, "minute_aggs_v1")
        start_year = datetime.now().year - max_years_back
        max_days_back = max_years_back * 365
        
        logger.info(f"📁 Fetching maximum {self.symbol} data via Flat Files (starting from {start_year})")
        
        return await self.flat_files_client.fetch_maximum_historical_data(
            symbol=self.symbol,
            data_type=data_type,
            start_year=start_year,
            max_days_back=max_days_back
        )
    
    async def _load_maximum_via_rest_api(self, timespan: str, max_years_back: int) -> List[HistoricalDataPoint]:
        """Load maximum historical data via REST API (with extended range)."""
        logger.info(f"📡 Fetching maximum {self.symbol} data via REST API")
        
        # For REST API, we'll try multiple approaches to get maximum data
        all_data = []
        current_date = datetime.now()
        
        # 1. First get recent data (last 7 days) to ensure we have current data
        recent_start = current_date - timedelta(days=7)
        logger.info(f"🕐 Fetching recent data: {recent_start.date()} to {current_date.date()}")
        try:
            recent_data = await self.historical_fetcher.fetch_historical_aggregates(
                symbol=self.symbol,
                timespan=timespan,
                from_date=recent_start,
                to_date=current_date,
                limit=50000
            )
            if recent_data:
                all_data.extend(recent_data)
                logger.info(f"✅ Added {len(recent_data)} recent data points")
        except Exception as e:
            logger.warning(f"Could not fetch recent data: {e}")
        
        # 2. Try our existing max fetcher for bulk data
        logger.info("📊 Trying existing max fetcher for bulk historical data...")
        try:
            max_data = await self.historical_fetcher.fetch_max_historical_data(
                symbol=self.symbol,
                timespan=timespan
            )
            if max_data:
                all_data.extend(max_data)
                logger.info(f"✅ Added {len(max_data)} points from max fetcher")
        except Exception as e:
            logger.warning(f"Max fetcher failed: {e}")
        
        # 3. Fill gaps with additional chunks going backwards
        logger.info("🔄 Fetching additional historical chunks...")
        for months_back in range(1, max_years_back * 12, 3):  # Start from 1 month, not 6
            try:
                from_date = current_date - timedelta(days=months_back * 30 + 90)  # 3 months back
                to_date = current_date - timedelta(days=months_back * 30)
                
                logger.info(f"📊 Fetching chunk: {from_date.date()} to {to_date.date()}")
                
                chunk_data = await self.historical_fetcher.fetch_historical_aggregates(
                    symbol=self.symbol,
                    timespan=timespan,
                    from_date=from_date,
                    to_date=to_date,
                    limit=50000
                )
                
                if chunk_data:
                    # Add to beginning to maintain chronological order
                    all_data = chunk_data + all_data
                    logger.info(f"✅ Added {len(chunk_data)} points, total: {len(all_data)}")
                else:
                    logger.info(f"No data for period {from_date.date()} to {to_date.date()}")
                
                # Rate limiting
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Error fetching chunk for {from_date.date()}: {e}")
                continue
        
        # Remove duplicates and sort
        seen_timestamps = set()
        unique_data = []
        for point in sorted(all_data, key=lambda x: x.timestamp):
            if point.timestamp not in seen_timestamps:
                unique_data.append(point)
                seen_timestamps.add(point.timestamp)
        
        if unique_data:
            earliest = min(point.timestamp for point in unique_data)
            latest = max(point.timestamp for point in unique_data)
            logger.info(f"📈 REST API fetched {len(unique_data)} unique data points")
            logger.info(f"📅 Date range: {earliest.date()} to {latest.date()}")
        
        return unique_data