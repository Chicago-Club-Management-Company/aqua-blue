"""
Market data processor for converting polygon.io streams to aqua-blue TimeSeries.

This module handles the conversion of real-time market data from polygon.io
into structured TimeSeries objects that can be used with aqua-blue models.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Deque
from collections import deque
from dataclasses import dataclass, field

import sys
import os

# Add the parent directory to the path to import aqua_blue
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aqua_blue

from .polygon_client import PolygonMessage

logger = logging.getLogger(__name__)


@dataclass
class MarketDataPoint:
    """Represents a single market data point."""
    timestamp: datetime
    symbol: str
    price: float
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None


@dataclass 
class MarketDataProcessor:
    """
    Processes real-time market data and maintains rolling time series.
    
    This processor accumulates market data from polygon.io streams and
    converts it into aqua-blue TimeSeries objects for model training and data processing.
    
    Attributes:
        symbol (str): The stock symbol being processed
        window_size (int): Maximum number of data points to keep in memory
        time_interval (timedelta): Expected time interval between data points
        data_buffer (Deque): Rolling buffer of market data points
        current_series (Optional[TimeSeries]): Current time series object
    """
    
    symbol: str
    window_size: int = 1000
    time_interval: timedelta = timedelta(seconds=1)
    data_buffer: Deque[MarketDataPoint] = field(default_factory=lambda: deque(maxlen=1000))
    current_series: Optional[aqua_blue.time_series.TimeSeries] = field(default=None, init=False)
    last_processed_time: Optional[datetime] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize the data buffer with the specified window size."""
        self.data_buffer = deque(maxlen=self.window_size)
    
    def process_trade_message(self, message: PolygonMessage):
        """
        Process a trade message from polygon.io.
        
        Args:
            message (PolygonMessage): Trade message containing price and volume data
        """
        if message.symbol != self.symbol:
            return
            
        try:
            data_point = MarketDataPoint(
                timestamp=message.timestamp,
                symbol=message.symbol,
                price=message.data["price"],
                volume=message.data.get("size", 0)
            )
            
            self._add_data_point(data_point)
            
        except Exception as e:
            logger.error(f"Error processing trade message: {e}")
    
    def process_quote_message(self, message: PolygonMessage):
        """
        Process a quote message from polygon.io.
        
        Args:
            message (PolygonMessage): Quote message containing bid/ask data
        """
        if message.symbol != self.symbol:
            return
            
        try:
            # Use mid-price as the main price for time series
            bid_price = message.data["bid_price"]
            ask_price = message.data["ask_price"]
            mid_price = (bid_price + ask_price) / 2.0 if bid_price > 0 and ask_price > 0 else 0.0
            
            data_point = MarketDataPoint(
                timestamp=message.timestamp,
                symbol=message.symbol,
                price=mid_price,
                bid=bid_price,
                ask=ask_price
            )
            
            self._add_data_point(data_point)
            
        except Exception as e:
            logger.error(f"Error processing quote message: {e}")
    
    def process_aggregate_message(self, message: PolygonMessage):
        """
        Process an aggregate/bar message from polygon.io.
        
        Args:
            message (PolygonMessage): Aggregate message containing OHLCV data
        """
        if message.symbol != self.symbol:
            return
            
        try:
            data_point = MarketDataPoint(
                timestamp=message.timestamp,
                symbol=message.symbol,
                price=message.data["close"],
                volume=message.data.get("volume", 0),
                open=message.data.get("open", 0.0),
                high=message.data.get("high", 0.0),
                low=message.data.get("low", 0.0)
            )
            
            self._add_data_point(data_point)
            
        except Exception as e:
            logger.error(f"Error processing aggregate message: {e}")
    
    def _add_data_point(self, data_point: MarketDataPoint):
        """
        Add a data point to the buffer and update the time series.
        
        Args:
            data_point (MarketDataPoint): New market data point to add
        """
        # Skip if price is invalid
        if data_point.price <= 0:
            return
            
        # Add to buffer (automatically removes oldest if at capacity)
        self.data_buffer.append(data_point)
        self.last_processed_time = data_point.timestamp
        
        # Update the time series if we have enough data
        if len(self.data_buffer) >= 10:  # Minimum 10 points for meaningful series
            self._update_time_series()
    
    def _update_time_series(self):
        """Update the current TimeSeries object from the data buffer."""
        try:
            # Convert buffer to lists for TimeSeries creation
            timestamps = []
            prices = []
            volumes = []
            
            # Sort data points by timestamp to ensure chronological order
            sorted_data = sorted(self.data_buffer, key=lambda x: x.timestamp)
            
            for point in sorted_data:
                timestamps.append(point.timestamp)
                prices.append(point.price)
                volumes.append(point.volume if point.volume is not None else 0)
            
            if len(timestamps) < 2:
                return
            
            # Create uniformly spaced timestamps for aqua-blue compatibility
            start_time = timestamps[0]
            end_time = timestamps[-1]
            total_duration = end_time - start_time
            
            # Calculate a reasonable time interval based on the data
            if len(timestamps) > 1:
                # Use average interval, but ensure it's reasonable for market data
                avg_interval_seconds = total_duration.total_seconds() / (len(timestamps) - 1)
                # Clamp to reasonable values (1 second to 5 minutes)
                interval_seconds = max(1, min(300, avg_interval_seconds))
            else:
                interval_seconds = 60  # Default to 1 minute
            
            interval = timedelta(seconds=interval_seconds)
            
            # Create uniform timestamp array
            uniform_timestamps = []
            current_time = start_time
            while current_time <= end_time:
                uniform_timestamps.append(current_time)
                current_time += interval
            
            # Ensure we have at least 2 points
            if len(uniform_timestamps) < 2:
                # Add one more point if needed
                uniform_timestamps.append(start_time + interval)
            
            # Convert to numpy arrays for interpolation
            original_times = np.array([t.timestamp() for t in timestamps])
            uniform_times = np.array([t.timestamp() for t in uniform_timestamps])
            
            # Interpolate data to uniform timestamps
            uniform_prices = np.interp(uniform_times, original_times, prices)
            uniform_volumes = np.interp(uniform_times, original_times, volumes)
            
            # Create 2D array: [price, volume] for each timestep
            dependent_variables = np.column_stack([uniform_prices, uniform_volumes])
            
            # Create TimeSeries object with uniform timestamps
            self.current_series = aqua_blue.time_series.TimeSeries(
                dependent_variable=dependent_variables,
                times=uniform_timestamps
            )
            
            logger.debug(f"Updated time series for {self.symbol} with {len(uniform_timestamps)} uniform points "
                        f"(interval: {interval_seconds:.1f}s)")
            
        except Exception as e:
            logger.error(f"Error updating time series: {e}")
            self.current_series = None
    
    def get_latest_series(self, min_length: int = 50) -> Optional[aqua_blue.time_series.TimeSeries]:
        """
        Get the latest time series if it meets minimum length requirements.
        
        Args:
            min_length (int): Minimum number of data points required
            
        Returns:
            Optional[TimeSeries]: Current time series or None if insufficient data
        """
        if (self.current_series is not None and 
            len(self.current_series.dependent_variable) >= min_length):
            return self.current_series
        return None
    
    def get_latest_slice(self, length: int) -> Optional[aqua_blue.time_series.TimeSeries]:
        """
        Get the most recent slice of the time series.
        
        Args:
            length (int): Number of recent data points to include
            
        Returns:
            Optional[TimeSeries]: Sliced time series or None if insufficient data
        """
        if self.current_series is None or len(self.current_series.dependent_variable) < length:
            return None
        
        # Return the last 'length' data points
        return self.current_series[-length:]
    
    def clear_buffer(self):
        """Clear the data buffer and reset the time series."""
        self.data_buffer.clear()
        self.current_series = None
        self.last_processed_time = None
        logger.info(f"Cleared data buffer for {self.symbol}")
    
    def get_buffer_stats(self) -> Dict[str, any]:
        """
        Get statistics about the current data buffer.
        
        Returns:
            Dict: Statistics including buffer size, time range, etc.
        """
        if not self.data_buffer:
            return {"buffer_size": 0, "time_range": None, "latest_price": None}
        
        sorted_data = sorted(self.data_buffer, key=lambda x: x.timestamp)
        time_range = sorted_data[-1].timestamp - sorted_data[0].timestamp
        
        return {
            "buffer_size": len(self.data_buffer),
            "time_range": time_range.total_seconds(),
            "latest_price": sorted_data[-1].price,
            "latest_time": sorted_data[-1].timestamp,
            "price_range": {
                "min": min(point.price for point in self.data_buffer),
                "max": max(point.price for point in self.data_buffer),
                "avg": sum(point.price for point in self.data_buffer) / len(self.data_buffer)
            }
        }
    
    def resample_to_interval(self, target_interval: timedelta) -> Optional[aqua_blue.time_series.TimeSeries]:
        """
        Resample the current time series to a specific time interval.
        
        Args:
            target_interval (timedelta): Target time interval between points
            
        Returns:
            Optional[TimeSeries]: Resampled time series or None if insufficient data
        """
        if self.current_series is None or len(self.current_series.dependent_variable) < 2:
            return None
        
        try:
            # Get the time range
            start_time = self.current_series.times[0]
            end_time = self.current_series.times[-1]
            total_duration = end_time - start_time
            
            # Ensure we have a reasonable duration
            if total_duration.total_seconds() < target_interval.total_seconds():
                # If duration is too short, return the original series
                return self.current_series
            
            # Create new timestamp array with target interval
            new_timestamps = []
            current_time = start_time
            while current_time <= end_time:
                new_timestamps.append(current_time)
                current_time += target_interval
            
            if len(new_timestamps) < 2:
                # If we can't get at least 2 points, add one more
                new_timestamps.append(start_time + target_interval)
            
            # Convert to numpy arrays for interpolation
            original_times = np.array([t.timestamp() for t in self.current_series.times])
            new_times = np.array([t.timestamp() for t in new_timestamps])
            
            # Interpolate prices and volumes to new timestamps
            new_prices = np.interp(new_times, original_times, self.current_series.dependent_variable[:, 0])
            new_volumes = np.interp(new_times, original_times, self.current_series.dependent_variable[:, 1])
            
            new_dependent_vars = np.column_stack([new_prices, new_volumes])
            
            return aqua_blue.time_series.TimeSeries(
                dependent_variable=new_dependent_vars,
                times=new_timestamps
            )
            
        except Exception as e:
            logger.error(f"Error resampling time series: {e}")
            return None
    
    def process_historical_data_bulk(self, historical_points: List[MarketDataPoint], 
                                   target_interval_seconds: float = 60.0) -> bool:
        """
        Process a large batch of historical data points efficiently.
        
        This method is optimized for loading large amounts of historical data
        and creating a uniform time series suitable for aqua-blue models.
        Following the patterns used in aqua-blue examples for maximum compatibility.
        
        Args:
            historical_points (List[MarketDataPoint]): List of historical data points
            target_interval_seconds (float): Target interval in seconds between points
            
        Returns:
            bool: True if processing was successful
        """
        try:
            if not historical_points:
                return False
            
            logger.info(f"📊 Processing {len(historical_points):,} historical data points in bulk")
            
            # Sort by timestamp to ensure chronological order (like in examples)
            sorted_points = sorted(historical_points, key=lambda x: x.timestamp)
            
            # Extract data and filter invalid prices
            valid_points = []
            for point in sorted_points:
                if point.price > 0 and point.volume is not None and point.volume >= 0:
                    valid_points.append(point)
            
            if len(valid_points) < 10:  # Need minimum data for meaningful series
                logger.warning(f"Insufficient valid price data in historical points ({len(valid_points)} valid)")
                return False
            
            logger.info(f"✅ Filtered to {len(valid_points):,} valid data points")
            
            # Extract arrays following aqua-blue examples pattern
            timestamps = [point.timestamp for point in valid_points]
            prices = [point.price for point in valid_points]
            volumes = [point.volume for point in valid_points]
            
            # Create uniform time grid for aqua-blue compatibility
            start_time = timestamps[0]
            end_time = timestamps[-1]
            total_duration = end_time - start_time
            
            # Calculate number of uniform points needed
            target_interval = timedelta(seconds=target_interval_seconds)
            num_uniform_points = max(100, int(total_duration.total_seconds() / target_interval_seconds) + 1)
            
            # Create uniform timestamps (following examples pattern)
            uniform_timestamps = []
            for i in range(num_uniform_points):
                time_point = start_time + (i * target_interval)
                if time_point <= end_time:
                    uniform_timestamps.append(time_point)
            
            # Ensure we have at least 100 points for effective training
            if len(uniform_timestamps) < 100:
                # Adjust interval to get more points
                adjusted_interval = total_duration.total_seconds() / 100
                target_interval = timedelta(seconds=adjusted_interval)
                uniform_timestamps = []
                for i in range(100):
                    time_point = start_time + (i * target_interval)
                    uniform_timestamps.append(time_point)
            
            logger.info(f"🔧 Created {len(uniform_timestamps):,} uniform timestamps "
                       f"(interval: {target_interval.total_seconds():.1f}s)")
            
            # Convert for numpy interpolation (following examples)
            original_times = np.array([t.timestamp() for t in timestamps])
            uniform_times = np.array([t.timestamp() for t in uniform_timestamps])
            
            # Interpolate to uniform grid (like examples do for missing data)
            uniform_prices = np.interp(uniform_times, original_times, prices)
            uniform_volumes = np.interp(uniform_times, original_times, volumes)
            
            # Create dependent variable matrix (price, volume) - following examples format
            # Similar to goldstocks.csv which has multiple dependent variables
            dependent_variables = np.column_stack([uniform_prices, uniform_volumes])
            
            # Create TimeSeries object following examples pattern
            # Like csv-example.py and sine-cosine.py
            self.current_series = aqua_blue.time_series.TimeSeries(
                dependent_variable=dependent_variables,
                times=uniform_timestamps
            )
            
            # Validate the TimeSeries was created properly
            if self.current_series is None:
                logger.error("Failed to create TimeSeries object")
                return False
            
            series_length = len(self.current_series.dependent_variable)
            if series_length < 50:
                logger.warning(f"Created TimeSeries is too short ({series_length} points)")
                return False
            
            # Update buffer with recent data for real-time processing
            buffer_size = min(self.window_size, len(valid_points))
            recent_points = valid_points[-buffer_size:]
            
            self.data_buffer.clear()
            for point in recent_points:
                self.data_buffer.append(point)
            
            # Show statistics
            time_span_days = (end_time - start_time).days
            points_per_day = len(valid_points) / max(1, time_span_days)
            
            logger.info(f"✅ Successfully processed {len(historical_points):,} → {len(valid_points):,} → {series_length:,} points")
            logger.info(f"📅 Time span: {time_span_days:,} days ({time_span_days/365:.1f} years)")
            logger.info(f"📊 Data density: {points_per_day:.1f} points/day")
            logger.info(f"⚡ Uniform interval: {target_interval.total_seconds():.1f} seconds")
            logger.info(f"🧠 TimeSeries ready for aqua-blue model training!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing historical data in bulk: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            return False 