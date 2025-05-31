"""
Data caching system for historical stock data.

This module provides efficient caching of historical data to CSV files,
reducing API calls and improving performance for subsequent runs.
"""

import os
import csv
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass

from .polygon_client import HistoricalDataPoint

logger = logging.getLogger(__name__)


@dataclass
class CacheInfo:
    """Information about cached data."""
    file_path: str
    symbol: str
    start_date: datetime
    end_date: datetime
    total_points: int
    file_size_mb: float
    last_modified: datetime


class HistoricalDataCache:
    """
    Manages caching of historical stock data to CSV files.
    
    Features:
    - Saves historical data to organized CSV files
    - Checks for existing cached data before API calls
    - Only downloads missing/new data
    - Updates existing cache files
    - Provides cache management utilities
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the data cache.
        
        Args:
            cache_dir: Directory to store cached data files. If None, uses 
                      'data_cache' folder within the polygonDataPipeline module directory.
        """
        if cache_dir is None:
            # Get the directory where this module is located
            module_dir = Path(__file__).parent
            cache_dir = module_dir / "data_cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized data cache at {self.cache_dir.absolute()}")
    
    def _get_cache_file_path(self, symbol: str, timespan: str = "minute") -> Path:
        """Get the cache file path for a symbol and timespan."""
        filename = f"{symbol.upper()}_{timespan}_historical.csv"
        return self.cache_dir / filename
    
    def _get_cache_info_path(self, symbol: str, timespan: str = "minute") -> Path:
        """Get the cache info file path for metadata."""
        filename = f"{symbol.upper()}_{timespan}_info.txt"
        return self.cache_dir / filename
    
    def has_cached_data(self, symbol: str, timespan: str = "minute") -> bool:
        """Check if cached data exists for a symbol."""
        cache_file = self._get_cache_file_path(symbol, timespan)
        return cache_file.exists()
    
    def get_cache_info(self, symbol: str, timespan: str = "minute") -> Optional[CacheInfo]:
        """Get information about cached data."""
        cache_file = self._get_cache_file_path(symbol, timespan)
        info_file = self._get_cache_info_path(symbol, timespan)
        
        if not cache_file.exists():
            return None
        
        try:
            # Get file stats
            stat = cache_file.stat()
            file_size_mb = stat.st_size / (1024 * 1024)
            last_modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Try to read cache info if available
            start_date = None
            end_date = None
            total_points = 0
            
            if info_file.exists():
                with open(info_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('start_date:'):
                            start_date = datetime.fromisoformat(line.split(':', 1)[1].strip())
                        elif line.startswith('end_date:'):
                            end_date = datetime.fromisoformat(line.split(':', 1)[1].strip())
                        elif line.startswith('total_points:'):
                            total_points = int(line.split(':', 1)[1].strip())
            
            # If no info file, try to get dates from CSV data
            if start_date is None or end_date is None:
                try:
                    df = pd.read_csv(cache_file, nrows=1)
                    if not df.empty:
                        start_date = pd.to_datetime(df['timestamp'].iloc[0])
                    
                    df = pd.read_csv(cache_file, skiprows=lambda x: x != 0 and x < sum(1 for _ in open(cache_file)) - 1)
                    if not df.empty:
                        end_date = pd.to_datetime(df['timestamp'].iloc[0])
                    
                    # Get total rows
                    with open(cache_file, 'r') as f:
                        total_points = sum(1 for _ in f) - 1  # Subtract header
                except Exception as e:
                    logger.warning(f"Could not read cache metadata: {e}")
            
            return CacheInfo(
                file_path=str(cache_file),
                symbol=symbol.upper(),
                start_date=start_date or last_modified,
                end_date=end_date or last_modified,
                total_points=total_points,
                file_size_mb=file_size_mb,
                last_modified=last_modified
            )
            
        except Exception as e:
            logger.error(f"Error getting cache info for {symbol}: {e}")
            return None
    
    def save_historical_data(self, symbol: str, data: List[HistoricalDataPoint], 
                           timespan: str = "minute", append: bool = False) -> bool:
        """
        Save historical data to CSV cache.
        
        Args:
            symbol: Stock symbol
            data: List of historical data points
            timespan: Data timespan (minute, hour, day)
            append: Whether to append to existing file or overwrite
            
        Returns:
            True if saved successfully
        """
        if not data:
            logger.warning(f"No data to save for {symbol}")
            return False
        
        cache_file = self._get_cache_file_path(symbol, timespan)
        info_file = self._get_cache_info_path(symbol, timespan)
        
        try:
            # Sort data by timestamp
            sorted_data = sorted(data, key=lambda x: x.timestamp)
            
            # Prepare CSV data
            csv_data = []
            for point in sorted_data:
                csv_data.append({
                    'timestamp': point.timestamp.isoformat(),
                    'symbol': symbol.upper(),
                    'open': point.open,
                    'high': point.high,
                    'low': point.low,
                    'close': point.close,
                    'volume': point.volume,
                    'timespan': timespan
                })
            
            # Write to CSV
            mode = 'a' if append and cache_file.exists() else 'w'
            write_header = mode == 'w' or not cache_file.exists()
            
            with open(cache_file, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'timespan'
                ])
                
                if write_header:
                    writer.writeheader()
                
                writer.writerows(csv_data)
            
            # Save metadata
            start_date = sorted_data[0].timestamp
            end_date = sorted_data[-1].timestamp
            
            with open(info_file, 'w') as f:
                f.write(f"symbol: {symbol.upper()}\n")
                f.write(f"timespan: {timespan}\n")
                f.write(f"start_date: {start_date.isoformat()}\n")
                f.write(f"end_date: {end_date.isoformat()}\n")
                f.write(f"total_points: {len(sorted_data)}\n")
                f.write(f"saved_at: {datetime.now().isoformat()}\n")
            
            file_size = cache_file.stat().st_size / (1024 * 1024)
            logger.info(f"💾 Saved {len(sorted_data):,} data points for {symbol} to {cache_file.name} "
                       f"({file_size:.1f} MB)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving cache for {symbol}: {e}")
            return False
    
    def load_historical_data(self, symbol: str, timespan: str = "minute",
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[HistoricalDataPoint]:
        """
        Load historical data from CSV cache.
        
        Args:
            symbol: Stock symbol
            timespan: Data timespan
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of historical data points
        """
        cache_file = self._get_cache_file_path(symbol, timespan)
        
        if not cache_file.exists():
            logger.info(f"No cached data found for {symbol}")
            return []
        
        try:
            # Read CSV data
            df = pd.read_csv(cache_file)
            
            if df.empty:
                logger.warning(f"Empty cache file for {symbol}")
                return []
            
            # Convert timestamp column
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Apply date filters if provided
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
            
            # Convert to HistoricalDataPoint objects
            data_points = []
            for _, row in df.iterrows():
                point = HistoricalDataPoint(
                    timestamp=row['timestamp'].to_pydatetime(),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume']) if pd.notna(row['volume']) else 0
                )
                data_points.append(point)
            
            logger.info(f"📂 Loaded {len(data_points):,} cached data points for {symbol}")
            return data_points
            
        except Exception as e:
            logger.error(f"Error loading cache for {symbol}: {e}")
            return []
    
    def get_missing_date_ranges(self, symbol: str, requested_start: datetime, 
                              requested_end: datetime, timespan: str = "minute") -> List[Tuple[datetime, datetime]]:
        """
        Determine what date ranges are missing from cache.
        
        Args:
            symbol: Stock symbol
            requested_start: Start of requested range
            requested_end: End of requested range
            timespan: Data timespan
            
        Returns:
            List of (start, end) tuples for missing ranges
        """
        cache_info = self.get_cache_info(symbol, timespan)
        
        if not cache_info:
            # No cache, need entire range
            return [(requested_start, requested_end)]
        
        missing_ranges = []
        
        # Check if we need data before cached range
        if requested_start < cache_info.start_date:
            missing_ranges.append((requested_start, cache_info.start_date - timedelta(seconds=1)))
        
        # Check if we need data after cached range
        if requested_end > cache_info.end_date:
            missing_ranges.append((cache_info.end_date + timedelta(seconds=1), requested_end))
        
        return missing_ranges
    
    def update_cache(self, symbol: str, new_data: List[HistoricalDataPoint], 
                    timespan: str = "minute") -> bool:
        """
        Update existing cache with new data, avoiding duplicates.
        
        Args:
            symbol: Stock symbol
            new_data: New historical data points
            timespan: Data timespan
            
        Returns:
            True if updated successfully
        """
        if not new_data:
            return True
        
        try:
            # Load existing data
            existing_data = self.load_historical_data(symbol, timespan)
            
            # Create set of existing timestamps for duplicate detection
            existing_timestamps = {point.timestamp for point in existing_data}
            
            # Filter out duplicates from new data
            unique_new_data = [
                point for point in new_data 
                if point.timestamp not in existing_timestamps
            ]
            
            if not unique_new_data:
                logger.info(f"No new unique data to add for {symbol}")
                return True
            
            # Combine and sort all data
            all_data = existing_data + unique_new_data
            all_data.sort(key=lambda x: x.timestamp)
            
            # Save combined data
            success = self.save_historical_data(symbol, all_data, timespan, append=False)
            
            if success:
                logger.info(f"✅ Updated cache for {symbol}: added {len(unique_new_data):,} new points "
                           f"(total: {len(all_data):,})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating cache for {symbol}: {e}")
            return False
    
    def clear_cache(self, symbol: Optional[str] = None, timespan: Optional[str] = None) -> bool:
        """
        Clear cached data.
        
        Args:
            symbol: Specific symbol to clear (None for all)
            timespan: Specific timespan to clear (None for all)
            
        Returns:
            True if cleared successfully
        """
        try:
            if symbol and timespan:
                # Clear specific symbol/timespan
                cache_file = self._get_cache_file_path(symbol, timespan)
                info_file = self._get_cache_info_path(symbol, timespan)
                
                removed_files = 0
                if cache_file.exists():
                    cache_file.unlink()
                    removed_files += 1
                if info_file.exists():
                    info_file.unlink()
                    removed_files += 1
                
                logger.info(f"Cleared cache for {symbol} ({timespan}): {removed_files} files removed")
                
            else:
                # Clear all cache files
                removed_files = 0
                for file_path in self.cache_dir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        removed_files += 1
                
                logger.info(f"Cleared all cache: {removed_files} files removed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def list_cached_symbols(self) -> Dict[str, List[str]]:
        """
        List all cached symbols and their timespans.
        
        Returns:
            Dictionary mapping symbols to list of timespans
        """
        cached_symbols = {}
        
        try:
            for file_path in self.cache_dir.glob("*_historical.csv"):
                filename = file_path.stem
                # Parse: SYMBOL_TIMESPAN_historical
                parts = filename.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    timespan = parts[1]
                    
                    if symbol not in cached_symbols:
                        cached_symbols[symbol] = []
                    if timespan not in cached_symbols[symbol]:
                        cached_symbols[symbol].append(timespan)
            
            return cached_symbols
            
        except Exception as e:
            logger.error(f"Error listing cached symbols: {e}")
            return {}
    
    def get_cache_stats(self) -> Dict:
        """Get overall cache statistics."""
        try:
            total_files = 0
            total_size_mb = 0
            symbols = self.list_cached_symbols()
            
            for file_path in self.cache_dir.glob("*.csv"):
                if file_path.is_file():
                    total_files += 1
                    total_size_mb += file_path.stat().st_size / (1024 * 1024)
            
            return {
                "cache_directory": str(self.cache_dir.absolute()),
                "total_symbols": len(symbols),
                "total_files": total_files,
                "total_size_mb": round(total_size_mb, 2),
                "symbols": symbols
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {} 