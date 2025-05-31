#!/usr/bin/env python3
"""
Enhanced Stock Data Pipeline Example

This example demonstrates the data preparation pipeline:
1. Loads historical data for model training
2. Connects to real-time websocket data streams
3. Processes and caches data efficiently
4. Prepares timeseries data for model ingestion

Usage:
    python data_pipeline_example.py --symbol AAPL --max-historical-days 1825
"""

import asyncio
import logging
import argparse
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the parent directories to the path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables from .env files
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))  # Project root
load_dotenv('.env')  # Current directory

from polygonDataPipeline.stock_data_pipeline import StockDataPipeline


def setup_logging():
    """Set up comprehensive logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'data_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def on_data_update(data_stats):
    """Callback for data updates with progress tracking."""
    buffer_size = data_stats.get("buffer_size", 0)
    latest_price = data_stats.get('latest_price', 'N/A')
    data_rate = data_stats.get('data_rate', 0)
    
    # Show frequent updates early, then less frequent
    show_update = (
        buffer_size <= 20 or  # First 20 updates
        (buffer_size <= 100 and buffer_size % 10 == 0) or  # Every 10th up to 100
        buffer_size % 50 == 0  # Every 50th after that
    )
    
    if show_update:
        print(f"📊 Data Update #{buffer_size}: Latest ${latest_price} | Rate: {data_rate:.1f}/sec")
        
        # Show progress milestones
        if buffer_size == 1:
            print("   🎉 First data point received!")
        elif buffer_size == 10:
            print("   ⚡ Building data foundation...")
        elif buffer_size == 50:
            print("   📈 Data stream stabilizing...")
        elif buffer_size == 100:
            print("   🎯 Reached training threshold...")
        elif buffer_size == 500:
            print("   🚀 Data pipeline fully operational!")


def on_model_data_ready(training_data_info):
    """Callback when model training data is ready."""
    data_count = training_data_info.get('data_count', 0)
    time_span = training_data_info.get('time_span', 'N/A')
    data_quality = training_data_info.get('data_quality', 'N/A')
    
    print(f"\n🧠 MODEL TRAINING DATA READY!")
    print(f"   📊 Data Points: {data_count:,}")
    print(f"   ⏱️  Time Span: {time_span}")
    print(f"   ✅ Data Quality: {data_quality}")
    print(f"   🔄 Ready for timeseries model ingestion")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Stock Data Pipeline')
    
    # API and connection settings
    parser.add_argument('--api-key', type=str, help='Polygon.io API key (or set POLYGON_API_KEY env var)')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to process (default: AAPL)')
    
    # Historical data settings  
    parser.add_argument('--historical-days', type=int, default=1825, 
                       help='Days of historical data to load (default: 1825 = 5 years)')
    parser.add_argument('--max-historical', action='store_true',
                       help='Load MAXIMUM available historical data')
    parser.add_argument('--max-years-back', type=int, default=10,
                       help='Maximum years to search back when using --max-historical (default: 10)')
    parser.add_argument('--min-years', type=int, default=5,
                       help='Minimum years of historical data required (default: 5)')
    
    # S3 settings for Flat Files
    parser.add_argument('--s3-access-key', type=str, help='S3 access key for Flat Files (or set POLYGON_S3_ACCESS_KEY)')
    parser.add_argument('--s3-secret-key', type=str, help='S3 secret key for Flat Files (or set POLYGON_S3_SECRET_KEY)')
    
    # Data pipeline settings
    parser.add_argument('--training-window', type=int, default=500, 
                       help='Training window size for the model (default: 500)')
    parser.add_argument('--timeout', type=int, default=300, help='Runtime in seconds (default: 300)')
    parser.add_argument('--data-only', action='store_true', help='Only load data, do not start streaming')
    
    return parser.parse_args()


async def main():
    """Main function for the data pipeline."""
    args = parse_args()
    
    # Get S3 credentials (optional for Flat Files)
    s3_access_key = args.s3_access_key or os.getenv('POLYGON_S3_ACCESS_KEY')
    s3_secret_key = args.s3_secret_key or os.getenv('POLYGON_S3_SECRET_KEY')
    
    # Enforce minimum historical data requirement
    min_days = args.min_years * 365
    if not args.max_historical and args.historical_days < min_days:
        print(f"⚠️  Requested {args.historical_days} days, but minimum is {min_days} days ({args.min_years} years)")
        args.historical_days = min_days
        print(f"📈 Automatically adjusted to {min_days} days for optimal model training")
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🔄 " + "=" * 60)
    print("🔄 ENHANCED STOCK DATA PIPELINE")
    print("🔄 " + "=" * 60)
    print(f"📊 Symbol: {args.symbol}")
    
    if args.max_historical:
        print(f"📅 Historical Data: MAXIMUM available data ({args.max_years_back} years max)")
    else:
        years = args.historical_days / 365
        print(f"📅 Historical Data: {args.historical_days} days ({years:.1f} years)")
        if args.historical_days >= min_days:
            print(f"✅ Meets minimum requirement of {args.min_years} years")
        else:
            print(f"⚠️  Below minimum requirement of {args.min_years} years")
    
    print(f"🧠 Training Window: {args.training_window} points")
    
    # Show data source info
    if args.historical_days > 90 or args.max_historical:
        if s3_access_key and s3_secret_key:
            print("🚀 Data Source: Flat Files (S3) - Efficient for large datasets!")
        else:
            print("⚠️  Data Source: REST API - Consider S3 credentials for better performance")
    else:
        print("📡 Data Source: REST API - Good for smaller datasets")
    
    if not args.data_only:
        print(f"⏱️  Runtime: {args.timeout} seconds")
    else:
        print("📦 Mode: Data loading only")
    print("🔄 " + "=" * 60)
    print()
    
    start_time = datetime.now()
    pipeline = None  # Initialize pipeline to None
    
    try:
        # Initialize data pipeline
        print("🚀 Initializing data pipeline...")
        pipeline = StockDataPipeline(
            symbol=args.symbol,
            api_key=args.api_key or os.getenv('POLYGON_API_KEY'),
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            training_window=args.training_window
        )
        
        # Set up data callbacks (no prediction callbacks)
        pipeline.set_data_update_callback(on_data_update)
        if hasattr(pipeline, 'set_model_data_callback'):
            pipeline.set_model_data_callback(on_model_data_ready)
        
        # Load historical data to prepare the model training dataset
        historical_loaded = False
        
        if args.max_historical:
            print(f"📈 Loading MAXIMUM historical data for {args.symbol}...")
            historical_loaded = await pipeline.load_maximum_historical_data(
                timespan="minute",
                max_years_back=args.max_years_back
            )
        else:
            print(f"📈 Loading {args.historical_days} days of historical data for {args.symbol}...")
            historical_loaded = await pipeline.load_historical_data(
                days_back=args.historical_days,
                timespan="minute"
            )
        
        if historical_loaded:
            status = pipeline.get_model_status()
            print(f"✅ Loaded {status['data_count']:,} historical data points")
            
            # Show data statistics
            hist_info = pipeline.get_historical_data_info()
            print(f"📊 Data Range: {hist_info['date_range']['start_date']} to {hist_info['date_range']['end_date']}")
            print(f"📈 Price Range: ${hist_info['price_range']['min_price']:.2f} - ${hist_info['price_range']['max_price']:.2f}")
            print(f"📦 Data Quality: {hist_info['data_metrics']['data_density']}")
            print(f"💾 Cache Status: {hist_info['cache_info']['cache_status']}")
            
            if status['model_trained']:
                print("🧠 Model training data prepared!")
                print("✅ Data pipeline ready for timeseries model ingestion")
            else:
                print("⚠️  Model not yet trained - data processing in progress")
        else:
            print("❌ Could not load historical data")
            return
        
        # Start real-time data streaming (unless data-only mode)
        if not args.data_only:
            print(f"\n📡 Starting real-time data stream for {args.symbol}...")
            await pipeline.start_data_stream()
            
            # Wait for data streaming with timeout
            start_time = datetime.now()
            timeout_seconds = args.timeout
            
            print(f"🔄 Streaming data for {timeout_seconds} seconds...")
            print("📊 Monitoring data quality and processing...")
            
            while True:
                await asyncio.sleep(1)
                elapsed = datetime.now() - start_time
                
                if elapsed.total_seconds() >= timeout_seconds:
                    print(f"\n⏰ Data streaming completed after {elapsed}")
                    break
            
            # Stop the data stream
            if pipeline:
                await pipeline.stop_data_stream()
        else:
            print("📦 Data loading completed - skipping real-time streaming")
        
    except KeyboardInterrupt:
        runtime = datetime.now() - start_time
        print(f"\n🛑 Stopped by user after {runtime}")
        if pipeline:
            await pipeline.stop_data_stream()
    except Exception as e:
        runtime = datetime.now() - start_time
        logger.error(f"Error in data pipeline: {e}")
        print(f"❌ Error after {runtime}: {e}")
        if pipeline:
            await pipeline.stop_data_stream()
        import traceback
        traceback.print_exc()
    
    finally:
        # Show final data statistics
        if pipeline:
            status = pipeline.get_model_status()
            runtime = datetime.now() - start_time
            
            print(f"\n📈 DATA PIPELINE SUMMARY")
            print(f"{'='*50}")
            print(f"Total Data Points: {status['data_count']:,}")
            print(f"Model Training Ready: {'✅ Yes' if status['model_trained'] else '⚠️  Preparing'}")
            print(f"Runtime: {runtime}")
            
            # Show cache statistics
            try:
                cache_info = pipeline.data_cache.get_cache_info()
                print(f"Cache Files: {len(cache_info['files'])}")
                print(f"Cache Size: {cache_info['total_size']}")
            except:
                print("Cache Info: Not available")
        
        print(f"\n🔄 Data pipeline session completed!")
        print(f"📦 Historical data loaded and cached for future use")
        print(f"🧠 Data ready for timeseries model training and inference")


if __name__ == "__main__":
    asyncio.run(main()) 