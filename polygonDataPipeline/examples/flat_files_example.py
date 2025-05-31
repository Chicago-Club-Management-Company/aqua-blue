#!/usr/bin/env python3
"""
Polygon.io Flat Files Example

This example demonstrates how to use Polygon.io Flat Files for efficient access
to large amounts of historical market data via S3-compatible endpoints.

This is particularly useful for:
- Large historical datasets (months/years of minute data)
- Batch processing scenarios
- Research and backtesting with extensive historical data

Requirements:
- Polygon.io subscription with Flat Files access
- S3 access credentials from your Polygon.io dashboard

Usage:
    python flat_files_example.py --symbol AAPL --days 180 --s3-access-key YOUR_ACCESS_KEY --s3-secret-key YOUR_SECRET_KEY
    
    # Or with environment variables:
    export POLYGON_S3_ACCESS_KEY=your_access_key
    export POLYGON_S3_SECRET_KEY=your_secret_key
    python flat_files_example.py --symbol AAPL --days 365
"""

import asyncio
import logging
import argparse
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv('.env')

from polygonDataPipeline import PolygonFlatFilesClient, HistoricalDataPoint


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def demonstrate_flat_files():
    """Demonstrate Flat Files functionality."""
    parser = argparse.ArgumentParser(description='Polygon.io Flat Files demonstration')
    parser.add_argument('--s3-access-key', help='S3 access key (or set POLYGON_S3_ACCESS_KEY env var)')
    parser.add_argument('--s3-secret-key', help='S3 secret key (or set POLYGON_S3_SECRET_KEY env var)')
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data (default: 30)')
    parser.add_argument('--list-files', action='store_true', help='List available files')
    parser.add_argument('--data-type', default='minute_aggs_v1', 
                       help='Data type (minute_aggs_v1, day_aggs_v1, etc.)')
    
    args = parser.parse_args()
    
    # Get S3 credentials
    s3_access_key = args.s3_access_key or os.getenv('POLYGON_S3_ACCESS_KEY')
    s3_secret_key = args.s3_secret_key or os.getenv('POLYGON_S3_SECRET_KEY')
    
    if not s3_access_key or not s3_secret_key:
        print("❌ Error: S3 credentials required for Flat Files!")
        print("Get them from your Polygon.io dashboard and provide via:")
        print("  1. Command line: --s3-access-key KEY --s3-secret-key SECRET")
        print("  2. Environment: export POLYGON_S3_ACCESS_KEY=key POLYGON_S3_SECRET_KEY=secret")
        print("  3. .env file: Add POLYGON_S3_ACCESS_KEY and POLYGON_S3_SECRET_KEY")
        print("\n🔗 Access keys available at: https://polygon.io/dashboard")
        return
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("📁 " + "=" * 60)
    print("📁 POLYGON.IO FLAT FILES DEMONSTRATION")
    print("📁 " + "=" * 60)
    print(f"📊 Symbol: {args.symbol}")
    print(f"📅 Days: {args.days}")
    print(f"📋 Data Type: {args.data_type}")
    print("📁 " + "=" * 60)
    
    try:
        # Initialize Flat Files client
        print("\n🔌 Initializing Flat Files client...")
        client = PolygonFlatFilesClient(
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key
        )
        
        if args.list_files:
            print(f"\n📋 Listing available files...")
            files = await client.list_available_files("us_stocks_sip")
            
            print(f"Found {len(files)} files. Recent examples:")
            for file_path in sorted(files)[-10:]:  # Show last 10 files
                print(f"  📄 {file_path}")
            
            return
        
        # Fetch historical data
        print(f"\n📊 Fetching {args.days} days of data for {args.symbol}...")
        start_time = datetime.now()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        historical_data = await client.fetch_historical_data(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            data_type=args.data_type
        )
        
        fetch_time = datetime.now() - start_time
        
        if historical_data:
            print(f"\n✅ SUCCESS!")
            print(f"📊 Fetched: {len(historical_data):,} data points")
            print(f"⏱️  Time: {fetch_time.total_seconds():.1f} seconds")
            print(f"📈 Date range: {historical_data[0].timestamp.date()} to {historical_data[-1].timestamp.date()}")
            
            # Show sample data
            print(f"\n📋 Sample data (first 5 points):")
            for i, point in enumerate(historical_data[:5]):
                print(f"  {i+1}. {point.timestamp} | O:{point.open} H:{point.high} L:{point.low} C:{point.close} V:{point.volume}")
            
            # Calculate statistics
            prices = [point.close for point in historical_data]
            volumes = [point.volume for point in historical_data]
            
            print(f"\n📈 Statistics:")
            print(f"  💰 Price range: ${min(prices):.2f} - ${max(prices):.2f}")
            print(f"  📊 Avg volume: {sum(volumes) / len(volumes):,.0f}")
            print(f"  📅 Data points per day: {len(historical_data) / args.days:.0f}")
            
            # Performance comparison
            points_per_second = len(historical_data) / fetch_time.total_seconds()
            print(f"\n🚀 Performance:")
            print(f"  ⚡ {points_per_second:,.0f} data points per second")
            print(f"  💡 This would take ~{len(historical_data) / 50000:.1f} REST API calls")
            print(f"  🎯 Flat Files are much more efficient for large datasets!")
            
        else:
            print(f"❌ No data found for {args.symbol}")
            print(f"💡 Try different symbol or date range")
            
    except Exception as e:
        logger.error(f"Error in Flat Files demo: {e}")
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Check your S3 credentials")
        print("  2. Ensure you have Flat Files access in your subscription")
        print("  3. Try a different symbol or date range")


async def main():
    """Main function."""
    await demonstrate_flat_files()


if __name__ == "__main__":
    asyncio.run(main()) 