# Stock Data Pipeline Module

Real-time stock data streaming and historical data loading using polygon.io with aqua-blue integration.

## 🎯 Focus: Data Pipeline for Model Training

This module provides a robust data pipeline for loading, processing, and streaming stock market data:

| Component | Description | Benefit |
|-----------|-------------|---------|
| **📊 Historical Data Loading** | Intelligent fetching with 5+ year minimum | Comprehensive training datasets |
| **🚀 S3 Flat Files Support** | Efficient large dataset handling | Fast bulk data access |
| **📡 Real-time Streaming** | Live websocket data from polygon.io | Current market conditions |
| **💾 Data Caching** | Smart caching system | Avoid re-downloading data |
| **🎯 Data Quality** | Processing and validation | Clean, reliable datasets |

## Features

- 📊 **Historical data integration**: Load years of historical data efficiently
- 💾 **Multiple data sources**: REST API + Flat Files (S3) support  
- 🚀 **Maximum historical data**: Get ALL available data for any ticker
- 📁 **Efficient data access**: Automatic selection between REST API and Flat Files
- 💾 **Intelligent caching**: Avoid re-downloading existing data
- 📡 **Real-time streaming**: Live websocket data for current conditions
- 🎯 **Data validation**: Quality checks and processing
- 📈 **Bulk processing**: Efficient handling of large datasets

## Data Sources

### REST API (Default)
- Good for: moderate datasets (days to months)
- Requires: polygon.io API key only
- Rate limited but sufficient for most use cases

### Flat Files (S3) - For Large Datasets
- Good for: large datasets (months to years of minute data)
- Requires: polygon.io API key + S3 access credentials
- Much more efficient for bulk historical data
- Automatically used for >90 days of minute data when credentials available

### Maximum Historical Data
- 🚀 Get ALL available historical data for any ticker
- Uses intelligent discovery to find maximum date range
- Automatically chooses most efficient method (Flat Files preferred)
- Can fetch years worth of minute-level data efficiently
- Perfect for building comprehensive training datasets

## Quick Start

### Basic Data Loading

```python
import asyncio
from price_prediction import StockDataPipeline

async def main():
    # Create data pipeline
    pipeline = StockDataPipeline(
        symbol="AAPL",
        api_key="your_polygon_api_key"
    )
    
    # Load 5 years of historical data (minimum enforced)
    await pipeline.load_historical_data(days_back=1825)
    
    # Start real-time data streaming
    await pipeline.start_data_stream()

asyncio.run(main())
```

### Maximum Historical Data Loading

```bash
# Load ALL available data with S3 for efficiency
python examples/data_pipeline_example.py --symbol AAPL --max-historical \
  --s3-access-key YOUR_ACCESS_KEY --s3-secret-key YOUR_SECRET_KEY

# Quick maximum data test
python examples/data_pipeline_example.py --symbol TSLA --max-historical --data-only
```

### Data-Only Mode (No Streaming)

```python
# Load and cache data without starting real-time streaming
python examples/data_pipeline_example.py --symbol AAPL --data-only --max-historical
```

## Installation

```bash
# Install with data pipeline dependencies
pip install -e ".[data_pipeline]"

# Or install specific dependencies
pip install websockets aiohttp python-dateutil numpy
```

## API Keys Setup

### 1. Basic API Key (Required)
Get your API key from [polygon.io](https://polygon.io/)

### 2. S3 Credentials (Optional - for Flat Files)
Get S3 access credentials from your [Polygon.io Dashboard](https://polygon.io/dashboard) for Flat Files access.

### Environment Variables

Create a `.env` file in your project root:

```bash
# Required
POLYGON_API_KEY=your_api_key_here

# Optional (for Flat Files)
POLYGON_S3_ACCESS_KEY=your_s3_access_key
POLYGON_S3_SECRET_KEY=your_s3_secret_key
```

## Examples

### Basic Data Pipeline

```bash
# Load 5 years of data and start streaming
python examples/data_pipeline_example.py --symbol AAPL --historical-days 1825

# Data-only mode (no streaming)
python examples/data_pipeline_example.py --symbol AAPL --data-only --historical-days 1825
```

**Expected Output:**
```
📊 Loaded 805,623 historical data points for AAPL
📅 Date range: 2019-01-02 to 2025-01-15 (1,839 days)
📈 Average points per day: 438.2
🎯 Data ready for timeseries model training!
```

### Maximum Historical Data

```bash
# Load ALL available data with S3 for efficiency
python examples/data_pipeline_example.py --symbol AAPL --max-historical \
  --s3-access-key YOUR_ACCESS_KEY --s3-secret-key YOUR_SECRET_KEY

# Quick maximum data test
python examples/data_pipeline_example.py --symbol TSLA --max-historical --data-only
```

**Expected Output:**
```
🚀 Loading MAXIMUM available historical data for AAPL
📊 Total data points: 1,425,891
📅 Date range: 2015-06-29 to 2025-01-15 (3,487 days = 9.6 years)
📈 Average points per day: 409.0
🎯 Data ready for timeseries model training!
```

### Flat Files Examples

```bash
# Test Flat Files access
python examples/flat_files_example.py --symbol AAPL --days 180

# List available files
python examples/flat_files_example.py --list-files

# Different data types
python examples/flat_files_example.py --symbol AAPL --days 30 --data-type day_aggs_v1
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   polygon.io    │    │  Data Processor  │    │   Data Cache    │
│   WebSocket     │───▶│   & Buffer       │───▶│   & Storage     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐             ▼
│ Historical Data │    │     Quality      │    ┌─────────────────┐
│ (REST/S3)       │───▶│   Validation     │───▶│   Model Ready   │
│                 │    │                  │    │     Dataset     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Performance Comparison

### REST API vs Flat Files

| Dataset Size | REST API | Flat Files | Recommendation |
|--------------|----------|------------|----------------|
| 1-30 days    | ✅ Fast   | ⚠️ Overkill | Use REST API |
| 30-90 days   | ✅ Good   | ✅ Good     | Either works |
| 90+ days     | ⚠️ Slow   | ✅ Fast     | Use Flat Files |
| 1+ years     | ❌ Very slow | ✅ Efficient | Flat Files only |

### Data Coverage by Method

| Method | Typical Coverage | Data Points | Loading Time |
|--------|------------------|-------------|--------------|
| **Standard (REST)** | 1-2 years | 500K-1M | 2-5 minutes |
| **Maximum (REST)** | 3-5 years | 1M-2M | 5-15 minutes |
| **Maximum (S3)** | 5-10 years | 2M-5M | 1-3 minutes |

## Data Quality Features

### Minimum Requirements
- **5-year minimum**: Enforced for optimal data coverage
- **Data density validation**: Ensures sufficient data points per day
- **Date range verification**: Confirms recent data availability
- **Quality metrics**: Provides data quality assessment

### Caching Intelligence
- **Duplicate detection**: Avoids re-downloading existing data
- **Gap identification**: Fetches only missing date ranges
- **Cache validation**: Verifies cached data integrity
- **Automatic cleanup**: Manages disk space efficiently

### Processing Features
- **Bulk processing**: Efficient handling of large datasets
- **Data validation**: Quality checks and error handling
- **Format standardization**: Consistent data structures
- **Memory optimization**: Handles large datasets efficiently

## Data Access Patterns

### Historical Data Info
```python
data_info = pipeline.get_historical_data_info()
print(f"Status: {data_info['status']}")
print(f"Total points: {data_info['data_metrics']['total_points']:,}")
print(f"Data quality: {data_info['data_metrics']['data_density']}")
print(f"Ready for training: {data_info['model_readiness']['sufficient_for_training']}")
```

### Cache Management
```python
# Check cache status
cache_info = pipeline.get_cache_info()
print(f"Cache status: {cache_info['status']}")
print(f"File size: {cache_info['data_metrics']['file_size_mb']:.1f} MB")

# Clear cache if needed
pipeline.clear_cache()

# Get all cache statistics
all_stats = pipeline.get_all_cache_stats()
```

### Data Callbacks
```python
def on_data_update(stats):
    print(f"Received data point #{stats['buffer_size']}: ${stats['latest_price']:.2f}")

def on_model_data_ready(training_data_info):
    print(f"Training data ready: {training_data_info['data_count']:,} points")
    print(f"Time span: {training_data_info['time_span']}")
    print(f"Quality: {training_data_info['data_quality']}")

pipeline.set_data_update_callback(on_data_update)
pipeline.set_model_data_callback(on_model_data_ready)
```

## Error Handling

The module includes comprehensive error handling for:
- Network connectivity issues
- API rate limits and authentication  
- Data parsing and validation
- S3 access and file format issues
- Cache management errors
- Large dataset memory management

## Troubleshooting

### Common Issues

1. **"No API key provided"**
   - Set `POLYGON_API_KEY` environment variable
   - Or pass `--api-key` parameter

2. **"S3 credentials required for Flat Files"**
   - Get S3 credentials from polygon.io dashboard
   - Set `POLYGON_S3_ACCESS_KEY` and `POLYGON_S3_SECRET_KEY`

3. **"No data found for symbol"**
   - Check symbol spelling (e.g., 'AAPL' not 'Apple')
   - Try different date range
   - Verify market was open during requested period

4. **"Insufficient historical data"**
   - Module enforces 5-year minimum for optimal datasets
   - Use `--max-historical` for maximum available data
   - Check symbol has sufficient trading history

5. **Slow historical data loading**
   - Use Flat Files for large datasets (>90 days)
   - Reduce `historical_days` parameter for testing
   - Check network connection

## Data Pipeline Commands

```bash
# Load comprehensive dataset
python examples/data_pipeline_example.py --symbol AAPL --max-historical --data-only

# Test different symbols
python examples/data_pipeline_example.py --symbol TSLA --historical-days 1825 --data-only
python examples/data_pipeline_example.py --symbol MSFT --max-historical --data-only

# With S3 for large datasets
python examples/data_pipeline_example.py --symbol AAPL --max-historical \
  --s3-access-key YOUR_KEY --s3-secret-key YOUR_SECRET --data-only
```

## Integration with External Models

This data pipeline is designed to feed timeseries models:

```python
# Load and prepare data
pipeline = StockDataPipeline("AAPL", api_key)
await pipeline.load_maximum_historical_data()

# Get data status for model training
status = pipeline.get_model_status()
if status['model_trained']:  # Sufficient data for training
    print(f"✅ {status['data_count']:,} data points ready for model training")
    
    # Your model training code here
    # model.train(data=pipeline.data_processor.get_timeseries())
```

## License

MIT License - see project root for details.

## Links

- [Polygon.io API Documentation](https://polygon.io/docs)
- [Polygon.io Flat Files Guide](https://polygon.io/docs/flat-files/quickstart)
- [aqua-blue Reservoir Computing](https://github.com/Chicago-Club-Management-Company/aqua-blue)

---

*🎯 **Key Purpose**: Robust data pipeline for loading and streaming stock market data to feed external timeseries models.* 📊🚀 