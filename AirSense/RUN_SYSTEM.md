# How to Run the Complete AirSense System

## Prerequisites

1. **Python 3.8+** installed and accessible
2. **Java 8+** (required for Apache Spark)
3. **Git** (for cloning if needed)

## Quick Start - Run Complete System

### Option 1: Run All Components (Recommended)

```bash
# Navigate to project directory
cd AirSense

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the complete system (pipeline + API + dashboard)
python src/main.py all
```

### Option 2: Run Components Individually

```bash
# 1. Run data processing pipeline first
python src/main.py pipeline

# 2. In a new terminal, start the API server
python src/main.py api

# 3. In another terminal, start the dashboard
python src/main.py dashboard
```

## Docker Deployment (Alternative)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Access Points

Once running, you can access:

- **Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## Testing the System

### 1. Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Get data
curl http://localhost:8000/data?limit=10

# Simple forecast
curl -X POST http://localhost:8000/forecast/simple \
  -H "Content-Type: application/json" \
  -d '{"pollutant": "PM2.5", "method": "moving_average", "forecast_steps": 24}'

# Time pattern analysis
curl "http://localhost:8000/analysis/time-patterns?pollutant=PM2.5&analysis_type=daily"

# Correlation analysis
curl http://localhost:8000/analysis/correlations?analysis_type=comprehensive
```

### 2. Test Dashboard Features

Open http://localhost:8501 in your browser and explore:
- Data upload and exploration
- Real-time air quality monitoring
- Forecast visualization
- Health alerts
- Statistical analysis

## Troubleshooting

### Common Issues

1. **Python not found**
   ```bash
   # On Windows, try:
   py src/main.py all
   
   # Or find Python installation:
   where python
   ```

2. **Spark issues**
   - Ensure Java 8+ is installed
   - Set JAVA_HOME environment variable
   ```bash
   # On Windows:
   set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_291
   ```

3. **Port conflicts**
   ```bash
   # Use different ports
   python src/main.py api --port 8001
   python src/main.py dashboard --port 8502
   ```

4. **Missing dependencies**
   ```bash
   # Install specific packages
   pip install pyspark pandas fastapi streamlit plotly
   ```

### Environment Setup

If you encounter issues, verify your environment:

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(pyspark|fastapi|streamlit)"

# Check Java
java -version
```

## Data Requirements

The system expects air quality data in CSV format with columns:
- `datetime` (required)
- `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3` (at least one)
- `temperature`, `humidity`, `pressure`, `wind_speed`, `wind_direction` (optional)

Sample data is available at `data/beijing_demo.csv`.

## System Architecture

When running, the system consists of:

1. **Data Processing Pipeline**: Spark-based data cleaning and feature engineering
2. **API Server**: FastAPI backend with RESTful endpoints
3. **Dashboard**: Streamlit frontend for visualization
4. **Storage**: Processed data saved in Parquet format

## Monitoring

- API logs: Console output
- Dashboard logs: Console output  
- Processed data: `data/processed/` directory
- Models: `models/` directory

## Stopping the System

```bash
# If using Ctrl+C in individual terminals, stop each component
# If using Docker:
docker-compose down
```

## Next Steps

Once running:
1. Upload your own data or use the demo dataset
2. Explore the dashboard features
3. Test different forecasting methods
4. Analyze correlations and patterns
5. Generate reports and insights
