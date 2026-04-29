# AirSense - Enterprise Air Quality Analysis & Forecasting

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Apache Spark](https://img.shields.io/badge/Spark-3.4+-orange.svg)](https://spark.apache.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready Big Data analytics system for air quality monitoring and forecasting using Apache Spark, advanced ML models, and real-time APIs.

## 🏗️ Architecture

```
AirSense/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   │   ├── config.py      # Configuration management
│   │   ├── logging.py     # Structured logging
│   │   └── exceptions.py # Custom exceptions
│   ├── data/              # Data processing
│   │   ├── processor.py   # Spark data processor
│   │   ├── validator.py  # Data validation
│   │   └── schemas.py    # Data schemas
│   ├── models/            # ML models
│   │   ├── time_series.py # ARIMA/LSTM models
│   │   └── registry.py    # Model registry
│   └── api/               # FastAPI application
│       ├── main.py        # Application setup
│       └── routes.py      # API endpoints
├── tests/                 # Test suite
├── frontend/              # Streamlit dashboard
├── data/                  # Data directories
├── models/                # Model storage
├── logs/                  # Log files
├── Makefile              # Development commands
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Java 11+ (for Spark)

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd AirSense
cp .env.example .env
```

2. **Install dependencies:**
```bash
make install-dev  # Development environment
# or
make install      # Production only
```

3. **Run the system:**
```bash
make run-all  # Complete system
# or
python src/main.py all
```

## 📊 Features

### Big Data Processing
- **Apache Spark** for distributed data processing
- **Automated ETL** pipeline with data validation
- **Advanced feature engineering** (time-based, interactions, lag features)
- **Scalable parquet storage** with compression
- **Data quality monitoring** and reporting

### Machine Learning
- **ARIMA** models with automatic parameter selection
- **LSTM** neural networks for complex patterns
- **Model registry** with versioning
- **Performance evaluation** (MAE, RMSE, R², MAPE)
- **Confidence intervals** for forecasts

### API & Dashboard
- **FastAPI** REST API with automatic documentation
- **Streamlit** dashboard for real-time monitoring
- **Health checks** and system monitoring
- **Background task processing**
- **CORS support** for web integration

### Enterprise Features
- **Structured logging** with JSON output
- **Configuration management** with environment variables
- **Comprehensive testing** (pytest, coverage)
- **Security** best practices

## 🔧 Configuration

### Environment Variables
```bash
# Application
APP_NAME=AirSense
DEBUG=false

# Spark
SPARK_MASTER=local[*]
SPARK_EXECUTOR_MEMORY=4g

# API
API_HOST=0.0.0.0
API_PORT=8000

# ML
DEFAULT_FORECAST_STEPS=24
MODEL_RETRAIN_INTERVAL=24
```

### Data Schema
```python
# Required columns
datetime, PM2.5, PM10, NO2, SO2, CO, O3

# Optional weather data
temperature, humidity, pressure, wind_speed, wind_direction
```

## 📈 Usage Examples

### API Usage

```python
import requests

# Get current AQI
response = requests.get("http://localhost:8000/api/v1/aqi")
aqi_data = response.json()

# Generate forecast
forecast_request = {
    "target_pollutant": "PM2.5",
    "model_type": "arima",
    "steps": 24,
    "retrain": False
}
response = requests.post("http://localhost:8000/api/v1/forecast", json=forecast_request)
forecast = response.json()
```

### Python SDK

```python
from src.data.processor import SparkDataProcessor
from src.models import TimeSeriesForecaster

# Process data
processor = SparkDataProcessor()
result = processor.process_pipeline("data/raw/input.csv", "data/processed/output.parquet")

# Train model
forecaster = TimeSeriesForecaster()
model_result = forecaster.train_model("arima", data)

# Generate forecast
forecast = forecaster.forecast(model_result["model_name"], 24)
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test categories
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m spark
```

## 🔒 Security

- **Input validation** with Pydantic models
- **CORS configuration** for web access
- **Secret management** with environment variables
- **SQL injection prevention** (Spark SQL)
- **Rate limiting** (configurable)

## 📈 Performance

### Benchmarks
- **Data Processing:** 1M records in <2 minutes
- **Model Training:** ARIMA in <30 seconds, LSTM in <2 minutes
- **API Latency:** <100ms average response time
- **Memory Usage:** <2GB for typical workloads

### Scalability
- **Distributed Spark** processing for large datasets
- **Efficient data formats** (Parquet with compression)
- **Optimized memory usage** and caching strategies

## 🛠️ Development

### Code Quality
```bash
# Linting and formatting
make lint
make format

# Pre-commit hooks
pre-commit run --all-files
```

### Adding Features
1. **Create feature branch**
2. **Add tests** (pytest)
3. **Update documentation**
4. **Run quality checks**
5. **Submit pull request**

### Project Structure
- **`src/core/`** - Core utilities and configuration
- **`src/data/`** - Data processing and validation
- **`src/models/`** - Machine learning models
- **`src/api/`** - REST API implementation
- **`tests/`** - Comprehensive test suite

## 📝 API Documentation

### Endpoints

#### Health & Status
- `GET /health` - System health check
- `GET /` - API information

#### Data Operations
- `GET /api/v1/data` - Retrieve air quality data
- `GET /api/v1/stats` - Data statistics
- `POST /api/v1/pipeline/run` - Run data pipeline

#### Forecasting
- `POST /api/v1/forecast` - Generate forecasts
- `GET /api/v1/models` - List available models
- `GET /api/v1/aqi` - Current AQI and health info

### Response Format
```json
{
  "status": "success",
  "data": {...},
  "timestamp": "2024-01-01T12:00:00Z",
  "metadata": {...}
}
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

### Development Guidelines
- Follow **PEP 8** style guidelines
- Add **comprehensive tests** for new features
- Update **documentation** and API specs
- Use **type hints** throughout codebase
- Ensure **CI/CD** pipeline passes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation:** This README and inline code docs
- **Issues:** GitHub Issues for bug reports
- **Discussions:** GitHub Discussions for questions
- **Email:** team@airsense.com

## 🗺️ Roadmap

### v1.1 (Planned)
- [ ] Real-time data streaming
- [ ] Advanced anomaly detection
- [ ] Mobile app API
- [ ] Multi-city support

### v1.2 (Future)
- [ ] ML model auto-tuning
- [ ] Advanced visualizations
- [ ] Export capabilities
- [ ] Integration APIs

---

**Built with ❤️ by the AirSense Team**
