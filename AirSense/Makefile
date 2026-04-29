# AirSense Makefile
# Enterprise Air Quality Analysis System

.PHONY: help install install-dev test test-cov lint format clean build run run-api run-dashboard docker-build docker-up docker-down

# Default target
help:
	@echo "AirSense - Enterprise Air Quality Analysis System"
	@echo ""
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run code quality checks"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Clean cache and temporary files"
	@echo "  build        Build the application"
	@echo "  run          Run the complete system"
	@echo "  run-api      Run API server only"
	@echo "  run-dashboard Run dashboard only"
	@echo "  docker-build Build Docker images"
	@echo "  docker-up    Start services with Docker Compose"
	@echo "  docker-down  Stop Docker Compose services"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-watch:
	pytest tests/ -v -f

# Code quality
lint:
	flake8 src/ tests/
	mypy src/
	bandit -r src/
	safety check

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Building
build:
	python -m build

# Running
run:
	python run_all.py

run-api:
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	streamlit run frontend/dashboard.py --server.port 8501

# Docker
docker-build:
	docker build -t airsense:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-shell:
	docker-compose exec api bash

# Development helpers
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

ci: lint test-cov
	@echo "CI pipeline completed successfully!"

# Production
prod-build:
	docker build -t airsense:$(shell git rev-parse --short HEAD) .

prod-deploy:
	@echo "Deploy to production environment"
	# Add deployment commands here

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

# Monitoring
logs:
	tail -f logs/airsense_*.log

monitor:
	python -m src.monitoring.metrics_server
