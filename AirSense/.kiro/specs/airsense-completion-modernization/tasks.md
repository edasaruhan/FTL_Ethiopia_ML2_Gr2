# Implementation Plan: AirSense Completion and Modernization

## Overview

This implementation plan converts the AirSense modernization design into actionable coding tasks. The project modernizes an enterprise air quality monitoring system by removing Docker infrastructure, implementing Model Registry and Model Evaluator systems, updating deprecated code patterns (pandas methods, Pydantic v1 to v2), fixing import errors, and creating comprehensive development automation with a Makefile.

**Technology Stack**: Python 3.9-3.12, Apache Spark, FastAPI, Pydantic v2, pandas 2.0+, scikit-learn, TensorFlow, Streamlit

**Implementation Language**: Python

## Tasks

- [x] 1. Remove Docker infrastructure and update documentation
  - Delete Dockerfile and docker-compose.yml files
  - Remove all Docker references from README.md
  - Update README.md to reflect native Python deployment
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10_

- [x] 2. Fix import errors in existing codebase
  - Add `import sys` to src/data/processor.py
  - Add `import os` to src/api/routes.py
  - Organize imports according to PEP 8 (standard library, third-party, local)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 3. Modernize pandas methods to remove deprecation warnings
  - Replace `fillna(method='ffill')` with `ffill()` in src/models/time_series.py
  - Replace `fillna(method='bfill')` with `bfill()` in src/models/time_series.py
  - Replace `fillna(method='ffill')` with `ffill()` in src/api/routes.py
  - Replace `fillna(method='bfill')` with `bfill()` in src/api/routes.py
  - Verify no deprecation warnings with pandas 2.1+
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10_

- [x] 4. Migrate Pydantic schemas to v2 syntax
  - [x] 4.1 Update src/data/schemas.py with complete Pydantic v2 schema definitions
    - Create ForecastRequest schema with ConfigDict and field_validator
    - Create ForecastDataPoint schema
    - Create ForecastResponse schema
    - Create AQILevel enum and AQIResponse schema
    - Create DataQueryRequest schema
    - Create PipelineStatus schema
    - Create ModelRegistryEntry schema
    - Create EvaluationRequest and EvaluationResponse schemas
    - Replace `Field(regex=...)` with `Field(pattern=...)`
    - Replace nested `Config` class with `model_config = ConfigDict(...)`
    - Replace `@validator` with `@field_validator` and `@classmethod`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 7.11, 7.12, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9_
  
  - [x] 4.2 Update src/core/config.py to use Pydantic v2 settings
    - Import `BaseSettings` and `SettingsConfigDict` from `pydantic_settings`
    - Replace nested `Config` class with `model_config = SettingsConfigDict(...)`
    - Add `setup_directories()` method to create required directory structure
    - _Requirements: 8.3, 8.4, 10.6, 10.7_
  
  - [x] 4.3 Update code that uses Pydantic models
    - Replace `.dict()` calls with `.model_dump()` throughout codebase
    - Replace `.json()` calls with `.model_dump_json()` throughout codebase
    - _Requirements: 8.6, 8.7_

- [-] 5. Implement Model Registry system
  - [ ] 5.1 Create src/models/registry.py with ModelMetadata dataclass
    - Define ModelMetadata with all required fields (model_id, model_name, model_type, version, file_path, training_date, parameters, metrics, tags, target_pollutant, training_data_source, feature_set, created_at, updated_at)
    - Implement `to_dict()` method with datetime serialization
    - Implement `from_dict()` classmethod with datetime deserialization
    - _Requirements: 2.1, 2.2, 2.3, 2.10_
  
  - [ ] 5.2 Implement ModelRegistry class in src/models/registry.py
    - Initialize registry with directory structure creation
    - Implement `_load_registry()` method to load from registry.json
    - Implement `_save_registry()` method to persist to registry.json
    - Implement `_generate_model_id()` method for unique ID generation
    - Implement `_generate_version()` method for semantic versioning
    - _Requirements: 2.1, 2.3, 2.9_
  
  - [ ] 5.3 Implement model registration in ModelRegistry
    - Implement `register_model()` method with all parameters
    - Create model directory structure (models/{model_id}/)
    - Save model artifact using joblib
    - Save metadata.json file
    - Update registry.json index
    - Add comprehensive logging
    - _Requirements: 2.1, 2.2, 2.3, 2.7, 2.8, 14.6_
  
  - [ ] 5.4 Implement model retrieval in ModelRegistry
    - Implement `get_model()` method with error handling
    - Validate model exists in registry
    - Validate model file exists on disk
    - Load model using joblib
    - Return tuple of (model, metadata)
    - _Requirements: 2.4, 2.8, 14.1_
  
  - [ ] 5.5 Implement model listing and filtering in ModelRegistry
    - Implement `list_models()` method with optional filters
    - Support filtering by model_type
    - Support filtering by tags
    - Support filtering by target_pollutant
    - Sort results by creation date (newest first)
    - _Requirements: 2.5_
  
  - [ ] 5.6 Implement model management operations in ModelRegistry
    - Implement `delete_model()` method with file cleanup
    - Implement `update_tags()` method
    - Implement `get_latest_version()` method
    - Add error handling for all operations
    - _Requirements: 2.6, 2.9, 14.1_
  
  - [ ]* 5.7 Write unit tests for ModelRegistry
    - Test registry initialization and directory creation
    - Test model registration with all parameters
    - Test model retrieval and error cases
    - Test model listing with filters
    - Test model deletion
    - Test tag updates
    - Test version generation
    - Test registry persistence across instances
    - Test error handling for missing files
    - _Requirements: 11.1_

- [ ] 6. Implement Model Evaluator system
  - [ ] 6.1 Create src/models/evaluator.py with EvaluationResult dataclass
    - Define EvaluationResult with all required fields
    - Implement `to_dict()` method for serialization
    - _Requirements: 3.5, 3.7_
  
  - [ ] 6.2 Implement ModelEvaluator class with input validation
    - Initialize evaluator with logger
    - Implement `_validate_inputs()` method
    - Check for shape mismatches with descriptive errors
    - Check for NaN values with counts
    - Check for infinite values with counts
    - Check minimum sample size (at least 2 samples)
    - _Requirements: 3.8, 3.9, 14.2_
  
  - [ ] 6.3 Implement standard metrics calculation in ModelEvaluator
    - Implement `_calculate_standard_metrics()` method
    - Calculate MAE (Mean Absolute Error)
    - Calculate MSE and RMSE (Root Mean Squared Error)
    - Calculate R² (R-squared) score
    - Calculate MAPE (Mean Absolute Percentage Error) with zero-division handling
    - Calculate max_error, mean_error, std_error
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [ ] 6.4 Implement main evaluate() method in ModelEvaluator
    - Accept y_true, y_pred, and metadata parameters
    - Call validation and metrics calculation
    - Support additional custom metrics
    - Return EvaluationResult object
    - Add comprehensive logging
    - _Requirements: 3.5, 3.10, 14.2_
  
  - [ ] 6.5 Implement custom metrics and model comparison in ModelEvaluator
    - Implement `register_custom_metric()` method
    - Implement `compare_models()` method returning pandas DataFrame
    - Sort comparison by primary metric
    - _Requirements: 3.6, 3.10_
  
  - [ ] 6.6 Implement report generation in ModelEvaluator
    - Implement `generate_report()` method
    - Implement `_generate_performance_summary()` helper
    - Create structured report dictionary
    - Support JSON file output
    - Add performance assessments (Excellent/Good/Fair/Poor)
    - _Requirements: 3.7_
  
  - [ ] 6.7 Implement registry integration in ModelEvaluator
    - Implement `evaluate_model_from_registry()` method
    - Load model from registry
    - Generate predictions
    - Evaluate and return results
    - _Requirements: 3.6_
  
  - [ ]* 6.8 Write unit tests for ModelEvaluator
    - Test basic evaluation with sample data
    - Test perfect predictions (all metrics should be optimal)
    - Test shape mismatch error handling
    - Test NaN value detection
    - Test infinite value detection
    - Test custom metric registration and usage
    - Test model comparison with multiple models
    - Test report generation
    - _Requirements: 11.2_

- [ ] 7. Create comprehensive Makefile for development automation
  - [ ] 7.1 Create Makefile with help target and variable definitions
    - Define PYTHON, VENV, PIP, PYTEST, BLACK, RUFF, MYPY variables
    - Define directory variables (SRC_DIR, TEST_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR)
    - Create help target with all command descriptions
    - _Requirements: 4.9_
  
  - [ ] 7.2 Add installation targets to Makefile
    - Create `install` target for production dependencies
    - Create `install-dev` target for development dependencies
    - Create `check-deps` target to verify Python and Java availability
    - _Requirements: 4.1, 4.2, 4.10_
  
  - [ ] 7.3 Add testing targets to Makefile
    - Create `test` target to run all tests
    - Create `test-cov` target for coverage reporting (HTML, term-missing, XML)
    - Create `test-unit` target for unit tests only
    - Create `test-integration` target for integration tests only
    - _Requirements: 4.3, 4.4_
  
  - [ ] 7.4 Add code quality targets to Makefile
    - Create `lint` target running ruff and mypy
    - Create `format` target running black and ruff --fix
    - Create `format-check` target for CI validation
    - _Requirements: 4.5, 4.6_
  
  - [ ] 7.5 Add cleaning and utility targets to Makefile
    - Create `clean` target to remove Python cache files and build artifacts
    - Create `clean-data` target to remove processed data
    - Create `clean-models` target to remove model files
    - Create `setup-dirs` target to create directory structure with .gitkeep files
    - _Requirements: 4.8, 10.1, 10.2, 10.3, 10.4, 10.5_
  
  - [ ] 7.6 Add service running targets to Makefile
    - Create `run-all` target to start API and dashboard together
    - Create `run-api` target for API server with reload
    - Create `run-dashboard` target for Streamlit dashboard
    - _Requirements: 4.7_
  
  - [ ] 7.7 Add development helper targets to Makefile
    - Create `dev-setup` target combining check-deps, install-dev, setup-dirs
    - Create `ci` target combining lint and test-cov
    - Create `logs` target to tail application logs
    - Create `registry-info` and `registry-list` targets for model registry inspection
    - _Requirements: 4.9, 4.10_

- [ ] 8. Update dependencies in requirements files
  - Add `structlog>=23.1.0` to requirements.txt
  - Add `psutil>=5.9.0` to requirements.txt
  - Add `pydantic-settings>=2.0.0` to requirements.txt for Pydantic v2
  - Update `pydantic>=2.4.0` in requirements.txt
  - Ensure all ML dependencies are properly versioned
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

- [ ] 9. Integrate Model Registry with existing TimeSeriesForecaster
  - [ ] 9.1 Update src/models/time_series.py to import ModelRegistry
    - Add ModelRegistry import
    - Add registry parameter to TimeSeriesForecaster.__init__()
    - Initialize registry instance
    - _Requirements: 2.1_
  
  - [ ] 9.2 Update train_model() to register models after training
    - After successful model training, call registry.register_model()
    - Pass model, model_name, model_type, parameters, metrics, tags
    - Include target_pollutant, training_data_source, feature_set
    - Return model_id in training result
    - _Requirements: 2.1, 2.2, 2.7_
  
  - [ ] 9.3 Add load_model_from_registry() method to TimeSeriesForecaster
    - Accept model_id parameter
    - Call registry.get_model(model_id)
    - Store loaded model in self.models dictionary
    - Log successful load
    - _Requirements: 2.4_

- [ ] 10. Add API endpoints for Model Registry
  - [ ] 10.1 Create GET /models/registry endpoint in src/api/routes.py
    - List all registered models with optional filters
    - Support query parameters: model_type, tags, target_pollutant
    - Return list of ModelRegistryEntry schemas
    - _Requirements: 2.5_
  
  - [ ] 10.2 Create GET /models/registry/{model_id} endpoint
    - Retrieve specific model metadata
    - Return ModelRegistryEntry schema
    - Handle model not found with 404
    - _Requirements: 2.4_
  
  - [ ] 10.3 Create DELETE /models/registry/{model_id} endpoint
    - Delete model and its files
    - Return success message
    - Handle errors appropriately
    - _Requirements: 2.6_
  
  - [ ] 10.4 Create PUT /models/registry/{model_id}/tags endpoint
    - Update model tags
    - Accept list of tags in request body
    - Return updated metadata
    - _Requirements: 2.9_

- [ ] 11. Add API endpoints for Model Evaluator
  - [ ] 11.1 Create POST /models/evaluate endpoint in src/api/routes.py
    - Accept EvaluationRequest schema
    - Load model from registry
    - Load test data
    - Perform evaluation
    - Return EvaluationResponse schema
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [ ] 11.2 Create POST /models/compare endpoint
    - Accept list of model_ids
    - Evaluate all models on same test data
    - Use ModelEvaluator.compare_models()
    - Return comparison DataFrame as JSON
    - _Requirements: 3.6_

- [ ] 12. Update application startup to create directory structure
  - [ ] 12.1 Add startup event handler in src/api/main.py
    - Import settings
    - Call settings.setup_directories() on startup
    - Log directory creation
    - _Requirements: 10.6, 10.7, 10.8_
  
  - [ ] 12.2 Create .gitkeep files for empty directories
    - Create data/raw/.gitkeep
    - Create data/processed/.gitkeep
    - Create models/.gitkeep
    - Create logs/.gitkeep
    - _Requirements: 10.5_

- [ ] 13. Checkpoint - Verify core functionality
  - Run `make test` to ensure all unit tests pass
  - Run `make lint` to verify code quality
  - Test model registration and retrieval manually
  - Test model evaluation manually
  - Ensure all tests pass, ask the user if questions arise.

- [ ]* 14. Write integration tests for complete workflows
  - [ ]* 14.1 Create tests/integration/test_model_lifecycle.py
    - Test complete model lifecycle: train → register → load → evaluate
    - Test model registry persistence across instances
    - Test error handling in full workflow
    - _Requirements: 11.3_
  
  - [ ]* 14.2 Create tests/integration/test_api_endpoints.py
    - Test forecast endpoint with model registry integration
    - Test model registry API endpoints
    - Test model evaluation API endpoints
    - Test error responses and validation
    - _Requirements: 11.3_
  
  - [ ]* 14.3 Create tests/integration/test_pipeline.py
    - Test end-to-end data processing pipeline
    - Test data loading → processing → model training → evaluation
    - Verify no deprecation warnings
    - _Requirements: 11.3, 11.4_

- [ ] 15. Update documentation
  - [ ] 15.1 Update README.md with new features and installation
    - Remove all Docker references (installation, deployment, services table, badge)
    - Add Model Registry usage section with examples
    - Add Model Evaluator usage section with examples
    - Document all Makefile targets
    - Update Quick Start section for native Python
    - Update Architecture section to reflect changes
    - Document Python version compatibility (3.9-3.12)
    - Add directory structure documentation
    - _Requirements: 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 10.10_
  
  - [ ] 15.2 Create CHANGELOG.md
    - Document all changes in this modernization
    - List added features (Model Registry, Model Evaluator, Makefile)
    - List changed features (Pydantic v2, pandas methods, Docker removal)
    - List fixed issues (import errors, deprecation warnings)
    - Include migration notes
    - _Requirements: 12.10, 13.8_
  
  - [ ] 15.3 Add inline documentation to new modules
    - Add comprehensive docstrings to ModelRegistry class and methods
    - Add comprehensive docstrings to ModelEvaluator class and methods
    - Add type hints to all function parameters and return values
    - Add inline comments for complex logic
    - _Requirements: 12.7, 12.8, 12.9_
  
  - [ ] 15.4 Document Pydantic v2 migration
    - Add migration guide to README.md or separate MIGRATION.md
    - Document breaking changes
    - Provide before/after examples
    - _Requirements: 12.5, 13.8_

- [ ] 16. Add comprehensive error handling and logging
  - [ ] 16.1 Create custom exception classes in src/core/exceptions.py
    - Define AirSenseError base exception
    - Define RegistryError for model registry operations
    - Define EvaluationError for model evaluation operations
    - Define ModelError for model operations
    - Define ValidationError for data validation
    - _Requirements: 14.1, 14.2_
  
  - [ ] 16.2 Add error handling to Model Registry operations
    - Add try-except blocks with descriptive error messages
    - Log errors with context (model_id, file paths)
    - Raise RegistryError for registry-specific failures
    - _Requirements: 14.1, 14.7_
  
  - [ ] 16.3 Add error handling to Model Evaluator operations
    - Add try-except blocks in evaluate() method
    - Log evaluation failures with context
    - Raise EvaluationError for evaluation failures
    - _Requirements: 14.2, 14.7_
  
  - [ ] 16.4 Add error handling to API endpoints
    - Wrap endpoint logic in try-except blocks
    - Return appropriate HTTP status codes (400, 404, 500)
    - Provide descriptive error messages
    - Log all errors with request context
    - _Requirements: 14.5, 14.7_
  
  - [ ] 16.5 Add structured logging throughout codebase
    - Use structlog for all logging
    - Log model training operations with parameters and results
    - Log API requests with request IDs
    - Log data processing operations with record counts
    - Support configurable log levels
    - _Requirements: 14.6, 14.7, 14.8, 14.9, 14.10_

- [ ] 17. Add performance monitoring and optimization
  - [ ] 17.1 Add performance logging decorator
    - Create @log_performance decorator in src/core/logging.py
    - Measure and log function execution time
    - Apply to model training, evaluation, and registry operations
    - _Requirements: 15.1, 15.2, 15.3_
  
  - [ ] 17.2 Implement caching for Model Registry
    - Cache frequently accessed model metadata in memory
    - Implement cache invalidation on updates
    - _Requirements: 15.1, 15.6_
  
  - [ ] 17.3 Add performance benchmarks to test suite
    - Test Model Registry retrieval performance (<100ms)
    - Test Model Evaluator performance on 10,000 samples (<1s)
    - Test data processing pipeline performance
    - _Requirements: 15.1, 15.2, 15.3, 15.10_

- [ ] 18. Add input validation and security measures
  - [ ] 18.1 Add input validation to all API endpoints
    - Validate all inputs using Pydantic schemas
    - Add range validation for numeric inputs
    - Add pattern validation for string inputs
    - _Requirements: 16.1, 16.7_
  
  - [ ] 18.2 Add file path sanitization
    - Sanitize file paths in Model Registry to prevent directory traversal
    - Validate model names contain only alphanumeric and underscores
    - Validate file paths are within expected directories
    - _Requirements: 16.2, 16.5_
  
  - [ ] 18.3 Add model file integrity validation
    - Validate model files exist before loading
    - Check file sizes to prevent DoS
    - Validate file extensions
    - _Requirements: 16.3, 16.4_
  
  - [ ] 18.4 Add security logging
    - Log all validation failures
    - Log file access attempts
    - Log authentication events (when implemented)
    - _Requirements: 16.8_

- [ ] 19. Final checkpoint - Complete system verification
  - Run full test suite with coverage: `make test-cov`
  - Verify coverage ≥80% for new components
  - Run linting: `make lint`
  - Run formatting check: `make format-check`
  - Test complete installation from scratch in clean environment
  - Test all Makefile targets
  - Verify no deprecation warnings with Python 3.12 and pandas 2.1+
  - Test API endpoints manually or with integration tests
  - Verify documentation is complete and accurate
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 20. Prepare for deployment
  - [ ] 20.1 Create deployment documentation
    - Document installation process
    - Document service startup procedures
    - Document monitoring and logging
    - Document maintenance tasks
    - _Requirements: 12.11_
  
  - [ ] 20.2 Verify backward compatibility
    - Test existing API clients work without changes
    - Test existing model files load successfully
    - Test existing configuration files work
    - Test existing data files process correctly
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_
  
  - [ ] 20.3 Create migration guide for existing deployments
    - Document backup procedures
    - Document update steps
    - Document breaking changes
    - Document rollback procedures
    - _Requirements: 13.7, 13.8, 13.9_

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints (tasks 13 and 19) ensure incremental validation
- The implementation follows a logical dependency order:
  1. Clean up (Docker removal, import fixes, pandas modernization)
  2. Core infrastructure (Pydantic v2, dependencies)
  3. New features (Model Registry, Model Evaluator)
  4. Integration (API endpoints, TimeSeriesForecaster integration)
  5. Testing and documentation
  6. Error handling and security
  7. Performance and deployment readiness
- All code should include comprehensive error handling and logging
- All public functions should have docstrings and type hints
- Follow PEP 8 style guidelines throughout
