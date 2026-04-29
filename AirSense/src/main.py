"""Main entry point for AirSense system."""

import asyncio
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import get_settings
from src.core.logging import get_logger
from src.data.processor import SparkDataProcessor
from src.models import TimeSeriesForecaster
from src.api.main import create_app


def run_pipeline():
    """Run data processing pipeline."""
    logger = get_logger(__name__)
    settings = get_settings()
    
    try:
        logger.info("Starting AirSense data pipeline")
        
        # Initialize processor
        processor = SparkDataProcessor()
        
        # Find input file
        input_file = Path(settings.raw_data_dir) / "beijing_demo.csv"
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return False
        
        # Generate output path
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(settings.processed_data_dir) / f"processed_{timestamp}.parquet"
        
        # Run pipeline
        result = processor.process_pipeline(str(input_file), str(output_path))
        
        if result["status"] == "success":
            logger.info("Pipeline completed successfully", **result)
            return True
        else:
            logger.error("Pipeline failed", **result)
            return False
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return False
    
    finally:
        if 'processor' in locals():
            processor.stop()


def run_api():
    """Run API server."""
    import uvicorn
    
    logger = get_logger(__name__)
    settings = get_settings()
    
    logger.info("Starting AirSense API server")
    
    try:
        app = create_app()
        uvicorn.run(
            app,
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.api_reload,
            workers=1 if settings.api_reload else settings.api_workers
        )
    except Exception as e:
        logger.error(f"API server failed: {e}")
        return False


def run_dashboard():
    """Run Streamlit dashboard."""
    import subprocess
    import sys
    
    logger = get_logger(__name__)
    
    logger.info("Starting AirSense dashboard")
    
    try:
        dashboard_path = Path(__file__).parent.parent / "frontend" / "dashboard.py"
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path), "--server.port", "8501"
        ])
    except Exception as e:
        logger.error(f"Dashboard failed: {e}")
        return False


def run_all():
    """Run complete system."""
    import threading
    import time
    import webbrowser
    
    logger = get_logger(__name__)
    
    logger.info("Starting complete AirSense system")
    
    # Run pipeline first
    if not run_pipeline():
        logger.error("Pipeline failed, aborting startup")
        return False
    
    # Start API in background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Wait for API to start
    time.sleep(5)
    
    # Start dashboard in background thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    
    # Wait for dashboard to start
    time.sleep(3)
    
    # Open browser
    try:
        webbrowser.open("http://localhost:8501")
        logger.info("Dashboard opened in browser")
    except Exception as e:
        logger.warning(f"Could not open browser: {e}")
    
    logger.info("AirSense system is running")
    logger.info("Dashboard: http://localhost:8501")
    logger.info("API: http://localhost:8000")
    logger.info("API Docs: http://localhost:8000/docs")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down AirSense system")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AirSense - Enterprise Air Quality Analysis System")
    parser.add_argument(
        "command",
        choices=["pipeline", "api", "dashboard", "all"],
        help="Command to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Override settings if needed
    if args.debug:
        import os
        os.environ["DEBUG"] = "true"
    
    # Run command
    if args.command == "pipeline":
        success = run_pipeline()
    elif args.command == "api":
        success = run_api()
    elif args.command == "dashboard":
        success = run_dashboard()
    elif args.command == "all":
        success = run_all()
    else:
        print(f"Unknown command: {args.command}")
        success = False
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
