import pandas as pd
import streamlit as st
import os

# Define the path to the sample data file relative to this utils.py file
# Assuming a 'data' directory exists at the same level as the main script directory
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_beijing_airquality.csv")

@st.cache_data # Cache the loaded data for performance
def load_sample_data():
    """Loads the sample Beijing air quality dataset."""
    try:
        # Try loading from a predefined path (if you include a sample file)
        # For now, let's simulate loading or create a placeholder
        # In a real scenario, you'd fetch from a URL or load a local file
        # Placeholder: Create a simple DataFrame if file doesn't exist
        if os.path.exists(SAMPLE_DATA_PATH):
             df = pd.read_csv(SAMPLE_DATA_PATH)
             # Basic check for expected columns (example)
             if 'PM2.5' not in df.columns or 'year' not in df.columns:
                 st.warning("Sample data file might be missing expected columns.")
             return df
        else:
            st.warning(f"Sample data file not found at {SAMPLE_DATA_PATH}. Generating placeholder data.")
            # Generate some very basic placeholder data for structure demonstration
            data = {
                'year': [2017] * 24,
                'month': [1] * 24,
                'day': [1] * 24,
                'hour': list(range(24)),
                'PM2.5': [50 + i * 2 for i in range(24)],
                'PM10': [70 + i * 1.5 for i in range(24)],
                'SO2': [10 + i * 0.1 for i in range(24)],
                'NO2': [30 + i * 0.5 for i in range(24)],
                'CO': [800 + i * 10 for i in range(24)],
                'O3': [40 - i * 0.2 for i in range(24)],
                'TEMP': [0 + i * 0.1 for i in range(24)],
                'PRES': [1020 - i * 0.1 for i in range(24)],
                'DEWP': [-5 + i * 0.05 for i in range(24)],
                'RAIN': [0] * 24,
                'WSPM': [1 + i * 0.05 for i in range(24)]
            }
            df = pd.DataFrame(data)
            return df

    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        # Return an empty DataFrame or raise error depending on desired handling
        return pd.DataFrame()

def format_metrics(value, precision=2):
    """Formats numeric values for display in metrics."""
    if isinstance(value, (int, float)):
        return f"{value:,.{precision}f}"
    return str(value)

# Example usage (for testing):
if __name__ == "__main__":
    # This block will only run when utils.py is executed directly
    # Create a dummy data directory and file for testing load_sample_data
    test_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    test_file_path = os.path.join(test_data_dir, "sample_beijing_airquality.csv")
    
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
        
    # Create a dummy CSV if it doesn't exist
    if not os.path.exists(test_file_path):
        print(f"Creating dummy sample file at {test_file_path}")
        dummy_data = {
            'year': [2017] * 5, 'month': [1] * 5, 'day': [1] * 5, 'hour': list(range(5)),
            'PM2.5': [10, 11, 12, 13, 14], 'PM10': [20, 21, 22, 23, 24]
        }
        pd.DataFrame(dummy_data).to_csv(test_file_path, index=False)

    print("Testing load_sample_data...")
    sample_df = load_sample_data()
    print("Sample data loaded:")
    print(sample_df.head())
    
    # Clean up dummy file/dir if created
    # os.remove(test_file_path)
    # os.rmdir(test_data_dir)

