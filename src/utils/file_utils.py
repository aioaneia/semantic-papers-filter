import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def safe_convert(val):
    """Safely convert any value to a string format."""
    if isinstance(val, np.ndarray):
        return str(val.tolist())
    if pd.isna(val):
        return ""
    return str(val)


def convert_csv_to_excel(csv_file_path: str, excel_file_path: str, limit: int = None) -> str:
    """
    Convert a CSV file to an Excel file with optional row limit.

    Args:
        csv_file_path (str): Path to the input CSV file
        excel_file_path (str): Path to save the output Excel file
        limit (int, optional): Number of rows to process. If None, process all rows

    Returns:
        str: Status message
    """
    try:
        logger.info(f"Reading CSV file: {csv_file_path}")

        # Read CSV file with limit if specified
        if limit:
            df = pd.read_csv(csv_file_path, nrows=limit)
            logger.info(f"Reading first {limit} rows")
        else:
            df = pd.read_csv(csv_file_path)

        logger.info(f"Total rows read: {len(df)}")

        # Convert all columns to string, handling numpy arrays
        for col in df.columns:
            df[col] = df[col].apply(safe_convert)

        # If limit is specified, also save a test CSV file
        if limit:
            # Create test file paths
            base_path = os.path.dirname(csv_file_path)
            test_csv_path = os.path.join(base_path, f'test_{limit}_rows.csv')
            test_excel_path = os.path.join(base_path, f'test_{limit}_rows.xlsx')

            # Save test CSV
            logger.info(f"Saving test CSV file: {test_csv_path}")
            df.to_csv(test_csv_path, index=False)

            # Save test Excel
            logger.info(f"Saving test Excel file: {test_excel_path}")
            df.to_excel(test_excel_path, index=False, engine='openpyxl')

            return f"""Conversion successful with {limit} rows.
            Test CSV saved at: {test_csv_path}
            Test Excel saved at: {test_excel_path}"""

        # If no limit, save full Excel file
        logger.info(f"Writing full Excel file: {excel_file_path}")
        df.to_excel(excel_file_path, index=False, engine='openpyxl')

        return f"Full conversion successful. Excel file saved at: {excel_file_path}"

    except Exception as e:
        error_msg = f"Error during conversion: {str(e)}"
        logger.error(error_msg)
        raise


if __name__ == '__main__':
    try:
        # Get absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(current_dir, '..', '..', 'data', 'collection_with_abstracts.csv')
        excel_file_path = os.path.join(current_dir, '..', '..', 'data', 'collection_with_abstracts.xlsx')

        # Convert only first 25 rows
        limit = 40

        # Convert the CSV to Excel with limit
        status = convert_csv_to_excel(csv_file_path, excel_file_path, limit=limit)
        print(status)

    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")