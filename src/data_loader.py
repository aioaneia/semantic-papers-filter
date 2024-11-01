
import pandas as pd

from typing import Optional


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the CSV data and perform initial validation.
        """
        self.data = pd.read_csv(self.file_path)

        self.validate_data()

        return self.data


    def validate_data(self) -> None:
        """
        Validate required columns exist.
        """
        required_columns = ['PMID', 'Title', 'Abstract']

        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")


    def get_basic_stats(self) -> dict:
        """
        Return basic statistics about the dataset.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return {
            'total_records':         len(self.data),
            'records_with_abstract': self.data['Abstract'].notna().sum(),
            'publication_years':     self.data['Publication Year'].value_counts().sort_index().to_dict(),
            'missing_values':        self.data.isnull().sum().to_dict()
        }
