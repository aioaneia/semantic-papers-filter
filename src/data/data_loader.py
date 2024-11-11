
import pandas as pd


class DataLoader:
    """
    Class for loading and saving data.
    """

    def load_csv_data(self, file_path) -> pd.DataFrame:
        """
        Load the CSV data and perform initial validation.
        """
        data = pd.read_csv(file_path)

        self.validate_data(data)

        return data


    def load_xlsx_data(self, file_path) -> pd.DataFrame:
        """
        Load the Excel data and perform initial validation.
        """
        data = pd.read_excel(file_path)

        self.validate_data(data)

        return data


    @staticmethod
    def save_xlsx_data(data: pd.DataFrame, file_path: str) -> None:
        """
        Save the data to an Excel file.
        """
        data.to_excel(file_path, index=False)


    @staticmethod
    def validate_data(data) -> None:
        """
        Validate required columns exist.
        """
        required_columns = [
            'PMID',
            'Title',
            'Abstract',
            'Publication Year'
        ]

        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
