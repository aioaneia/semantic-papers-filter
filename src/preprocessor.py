
import pandas as pd
import re


class Preprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        """
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep important ones
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text

    @staticmethod
    def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataset for analysis.
        """
        df = df.copy()

        # Clean title and abstract
        df['clean_title'] = df['Title'].apply(Preprocessor.clean_text)
        df['clean_abstract'] = df['Abstract'].apply(Preprocessor.clean_text)

        return df
