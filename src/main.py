
import os
import logging

from pathlib import Path

import src.utils.file_utils as file_utils

from src.data.data_loader        import DataLoader
from src.data.preprocessor       import Preprocessor
from src.pipelines.pipeline      import Pipeline
from src.filters.semantic_filter import SemanticFilter
from src.utils.visualization     import StatsVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperFilteringApp:
    """
    Main application class for paper filtering system.
    """
    def __init__(self, config_path: str):
        """
        Initialize the application with configuration.
        """
        self.config = file_utils.load_config(config_path)

        self.logger = logging.getLogger(__name__)

        self.logger.setLevel(logging.INFO)

        self.pipeline = Pipeline(
            data_loader=DataLoader(),
            preprocessor=Preprocessor(),
            semantic_filter=SemanticFilter(
                self.config['SPACY_MODEL'],
                self.config['TRANSFORMER_MODEL']
            ),
            visualizer=StatsVisualizer()
        )

    def run(self):
        """
        Run the paper filtering pipeline.
        """
        try:
            self.setup_output_dirs()

            self.logger.info("Starting paper processing pipeline...")

            # Process papers and get relevant and irrelevant DataFrames
            relevant_df, irrelevant_df, time_taken = self.pipeline.process_papers(
                self.config['DATASET_PATH'],    # Path to dataset
                self.config['DATASET_LIMIT'],   # Limit number of papers
                "../results/files"  # Output directory
            )

            # Generate statistics
            stats = self.pipeline.generate_statistics(
                relevant_df,
                irrelevant_df,
                os.path.join('../results/stats/', 'nlp_statistics.json'),
                time_taken
            )

            # Plot statistics
            self.pipeline.plot_statistics(
                relevant_df,
                irrelevant_df,
                stats,
                "../results/plots"
            )

            self.logger.info("Paper processing complete.")
        except Exception as e:
            print(f"Error during processing: {e}")
            raise e

    @staticmethod
    def setup_output_dirs():
        """
        Create necessary output directories.
        """
        for dir_path in [
            Path("../results"),
            Path("../results/files"),
            Path("../results/stats"),
            Path("../results/plots"),
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


def main():
    """
    Main entry point for the application script to run the pipeline and process papers from the dataset file.
    the configuration file is loaded and the pipeline is initialized with the necessary components.
    """
    app = PaperFilteringApp(
        '../config.yaml'
    )
    app.run()

if __name__ == "__main__":
    main()
