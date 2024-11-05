
import yaml
import os
import logging
from pathlib import Path

from src.data.data_loader        import DataLoader
from src.data.preprocessor       import Preprocessor
from src.pipelines.pipeline      import Pipeline
from src.filters.semantic_filter import SemanticFilter
from src.utils.visualization     import StatsVisualizer


class PaperFilteringApp:
    """
    Main application class for paper filtering system.
    """

    def __init__(self, config_path: str):
        """
        Initialize the application with configuration.
        """
        self.config         = self.load_config(config_path)
        self.logger         = logging.getLogger(__name__)
        self.data_loader    = DataLoader()
        self.preprocessor   = Preprocessor()
        self.nlp_classifier = SemanticFilter(self.config['SPACY_MODEL'], self.config['TRANSFORMER_MODEL'])
        self.visualizer     = StatsVisualizer()

        self.pipeline = Pipeline(
            data_loader=self.data_loader,
            preprocessor=self.preprocessor,
            semantic_filter=self.nlp_classifier,
            visualizer=self.visualizer
        )


    def run(self):
        """
        Run the paper filtering pipeline.
        """
        try:
            self.setup_output_dirs()

            self.logger.info("Starting paper processing pipeline...")

            # Process papers and get relevant and irrelevant DataFrames
            relevant_df, irrelevant_df = self.pipeline.process_papers(
                self.config['DATASET_PATH'],    # Path to dataset
                self.config['DATASET_LIMIT'],   # Limit number of papers
                "../results/files"  # Output directory
            )

            # Generate statistics
            stats = self.pipeline.generate_statistics(
                relevant_df,
                irrelevant_df,
                os.path.join('../results/stats/', 'nlp_statistics.json')
            )

            # Plot statistics
            self.pipeline.plot_statistics(
                relevant_df,
                stats,
                "../results/plots"
            )

            self.logger.info("Paper processing complete.")
        except Exception as e:
            print(f"Error during processing: {e}")
            raise

    @staticmethod
    def load_config(config_path: str) -> dict:
        """
        Load configuration from YAML file.
        """
        try:
            with open(config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}")


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
    Main entry point for the application.
    """
    app = PaperFilteringApp(
        '../config.yaml'
    )

    app.run()


if __name__ == "__main__":
    main()
