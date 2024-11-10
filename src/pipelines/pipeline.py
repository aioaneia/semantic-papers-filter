
import os
import json
import logging
import time
import pandas as pd

from logging import Logger
from typing import Optional, Tuple


class Pipeline:
    """
    Main pipeline class for processing papers.
    """
    logger: Logger

    def __init__(self, data_loader, preprocessor, semantic_filter, visualizer):
        """
        Initialize the pipeline with the specified components.
        """
        self.data_loader     = data_loader
        self.preprocessor    = preprocessor
        self.semantic_filter = semantic_filter
        self.visualizer      = visualizer
        self.logger          = logging.getLogger(__name__)


    def process_papers(self, dataset_path: str, limit: Optional[int] = None, output_dir: str = '../results/files') \
            -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Process the papers and save the results to files.

        Returns:
            relevant_df: DataFrame of relevant papers
            irrelevant_df: DataFrame of irrelevant papers
        """
        try:
            start_time: float = time.time()

            df = self.data_loader.load_csv_data(dataset_path)
            self.logger.info(f'Data loaded in {time.time() - start_time:.2f} seconds.\n')

            # Limit the number of papers if specified
            if limit > 0:
                df = df.head(limit)
                self.logger.info(f'Data limited to {limit} papers.\n')

            # Prepare the dataset for processing
            df = self.preprocessor.prepare_dataset(df)

            self.logger.info('Starting semantic filtering...\n')
            start_time = time.time()

            # Combine title and abstract for processing
            df['combined_text'] = df['clean_title'] + ' ' + df['clean_abstract'] + ' ' + df['Journal/Book']

            # Apply semantic filtering to the combined text
            df['is_relevant'], df['scores'], df['reasoning'] = zip(*df['combined_text'].apply(
                lambda x: self.semantic_filter.is_semantic_relevant_by_pattern_matching(x)
                if isinstance(x, str) else (False, {})
            ))

            # Filter out irrelevant papers and keep relevant papers
            relevant_df = df[df['is_relevant']].copy()
            irrelevant_df = df[~df['is_relevant']].copy()

            # Classify method type for relevant papers
            relevant_df['Method Type'] = relevant_df['combined_text'].apply(self.semantic_filter.classify_method)

            # Extract method name for relevant papers
            relevant_df['Method Name'] = relevant_df['combined_text'].apply(self.semantic_filter.extract_method_name)

            # Select the columns to keep in the final output
            columns_to_keep_for_relevant_df = [
                'PMID', 'Title', 'Abstract', 'Publication Year', 'Journal/Book', 'First Author',
                'Method Type', 'Method Name', 'scores', 'reasoning'
            ]

            columns_to_keep_for_irrelevant_df = [
                'PMID', 'Title', 'Abstract', 'Publication Year', 'Journal/Book', 'First Author',
                'scores', 'reasoning'
            ]

            # Keep only the specified columns
            relevant_df_to_save = relevant_df[columns_to_keep_for_relevant_df]
            irrelevant_dt_to_save = irrelevant_df[columns_to_keep_for_irrelevant_df]

            # Save relevant and irrelevant papers to Excel files
            relevant_papers_path = os.path.join(output_dir, 'relevant_papers.xlsx')
            irrelevant_papers_path = os.path.join(output_dir, 'filtered_out_papers.xlsx')

            relevant_df_to_save.to_excel(relevant_papers_path, index=False)
            irrelevant_dt_to_save.to_excel(irrelevant_papers_path, index=False)

            self.logger.info(f'Results saved to {relevant_papers_path} and {irrelevant_papers_path}.\n')

            self.logger.info(f'Semantic filtering completed in {time.time() - start_time:.2f} seconds.\n')

            self.logger.info(f'Average time per paper: {(time.time() - start_time) / len(df):.2f} seconds.\n')

            return relevant_df_to_save, irrelevant_dt_to_save, time.time() - start_time

        except Exception as e:
            self.logger.error(f"Error in process_papers: {e}")
            raise e


    def generate_statistics(self, relevant_df: pd.DataFrame, irrelevant_df: pd.DataFrame, output_file_path: str, time_taken: float) -> dict:
        """
        Generate statistics from the processed results and save to a JSON file.

        Returns:
            stats: Dictionary containing statistics
        """
        if relevant_df.empty:
            self.logger.warning('No relevant papers found. Skipping statistics generation.')
            return

        total_relevant_papers = len(relevant_df)
        total_papers_processed = total_relevant_papers + len(irrelevant_df)

        # Method type distribution
        method_type_counts = relevant_df['Method Type'].value_counts().to_dict()
        method_type_percentages = {
            k: round((v / total_relevant_papers) * 100, 2)
            for k, v in method_type_counts.items()
        }

        # Top method names overall
        relevant_df_method_names = relevant_df[relevant_df['Method Name'].notna() & (relevant_df['Method Name'] != '')]

        top_method_names = (
            relevant_df_method_names['Method Name']
                .apply(self.clean_method_names)
                .explode()
                .value_counts()
                .head(15)
                .to_dict()
        )

        # Aggregate reasoning for irrelevant papers
        reasoning_counts = irrelevant_df['reasoning'].value_counts().to_dict()

        stats = {
            'total_relevant_papers':   total_relevant_papers,
            'total_papers_processed':  total_papers_processed,
            'relevance_percentage':    round((total_relevant_papers / total_papers_processed) * 100, 2),
            'method_type_counts':      method_type_counts,
            'method_type_percentages': method_type_percentages,
            'top_method_names':        top_method_names,
            'reasoning_counts':        reasoning_counts,
            'time_taken':              time_taken,
        }

        with open(output_file_path, "w") as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f'Statistics generated and saved to {output_file_path}.\n')

        return stats


    @staticmethod
    def clean_method_names(method_names_str):
        method_names_str = method_names_str.lower()

        tokens = [token.strip() for token in method_names_str.split(',')]

        standardized_methods = []
        for token in tokens:
            standardized_methods.append(token)

        # Remove duplicates
        unique_methods = list(set(standardized_methods))

        return unique_methods


    def plot_statistics(self, relevant_df: pd.DataFrame, irrelevant_df: pd.DataFrame,
                        statistics: dict, output_dir: str):
        """
        Plot statistics using the relevant papers DataFrame and statistics dictionary.
        """
        try:
            method_percentages = statistics['method_type_percentages']
            top_method_names   = statistics['top_method_names']

            self.visualizer.plot_method_type_percentage(method_percentages, output_dir)

            self.visualizer.plot_trend_of_top_method_names_over_time(top_method_names, relevant_df, output_dir)

            self.visualizer.plot_top_journals_by_method_type(relevant_df, output_dir)

            self.visualizer.plot_top_authors(relevant_df, output_dir)

            self.visualizer.plot_publications_per_journal_over_time(relevant_df, output_dir)

            ## A pie chart of the distribution of relevant papers per journal
            self.visualizer.plot_publication_distribution_per_journal(relevant_df, output_dir)

            ## Plotting relevant vs irrelevant papers comparison
            self.visualizer.plot_journal_comparison(relevant_df, irrelevant_df, output_dir)

            ## A network plot of method occurrence in relevant papers (Method Name)
            # self.visualizer.plot_method_occurrence_network(relevant_df, output_dir, text_column='Method Name')

            ## A cloud plot of method occurrence in relevant papers (Method Name)
            self.visualizer.plot_word_cloud(relevant_df, output_dir, text_column='Method Name')

            self.visualizer.plot_irrelevant_papers_by_reasoning(irrelevant_df, output_dir)

            self.logger.info('Plots generated successfully.\n')

        except Exception as e:
            self.logger.error(f'Error in plot_statistics: {e}')
            raise

