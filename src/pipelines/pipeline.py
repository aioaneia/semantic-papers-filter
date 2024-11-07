
import os
import json
import logging
import time
from logging import Logger

from typing import Optional, Tuple

import pandas as pd


class Pipeline:
    """
    Main pipeline class for processing papers.
    """
    logger: Logger

    def __init__(self, data_loader, preprocessor, semantic_filter, visualizer):
        """
        Initialize the pipeline with the specified components.
        """
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.semantic_filter = semantic_filter
        self.visualizer = visualizer
        self.logger = logging.getLogger(__name__)

        self.logger.setLevel(logging.INFO)


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
            self.logger.info(f'Data loaded in {time.time() - start_time:.2f} seconds.')

            # Limit the number of papers if specified
            if limit is not None:
                df = df.head(limit)
                self.logger.info(f'Data limited to {limit} papers.')

            # Prepare the dataset for processing
            df = self.preprocessor.prepare_dataset(df)

            self.logger.info('Starting semantic filtering...')
            start_time = time.time()

            # Combine title and abstract for processing
            df['combined_text'] = df['clean_title'] + ' ' + df['clean_abstract']

            # Apply semantic filtering to the combined text
            df['is_relevant'], df['scores'] = zip(*df['combined_text'].apply(
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
                'Method Type', 'Method Name', 'scores',
            ]

            columns_to_keep_for_irrelevant_df = [
                'PMID', 'Title', 'Abstract', 'Publication Year', 'Journal/Book', 'First Author',
                'scores',
            ]

            # Keep only the specified columns
            relevant_df_to_save = relevant_df[columns_to_keep_for_relevant_df]
            irrelevant_dt_to_save = irrelevant_df[columns_to_keep_for_irrelevant_df]

            # Save relevant and irrelevant papers to Excel files
            relevant_papers_path = os.path.join(output_dir, 'relevant_papers.xlsx')
            irrelevant_papers_path = os.path.join(output_dir, 'filtered_out_papers.xlsx')

            relevant_df_to_save.to_excel(relevant_papers_path, index=False)
            irrelevant_dt_to_save.to_excel(irrelevant_papers_path, index=False)

            self.logger.info(f'Results saved to {relevant_papers_path} and {irrelevant_papers_path}')

            self.logger.info(f'Semantic filtering completed in {time.time() - start_time:.2f} seconds.')

            self.logger.info(f'Average time per paper: {(time.time() - start_time) / len(df):.2f} seconds.')

            return relevant_df_to_save, irrelevant_dt_to_save, time.time() - start_time

        except Exception as e:
            self.logger.error(f"Error in process_papers: {e}")
            raise


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
        method_counts = relevant_df['Method Type'].value_counts().to_dict()
        method_percentages = {
            k: round((v / total_relevant_papers) * 100, 2)
            for k, v in method_counts.items()
        }

        # Top method names overall
        relevant_df_non_empty_methods = relevant_df[
            relevant_df['Method Name'].notna() & (relevant_df['Method Name'] != '')]
        top_methods_overall = relevant_df_non_empty_methods['Method Name'].value_counts().head(10).to_dict()


        # Prepare 'year' column
        relevant_df['year'] = pd.to_numeric(relevant_df['Publication Year'], errors='coerce')
        relevant_df = relevant_df.dropna(subset=['year'])
        relevant_df['year'] = relevant_df['year'].astype(int)

        # Top method names by year
        top_methods_by_year = {}
        for year in sorted(relevant_df['year'].unique()):
            year_df = relevant_df[relevant_df['year'] == year]
            methods_in_year = year_df['Method Name'].dropna().replace('', pd.NA).dropna()
            top_methods = methods_in_year.value_counts().head(5).to_dict()
            if top_methods:
                top_methods_by_year[int(year)] = top_methods

        stats = {
            'total_relevant_papers': total_relevant_papers,
            'total_papers_processed': total_papers_processed,
            'relevance_percentage': round((total_relevant_papers / total_papers_processed) * 100, 2),
            'method_counts': method_counts,
            'method_percentages': method_percentages,
            'top_method_names_overall': top_methods_overall,
            'top_method_names_by_year': top_methods_by_year,
            'time_taken': time_taken,
        }

        with open(output_file_path, "w") as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f'Statistics generated and saved to {output_file_path}')

        return stats


    def plot_statistics(self, relevant_df: pd.DataFrame, irrelevant_df: pd.DataFrame,
                        statistics: dict, output_dir: str):
        """
        Plot statistics using the relevant papers DataFrame and statistics dictionary.
        """
        try:
            method_counts      = statistics['method_counts']
            method_percentages = statistics['method_percentages']
            top_method_names   = statistics['top_method_names_overall']

            # Prepare 'year' column
            relevant_df['year'] = pd.to_numeric(relevant_df['Publication Year'], errors='coerce').fillna(0).astype(int)

            self.visualizer.plot_method_type_percentage(method_percentages, output_dir)

            self.visualizer.plot_method_type_distribution_over_time(relevant_df, output_dir)

            self.visualizer.plot_trend_of_top_method_names_over_time(top_method_names, relevant_df, output_dir)

            self.visualizer.plot_top_journals_by_method_type(relevant_df, output_dir)

            self.visualizer.plot_top_authors(relevant_df, output_dir)

            self.visualizer.plot_publications_per_journal_over_time(relevant_df, output_dir)

            ## A pie chart of the distribution of relevant papers per journal
            self.visualizer.plot_publication_distribution_per_journal(relevant_df, output_dir)

            ## Plotting relevant vs irrelevant papers comparison
            self.visualizer.plot_journal_comparison(relevant_df, irrelevant_df, output_dir)

            # self.visualizer.plot_irrelevant_papers_per_year(irrelevant_df, output_dir)

            self.visualizer.plot_irrelevance_scores_distribution(irrelevant_df, output_dir)

            ## A network plot of method occurrence in relevant papers (Method Name)
            self.visualizer.plot_method_occurrence_network(relevant_df, output_dir, text_column='Method Name')

            ## A cloud plot of method occurrence in relevant papers (Method Name)
            self.visualizer.plot_word_cloud(relevant_df, output_dir, text_column='Method Name')

            self.logger.info('Plots generated.')

        except Exception as e:
            self.logger.error(f'Error in plot_statistics: {e}')
            raise

