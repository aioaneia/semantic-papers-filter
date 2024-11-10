
import time
import asyncio

import yaml
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.data.data_loader import DataLoader
from src.filters.semantic_filter import SemanticFilter
from src.filters.llm_semantic_filter import LLMSemanticFilter, ClassificationResult

logging.basicConfig(level=logging.INFO, format='%(message)s')


class SemanticFilterTest:
    """
    Test the semantic filtering process
    """

    def __init__(self):
        self.data_loader = DataLoader()
        self.logger = logging.getLogger(__name__)

        with open('../../config.yaml', 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        self.semantic_filter = SemanticFilter(
            spacy_model='en_core_web_lg',
            transformer_model='all-MiniLM-L6-v2'
        )


    def test_two(self):
        """
        Test the semantic filtering process.
        """
        df = self.data_loader.load_xlsx_data(self.config['TEST_DATASET_PATH'])
        start_time = time.time()

        # Initialize new columns for NLP predictions
        df['Is Relevant'] = False
        df['Scores'] = ''
        df['Is Relevant by Similarity'] = False
        df['Similarity Scores'] = ''
        df['Method Type'] = ''
        df['Method Name'] = ''
        df['DL Similarity Score'] = 0.0

        self.logger.info('Starting semantic filtering...')

        for index, row in tqdm(df.iterrows(), total=len(df), desc='Processing Papers'):
            combined_text = f"{row['Title']} {row['Abstract']}"

            is_relevant, scores = self.semantic_filter.is_semantic_relevant_by_pattern_matching(combined_text)

            is_relevant_by_similarity, similarity_scores = self.semantic_filter.is_semantic_relevant_by_similarity(combined_text)

            # Update dataframe with results
            df.at[index, 'Is Relevant'] = is_relevant
            df.at[index, 'Scores'] = str(scores)

            df.at[index, 'Is Relevant by Similarity'] = is_relevant_by_similarity
            df.at[index, 'Similarity Scores'] = str(similarity_scores)
            df.at[index, 'DL Similarity Score'] = similarity_scores.get('dl_similarity_score', 0.0)

            method_type = ''
            method_name = ''

            if is_relevant:
                # Classify method type
                method_type = self.semantic_filter.classify_method(combined_text)
                method_name = self.semantic_filter.extract_method_name(combined_text)

                # Update dataframe
                df.at[index, 'Method Type'] = method_type
                df.at[index, 'Method Name'] = method_name

            self.logger.info("=====================")
            self.logger.info(f'Paper {index + 1}/{len(df)}')
            self.logger.info(f'Title:       {row["Title"]}')
            self.logger.info(f'Abstract:    {row["Abstract"]}')
            self.logger.info(f'Year:        {row["Publication Year"]}')
            self.logger.info(f'Journal:     {row["Journal/Book"]}')

            # Ground truth relevance
            self.logger.info(f'Ground truth: {row["Ground Is Relevant"]}')
            self.logger.info(f'Ground truth method type: {row["Ground Method Type"]}')
            self.logger.info(f'Ground truth method name: {row["Ground Method Name"]}')

            # Predictions from NLP
            self.logger.info(f'Is relevant: {is_relevant}')
            self.logger.info(f'Scores:      {scores}')
            self.logger.info(f'Method type: {method_type}')
            self.logger.info(f'Method name: {method_name}')

            # Predictions from similarity
            self.logger.info(f'By similarity: {is_relevant_by_similarity}')
            self.logger.info(f'Similarity Scores: {similarity_scores}')

            self.logger.info('=====================')

        # Print evaluation metrics
        self.logger.info('Calculating evaluation metrics...')

        # Convert boolean predictions and ground truths to integer labels
        y_true = df['Ground Is Relevant'].astype(int)
        y_pred = df['Is Relevant'].astype(int)
        y_pred_similarity = df['Is Relevant by Similarity'].astype(int)

        # Calculate metrics for pattern matching method
        self.logger.info('Evaluation Metrics for Pattern Matching Method:')
        self.logger.info(classification_report(y_true, y_pred))
        self.logger.info('Confusion Matrix:')
        self.logger.info(confusion_matrix(y_true, y_pred))

        # Calculate metrics for similarity method
        self.logger.info('Evaluation Metrics for Similarity Method:')
        self.logger.info(classification_report(y_true, y_pred_similarity))
        self.logger.info('Confusion Matrix:')
        self.logger.info(confusion_matrix(y_true, y_pred_similarity))

        # Plot confusion matrices
        self.plot_confusion_matrix(y_true, y_pred, 'Pattern Matching')
        self.plot_confusion_matrix(y_true, y_pred_similarity, 'Similarity')

        # # Find optimal threshold for similarity method
        threshold, f1 = self.find_optimal_threshold(y_true, df['DL Similarity Score'])

        self.logger.info(f'Optimal threshold for similarity method: {threshold:.2f}')
        self.logger.info(f'F1 score at optimal threshold: {f1:.2f}')

        # Save the results
        self.data_loader.save_xlsx_data(df, self.config['EVALUATION_DATASET_PATH'])

        # Identify misclassified papers
        misclassified_pattern    = df[df['Ground Is Relevant'] != df['Is Relevant']]
        misclassified_similarity = df[df['Ground Is Relevant'] != df['Is Relevant by Similarity']]

        # Save misclassified papers to separate Excel files
        misclassified_pattern_path    = '../../data/misclassified_pattern.xlsx'
        misclassified_similarity_path = '../../data/misclassified_similarity.xlsx'

        self.data_loader.save_xlsx_data(misclassified_pattern, misclassified_pattern_path)
        self.data_loader.save_xlsx_data(misclassified_similarity, misclassified_similarity_path)

        self.logger.info(f'Misclassified papers by pattern matching saved to {misclassified_pattern_path}')
        self.logger.info(f'Misclassified papers by similarity saved to {misclassified_similarity_path}')

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f"Test completed in {duration:.2f} seconds.")

        self.logger.info('Test complete.')


    async def test_three(self):
        """
        Test the semantic filtering process.
        """
        classifier = LLMSemanticFilter()

        df = self.data_loader.load_xlsx_data(self.config['TEST_DATASET_PATH'])

        # Initialize new columns for predictions
        df['Is Relevant Pattern']    = False
        df['Is Relevant Similarity'] = False
        df['Is Relevant LLM']        = False

        df['Reasoning Pattern']    = ''
        df['Reasoning Similarity'] = ''
        df['Reasoning LLM']        = ''

        df['Method Type Pattern']    = ''
        df['Method Type LLM']        = ''

        df['Method Name Pattern']    = ''
        df['Method Name LLM']        = ''

        self.logger.info('Starting semantic filtering...')

        # average time for each method
        total_pattern_time = 0
        total_similarity_time = 0
        total_llm_time = 0
        total_classify_time = 0

        for index, row in tqdm(df.iterrows(), total=len(df), desc='Processing Papers'):
            combined_text = f"{row['Title']} {row['Abstract']}"

            # Pattern Matching Method for Semantic Filtering
            start_time = time.time()
            is_relevant_pattern, scores_pattern, reasoning_pattern = self.semantic_filter.is_semantic_relevant_by_pattern_matching(combined_text)
            pattern_time = time.time() - start_time

            # Semantic Similarity Method for Semantic Filtering
            start_time = time.time()
            is_relevant_similarity, similarity_scores, reasoning_similarity = self.semantic_filter.is_semantic_relevant_by_similarity(combined_text)
            similarity_time = time.time() - start_time

            # LLM Method for Semantic Filtering
            start_time = time.time()
            result: ClassificationResult = await classifier.classify(combined_text)
            llm_time = time.time() - start_time

            # Classify method type and extract method name
            start_time = time.time()
            method_type_pattern = self.semantic_filter.classify_method(combined_text) if is_relevant_pattern else ''
            method_name_pattern = self.semantic_filter.extract_method_name(combined_text) if is_relevant_pattern else ''
            classify_time = time.time() - start_time

            # Update dataframe with results from LLM method
            df.at[index, 'Is Relevant Pattern']    = is_relevant_pattern
            df.at[index, 'Is Relevant Similarity'] = is_relevant_similarity
            df.at[index, 'Is Relevant LLM']        = result.relevant

            df.at[index, 'Reasoning Pattern']    = reasoning_pattern
            df.at[index, 'Reasoning Similarity'] = reasoning_similarity
            df.at[index, 'Reasoning LLM']        = result.reasoning

            df.at[index, 'Method Type Pattern'] = method_type_pattern
            df.at[index, 'Method Type LLM']     = result.method_type

            df.at[index, 'Method Name Pattern'] = method_name_pattern
            df.at[index, 'Method Name LLM']     = result.method_name

            self.logger.info("=====================")
            self.logger.info(f'Paper {index + 1}/{len(df)}')
            self.logger.info(f'PMID:        {row["PMID"]}')
            self.logger.info(f'Title:       {row["Title"]}')
            self.logger.info(f'Abstract:    {row["Abstract"]}')
            self.logger.info(f'Year:        {row["Publication Year"]}')
            self.logger.info(f'Journal:     {row["Journal/Book"]}')

            # Ground truth relevance
            self.logger.info(f'Ground truth is relevant: {row["Ground Is Relevant"]}')
            self.logger.info(f'Ground truth method type: {row["Ground Method Type"]}')
            self.logger.info(f'Ground truth method name: {row["Ground Method Name"]}')

            # Predictions
            self.logger.info(f'Is relevant by pattern matching: {is_relevant_pattern}')
            self.logger.info(f'Is relevant by similarity:       {is_relevant_similarity}')
            self.logger.info(f'Is relevant by LLM:              {result.relevant}')

            self.logger.info(f'Reasoning by LLM:                {result.reasoning}')

            self.logger.info(f'Method type: {result.method_type}')
            self.logger.info(f'Method name: {result.method_name}')

            self.logger.info('=====================')

            total_pattern_time    += pattern_time
            total_similarity_time += similarity_time
            total_llm_time        += llm_time
            total_classify_time   += classify_time

        self.logger.info('Calculating evaluation metrics...')

        avr_pattern_time    = total_pattern_time / len(df)
        avr_similarity_time = total_similarity_time / len(df)
        avr_llm_time        = total_llm_time / len(df)
        avr_classify_time   = total_classify_time / len(df)

        self.logger.info('-- Average Time --')
        # Log average time for each method
        self.logger.info(f'Average time for pattern matching: {avr_pattern_time:.2f} seconds')
        self.logger.info(f'Average time for similarity: {avr_similarity_time:.2f} seconds')
        self.logger.info(f'Average time for LLM: {avr_llm_time:.2f} seconds')
        self.logger.info(f'Average time for classify method: {avr_classify_time:.2f} seconds')
        # Log expected time for each method for 10.000 papers (based on average time)
        self.logger.info(f'Expected time for pattern matching for 10.000 papers: {avr_pattern_time * 10000:.2f} seconds')
        self.logger.info(f'Expected time for similarity for 10.000 papers: {avr_similarity_time * 10000:.2f} seconds')
        self.logger.info(f'Expected time for LLM method for 10.000 papers: {avr_llm_time * 10000:.2f} seconds')
        self.logger.info(f'Expected time for classify method for 10.000 papers: {avr_classify_time * 10000:.2f} seconds')
        self.logger.info('------------------')

        # Convert boolean predictions and ground truths to integer labels
        y_true            = df['Ground Is Relevant'].astype(int)
        y_pred_pattern    = df['Is Relevant Pattern'].astype(int)
        y_pred_similarity = df['Is Relevant Similarity'].astype(int)
        y_pred_llm        = df['Is Relevant LLM'].astype(int)


        # Calculate metrics for pattern matching method
        self.logger.info('Evaluation Metrics for Pattern Matching Method:')
        self.logger.info(classification_report(y_true, y_pred_pattern))
        self.logger.info('Confusion Matrix for Pattern Matching:')
        self.logger.info(confusion_matrix(y_true, y_pred_pattern))
        self.plot_confusion_matrix(y_true, y_pred_pattern, 'Pattern Matching')

        # Calculate metrics for similarity method
        self.logger.info('Evaluation Metrics for Similarity Method:')
        self.logger.info(classification_report(y_true, y_pred_similarity))
        self.logger.info('Confusion Matrix for Similarity:')
        self.logger.info(confusion_matrix(y_true, y_pred_similarity))
        self.plot_confusion_matrix(y_true, y_pred_similarity, 'Similarity')

        # Calculate metrics for LLM method
        self.logger.info('Evaluation Metrics for LLM Method:')
        self.logger.info(classification_report(y_true, y_pred_llm))
        self.logger.info('Confusion Matrix for LLM:')
        self.logger.info(confusion_matrix(y_true, y_pred_llm))
        self.plot_confusion_matrix(y_true, y_pred_llm, 'LLM')

        # Save the results to Excel file for further analysis
        self.data_loader.save_xlsx_data(df, self.config['EVALUATION_DATASET_PATH'])

        # Identify misclassified papers
        misclassified_pattern = df[df['Ground Is Relevant'] != df['Is Relevant Pattern']]
        misclassified_similarity = df[df['Ground Is Relevant'] != df['Is Relevant Similarity']]
        misclassified_llm = df[df['Ground Is Relevant'] != df['Is Relevant LLM']]

        # Save misclassified papers to separate Excel files
        misclassified_pattern_path    = '../../data/misclassified_pattern.xlsx'
        misclassified_similarity_path = '../../data/misclassified_similarity.xlsx'
        misclassified_llm_path        = '../../data/misclassified_llm.xlsx'

        self.data_loader.save_xlsx_data(misclassified_pattern, misclassified_pattern_path)
        self.data_loader.save_xlsx_data(misclassified_similarity, misclassified_similarity_path)
        self.data_loader.save_xlsx_data(misclassified_llm, misclassified_llm_path)

        self.logger.info(f'Misclassified papers by llm matching saved to {misclassified_pattern_path}')
        self.logger.info(f'Misclassified papers by similarity saved to {misclassified_similarity_path}')
        self.logger.info(f'Misclassified papers by llm saved to {misclassified_llm_path}')

        self.logger.info('Test complete.')


    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, method_name):
        """
        Plot the confusion matrix for the given method.
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6,4))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

        plt.title(f'Confusion Matrix for {method_name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()


    @staticmethod
    def find_optimal_threshold(y_true, similarity_scores):
        thresholds = [i * 0.05 for i in range(1, 20)]
        best_threshold = 0.0
        best_f1 = 0.0

        for threshold in thresholds:
            y_pred = (similarity_scores >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold, best_f1


if __name__ == '__main__':
    tester = SemanticFilterTest()

    # tester.test_two()

    asyncio.run(tester.test_three())

