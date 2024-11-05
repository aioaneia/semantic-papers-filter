
import yaml

from src.data.data_loader import DataLoader
from src.filters.semantic_filter import SemanticFilter

# Load configuration
with open('../../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Load configuration
data_loader     = DataLoader()
semantic_filter = SemanticFilter()


def test_semantic_filtering():
    """
    Test the semantic filtering process.
    """
    df = data_loader.load_xlsx_data(config['TEST_DATASET_PATH'])

    # Initialize new columns for NLP predictions
    df['Is Relevant'] = False
    df['Scores'] = ''
    df['Method Type'] = ''
    df['Method Name'] = ''

    print("Starting semantic filtering...")

    for index, row in df.iterrows():
        combined_text = f"{row['Title']} {row['Abstract']}"

        is_relevant, scores = semantic_filter.is_semantic_relevant(combined_text)

        is_relevant_by_similarity, similarity_scores = semantic_filter.is_semantic_relevant_by_similarity(combined_text)

        # Update dataframe with results
        df.at[index, 'Is Relevant'] = is_relevant
        df.at[index, 'Scores'] = str(scores)
        df.at[index, 'Similarity Scores'] = str(similarity_scores)
        df.at[index, 'Is Relevant by Similarity'] = is_relevant_by_similarity

        method_type = ''
        method_name = ''

        if is_relevant:
            # Classify method type
            method_type = semantic_filter.classify_method(combined_text)
            method_name = semantic_filter.extract_method_name(combined_text)

            # Update dataframe
            df.at[index, 'Method Type'] = method_type
            df.at[index, 'Method Name'] = method_name

        print("=====================")
        print(f"Paper {index + 1}/{len(df)}")
        print(f"Title:       {row['Title']}")
        print(f"Abstract:    {row['Abstract']}")
        print(f'Year:        {row["Publication Year"]}')
        print(f'Journal:     {row["Journal/Book"]}')
        print(f"Is relevant: {is_relevant}")
        print(f"By similarity: {is_relevant_by_similarity}")
        print(f"Similarity Scores: {similarity_scores}")
        print(f"Scores:      {scores}")
        print(f"Method type: {method_type}")
        print(f"Method name: {method_name}")
        print("=====================")

    # Save the results
    data_loader.save_xlsx_data(df, config['EVALUATION_DATASET_PATH'])

    print("Test complete.")


if __name__ == "__main__":
    # Run the test
    test_semantic_filtering()
