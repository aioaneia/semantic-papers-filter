
import os
import yaml
import asyncio

from data_loader         import DataLoader
from preprocessor        import Preprocessor
from pipeline            import Pipeline
from semantic_filter     import SemanticFilter
from semantic_filter_llm import PaperClassifier


# Load configuration
with open('../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)


# Main function
def main(process_by_nlp=True, process_by_llm=False):
    # Create output directory
    os.makedirs("../results", exist_ok=True)

    # Initialize components
    data_loader     = DataLoader(config['DATASET_PATH'])
    preprocessor    = Preprocessor()
    nlp_classifier  = SemanticFilter()
    llm_classifier  = PaperClassifier()

    # Create and run pipeline
    pipeline = Pipeline(
        data_loader,
        preprocessor,
        nlp_classifier,
        llm_classifier
    )

    print("Processing papers...")

    if process_by_nlp:
        # Process papers using NLP pipeline
        pipeline.process_by_nlp(limit=20)

        print("Saving results...")

        pipeline.save_results("../results", "nlp")

    if process_by_llm:
        # Process papers using LLM pipeline
        asyncio.run(pipeline.process_by_llm(limit=20))

    print("Processing complete.")

    print("Plot statistics...")

    pipeline.plot_statistics("../results")


if __name__ == "__main__":
    main(process_by_nlp=True, process_by_llm=False)
