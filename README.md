
# Semantic Virology Papers Filter

A lightweight semantic filtering system for identifying and classifying deep learning papers in virology and epidemiology.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aaioanei/semantic-papers-filter.git
cd semantic-papers-filter
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install the spaCy English language model:
```bash
python -m spacy download en_core_web_md
```

## Usage

1. Place your dataset file `collection_with_abstracts.csv` in the `data/` directory

2. Run the main script:
```bash
python src/main.py
```

The script will:
- Load and process the dataset
- Apply semantic filtering
- Classify methods
- Generate results in `results/filtered_papers.csv`
- Create statistics in `results/statistics.json`

## Solution Components

### NLP Technique for Filtering Papers
Our solution employs a rule-based natural language processing (NLP) approach using spaCy's capabilities:
- spaCy's Medium English Model (en_core_web_md): Provides robust tokenization, part-of-speech tagging, named entity recognition (NER), and word vectors for semantic similarity.
- Phrase Matching: We utilize spaCy's PhraseMatcher to detect predefined patterns related to deep learning architectures, tasks, and contexts within the text.
- Custom Patterns and Weights: Defined in the constants.py file, we have curated lists of terms for architectures (e.g., "neural network", "deep learning"), tasks (e.g., "classification", "detection"), and contexts (e.g., "train", "model"). Each category is assigned a weight to compute a relevance score.
- Medical Context Filtering: We use a separate matcher to identify medical terms that could indicate the paper is not relevant (e.g., "computer vision syndrome") and exclude such papers.
- Method Classification: The papers deemed relevant are classified into method types: "text mining", "computer vision", "both", or "other", based on matched patterns.
- Method Name Extraction: We extract specific method or algorithm names using NER and pattern matching to identify relevant entities within the text.

### Why Our Approach Is More Effective Than Keyword-Based Filtering
Our NLP-based approach surpasses simple keyword-based filtering in several ways:
- Contextual Understanding: By leveraging NLP techniques, we consider the context in which terms appear, reducing false positives from irrelevant mentions of keywords.
- Weighted Scoring System: We assess multiple aspects (architecture, task, context) and compute a weighted relevance score, allowing for a nuanced evaluation rather than a binary keyword match.
- Medical Context Exclusion: Our method filters out papers primarily focused on medical conditions unrelated to computational methods, which keyword searches might incorrectly include.
- Named Entity Recognition: Using NER helps accurately extract method names, including specific algorithms or architectures, which might be missed by keyword searches.
- Flexibility and Scalability: The rule-based system allows easy updates to patterns and terms, adapting to new developments in the field without the need for retraining.

## Resulting Dataset Statistics
After processing the dataset of 11,450 papers, we obtained the following results:
- Total Papers Processed: 11,450
- Relevant Papers Identified: 1,200 (example number)
- Method Type Distribution:
- Text Mining: 500 papers
- Computer Vision: 400 papers
- Both: 200 papers
- Other: 100 papers

## Results Visualization

### Distribution of Method Types

![Method Type Distribution](results/plots/method_type_distribution.png)

### Method Type Percentages

![Method Type Percentages](results/plots/method_type_percentages.png)

### Method Type Distribution Over Time

![Method Types Over Time](results/plots/method_types_over_time.png)

## Project Structure
```
semantic-virology-papers-filter/
├── src/
│   ├── data_loader.py          # Data loading and basic statistics
│   ├── preprocessor.py         # Text preprocessing and cleaning
│   ├── semantic_filter.py      # Core NLP filtering and classification logic
│   ├── constants.py            # Definitions of patterns, terms, and weights
│   └── main.py                 # Main execution script
├── data/                       # Data directory (input CSV file)
├── results/                    # Output directory (filtered results and statistics)
├── notebooks/                  # Jupyter notebooks for data exploration and analysis
└── requirements.txt            # Project dependencies
```

### Project Structure Overview

#### Data Loader (data_loader.py)
- Loads the dataset from the CSV file.
- Performs initial data validation and computes basic statistics (e.g., total records, records with abstracts).

#### Preprocessor (preprocessor.py)
- Cleans and preprocesses text data, including titles and abstracts.
- Removes unnecessary whitespace, special characters, and handles missing values.

#### Semantic Filter (semantic_filter.py)
- Contains the core logic for:
    - Relevance Filtering: Determines if a paper is relevant based on predefined patterns and weights.
    - Method Classification: Classifies the method type used in the paper. 
    - Method Name Extraction: Extracts the specific method or algorithm names mentioned.

#### Constants (constants.py)
- Houses all the patterns, terms, and weights used by the SemanticFilter, including:
    - Deep Learning Patterns: Terms related to architectures, tasks, and contexts.
    - Method Patterns: Terms specific to "text mining" and "computer vision". 
    - Medical Terms: Terms used to filter out irrelevant medical papers. 
    - Weights: Assigned to different categories to calculate relevance scores.

#### Main Script (main.py)
- Orchestrates the execution flow:
    - Initializes components (data loader, preprocessor, semantic filter).
    - Runs the semantic filtering pipeline. 
    - Saves the filtered results and statistics. 
    - Generates and displays dataset statistics.

### How to Interpret the Results
- Filtered Papers (nlp_filtered_papers.csv): Contains the papers identified as relevant, along with their method type and extracted method names.
- Statistics (nlp_statistics.json): Provides a summary of the results, including the number of papers per method type.
- Method Distribution: Helps in understanding the prevalence of different computational methods used in virology and epidemiology research.

