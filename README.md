

# Semantic Virology/Epidemiology Papers Filter

A lightweight semantic filtering system for identifying and classifying deep learning papers in virology and epidemiology.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Solution Components](#solution-components)
  - [NLP Technique for Filtering Papers](#nlp-technique-for-filtering-papers)
  - [Why Our Approach Is More Effective Than Keyword-Based Filtering](#why-our-approach-is-more-effective-than-keyword-based-filtering)
- [Resulting Dataset Statistics](#resulting-dataset-statistics)
- [Results Visualization](#results-visualization)
- [Project Structure](#project-structure)
- [Comparative Analysis of NLP Techniques](#comparative-analysis-of-nlp-techniques)
- [Conclusion](#conclusion)
- [Contributors](#contributors)
- [References](#references)
- [Appendix](#appendix)

## Introduction
Identifying relevant papers in a specific domain can be challenging due to the vast amount of literature available. 
Traditional keyword-based searches often yield noisy results, 
requiring manual filtering to identify papers that focus on computational methods. 
To address this issue, we developed a semantic filtering system that leverages natural language processing (NLP) techniques 
to automatically identify and classify deep learning papers in the fields of virology and epidemiology.

Our system:
- Implements semantic NLP techniques to filter out papers not utilizing deep learning approaches in virology/epidemiology.
- Classifies relevant papers according to the method used: "text mining", "computer vision", "both", "other".
- Extracts and reports the name of the method used for each relevant paper.
- Provides detailed statistics and visualizations of the results.

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
python -m spacy download en_core_web_lg
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
- Generate results
  - Save filtered papers in `results/filtered_papers.csv`
  - Save filtered out papers in `results/filtered_out_papers.csv`
  - Create statistics in `results/statistics.json`
  - Save plots in `results/plots/`

## Project Structure
```
semantic-virology-papers-filter/
├── src/
│   ├── data/
│   ├──     data_loader.py            # Data loading and basic statistics
│   ├──     preprocessor.py           # Text preprocessing and cleaning
│   ├── filters/
│   ├──     semantic_filter.py        # Core NLP filtering and classification logic
│   ├──     llm_semantic_filter.py    # LLM-based semantic filter
│   ├── pipelines/
│   ├──     pipeline.py               # Pattern matching pipeline 
│   ├── tests/
│   ├──     test_semantic_filters.py  # Unit tests for semantic filter
│   ├── utils/
            constants.py              # Definitions of patterns, terms, and weights
│   ├──     file_utils.py             # File I/O utility functions
│   ├──     visualisation.py          # Plotting functions 
│   └── main.py                       # Main execution script
├── data/                             # Data directory (input CSV file) and test file for evaluation
├── results/                          # Output directory (filtered results and statistics)
│   ├── files/
│   ├── stats/
│   └── plots/
├── notebooks/                        # Jupyter notebooks for data exploration and analysis
└── config.yaml                       # Configuration file for setting parameters
└── README.md                         # Project overview and instructions
└── requirements.txt                  # Project dependencies
```

## Solution Components

### NLP Technique

Our solution employs a rule-based natural language processing (NLP) approach using spaCy's capabilities:
- **spaCy's Medium English Model (en_core_web_lg)** Provides robust tokenization, part-of-speech tagging, named entity recognition (NER), and word vectors for semantic similarity.
- **Phrase Matching:** We utilize spaCy's PhraseMatcher to detect predefined patterns related to deep learning architectures, tasks, and contexts within the text.
- **Custom Patterns and Weights:**
  - **Defined Patterns:** In the constants.py file, we have curated lists of terms for architectures (e.g., "neural network", "deep learning"), tasks (e.g., "classification", "detection"), and contexts (e.g., "training", "model"). 
  - **Weighted Scoring System:** Each category is assigned a weight to compute a relevance score, allowing us to prioritize certain aspects over others.
- **Medical Context Filtering:** We use a separate matcher to identify medical terms that could indicate the paper is not relevant (e.g., "computer vision syndrome") and exclude such papers.
- **Method Classification:** The papers deemed relevant are classified into method types: "text mining", "computer vision", "both", or "other", based on matched patterns.
- **Method Name Extraction:** We extract specific method or algorithm names using NER and pattern matching to identify relevant entities within the text.

### Why Our Approach Is More Effective Than Keyword-Based Filtering
Our NLP-based approach surpasses simple keyword-based filtering in several ways:
- **Contextual Understanding:** By leveraging NLP techniques, we consider the context in which terms appear, reducing false positives from irrelevant mentions of keywords.
- **Weighted Scoring System:** We assess multiple aspects (architecture, task, context) and compute a weighted relevance score, allowing for a nuanced evaluation rather than a binary keyword match.
- **Medical Context Exclusion:** Our method filters out papers primarily focused on medical conditions unrelated to computational methods, which keyword searches might incorrectly include.
- **Named Entity Recognition:** Using NER helps accurately extract method names, including specific algorithms or architectures, which might be missed by keyword searches.
- **Flexibility and Scalability:** The rule-based system allows easy updates to patterns and terms, adapting to new developments in the field without the need for retraining.

## Resulting Dataset Statistics
After processing the dataset of 11,450 papers, the following results were obtained:

### Overall Statistics

| **Metric**                     | **Value**       |
|---------------------------------|-----------------|
| Total Papers Processed          | 11,450          |
| Relevant Papers Identified      | 1,932           |
| Relevance Percentage            | 16.87%          |

### Method Type Distribution

| **Method Type**     | **Count** | **Percentage** |
|----------------------|-----------|----------------|
| Text Mining          | 211       | 10.92%         |
| Computer Vision      | 561       | 29.04%         |
| Both                 | 180       | 9.32%          |
| Other                | 980       | 50.72%         |

### Top Method Names Identified

| **Method Name**                 | **Occurrences** |
|---------------------------------|-----------------|
| Neural Network                  | 955             |
| Deep Learning                   | 796             |
| Machine Learning                | 652             |
| Artificial Intelligence         | 240             |
| Artificial Neural Network       | 236             |
| GAN                             | 194             |
| Convolutional Neural Network (CNN) | 171          |
| LSTM                            | 149             |

### Reasons for Irrelevant Papers Exclusion

The following table summarizes the reasons for excluding papers from relevance:

| **Exclusion Reason**                | **Count** |
|-------------------------------------|-----------|
| Context domain not found            | 6,501     |
| Deep learning context not found     | 1,431     |
| Only medical context found          | 970       |
| Negative keywords found             | 616       |

### Time Taken for Processing

The total time taken to process 11,450 papers was **732.46 seconds**.

## Results Visualization

### Method Type Percentages
![Method Type Percentages](results/plots/method_type_percentages.html)

### Method Type Distribution Over Time
![Method Types Over Time](results/plots/method_types_over_time.html)

### Top method names over Time
![Method Names Over Time](results/plots/method_names_over_time.html)

### Distribution of irrelevant papers
![Irrelevant Papers Distribution](results/plots/irrelevant_papers_by_reasoning.html)

### Word Cloud of Method Names
![Method Names Word Cloud](results/plots/word_cloud_method_name.html)

## Distribution of Jouranls over Time
![Journal Distribution Over Time](results/plots/publication_distribution_per_journal.html)

## Publications per Journal Over Time
![Publications per Journal Over Time](results/plots/publications_per_journal_over_time.html)

## Comparison of top journals in relevant and irrelevant papers
![Top Journals Comparison](results/plots/journal_comparison.html)

## Limitations and Future Work
- Limitations:
  - Heuristic Bias: Current rules may exclude some valid deep learning papers if they don't match predefined patterns.
  - Time Complexity: LLM-based filtering, while effective, is slower and less suitable for large-scale datasets.

- Future Work:
  - Expand domain-specific patterns to improve recall.
  - Integrate unsupervised learning for dynamic pattern discovery.
  - Explore lightweight transformer models for faster processing.

## Conclusion
The developed system effectively identifies and classifies deep learning papers in virology and epidemiology 
using a combination of rule-based NLP and semantic techniques. It provides a significant improvement over traditional keyword-based filtering, 
ensuring higher precision and recall while automating a previously manual process.

## Contributors
- [Andrei Aioanei]

## References
- [spaCy Documentation](https://spacy.io/usage)

## Appendix
This appendix provides a comprehensive evaluation of alternative methods, such as LLM-based filtering and semantic similarity, 
in comparison to the baseline pattern matching approach. The results are derived from a test dataset of 50 papers, 
specifically curated for assessing the performance of these methodologies in the domain of filtering relevant academic articles.

### Test Dataset
The dataset (test_collection_with_abstracts.csv) contains 50 paper entries with detailed abstracts. 
Each method—Pattern Matching, Semantic Similarity, and LLM-based filtering (using Llama 3.2)—was applied to this dataset 
to evaluate its performance based on standard metrics: Precision, Recall, F1-Score, and Average Time per Paper. 
These metrics were computed to assess the balance between efficiency and accuracy in identifying relevant articles.

### Evaluation Metrics
| Method              | Precision | Recall | F1-Score | Avg. Time per Paper | Expected Time for 10,000 Papers  |
|---------------------|-----------|--------|----------|---------------------|----------------------------------|
| Pattern Matching    | 1.00      | 1.00   | 1.00     | 0.09 seconds        | 929.46 seconds                   |
| Semantic Similarity | 0.64      | 0.76   | 0.57     | 0.15 seconds        | 1,475.88 seconds                 |
| LLM (Llama 3.2)     | 0.77      | 0.92   | 0.80     | 1.40 seconds        | 14,024.75 seconds                |

### Observations on Irrelevant Classifications
Examples of misclassified or excluded papers highlight the importance of domain-specific understanding:
- PMID 39285189: "Computer Vision Syndrome" was correctly classified as irrelevant due to its focus on medical ophthalmology, a field outside the intended scope.
- PMID 39155966: This urban planning study was excluded, as its content diverges significantly from the virology and epidemiology themes central to the task.

### Evaluation of Heuristics in LLM Filtering
Through this experiment, it became evident that the LLM-based method often relied on learned heuristics, 
inferred from common patterns within the dataset. For example:
- Paper PMID 38853172: Focused on "image segmentation of COVID-19 X-ray," it was identified as relevant despite using the Whale Optimization Algorithm (WOA), a heuristic optimization technique rather than a deep learning method. This suggests that the model's heuristics generalized relevance to methodologies tangentially connected to the domain.

The insights align with the heuristic hypothesis (Nikankin et al. 2024), 
emphasizing the importance of understanding the decision processes within LLMs. 
While they can mimic reasoning via heuristic composition, detailed interpretability remains a challenge. 
Future work could explore refining semantic similarity models and LLM prompts to enhance domain specificity 
while optimizing computational costs.