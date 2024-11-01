
import os
import pandas as pd
import json

import matplotlib.pyplot as plt

from typing import Optional


class Pipeline:
    def __init__(self, data_loader, preprocessor, semantic_filter, llm_classifier):
        """
        Initialize the pipeline with the specified components.
        """
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.semantic_filter = semantic_filter
        self.llm_classifier = llm_classifier
        self.results: Optional[pd.DataFrame] = None


    def process_by_nlp(self, limit: int = 1000):
        """
        Process the papers and return filtered results.
        """
        df = self.data_loader.load_data()

        # Limit the number of records for testing
        df = df.head(limit)

        stats = self.data_loader.get_basic_stats()

        print("Dataset Statistics:")
        print(f"Total records: {stats['total_records']}")
        print(f"Records with abstracts: {stats['records_with_abstract']}")
        print(f"Missing values: {stats['missing_values']}")

        df = self.preprocessor.prepare_dataset(df)

        print("Starting semantic filtering...")

        # Apply semantic filtering
        results = []
        for _, row in df.iterrows():
            combined_text = f"{row['clean_title']} {row['clean_abstract']}"

            is_relevant, scores = self.semantic_filter.is_semantic_relevant(combined_text)
            method_type = None
            method_name = None

            if is_relevant:
                # Classify method type
                method_type = self.semantic_filter.classify_method(combined_text)

                # Extract method name
                method_name = self.semantic_filter.extract_method_name(combined_text)

                result_dict = {
                    'PMID':        row['PMID'],
                    'title':       row['Title'],
                    'abstract':    row['Abstract'],
                    'method_type': method_type,
                    'method_name': method_name,
                }

                results.append(result_dict)

            print("=====================")
            print(f"Title:       {row['Title']}")
            print(f"Abstract:    {row['Abstract']}")
            print(f"Is relevant: {is_relevant}")
            print(f"Scores:      {scores}")
            print(f"Method type: {method_type}")
            print(f"Method name: {method_name}")
            print("")
            print("=====================")

        self.results = pd.DataFrame(results)

        print("Semantic filtering complete.")


    async def process_by_llm(self, limit: int = 1000):
        """
        Process the papers and return filtered results.
        """
        df = self.data_loader.load_data()

        # Limit the number of records for testing
        df = df.head(10)

        stats = self.data_loader.get_basic_stats()

        print("Dataset Statistics:")
        print(f"Total records: {stats['total_records']}")
        print(f"Records with abstracts: {stats['records_with_abstract']}")
        print(f"Missing values: {stats['missing_values']}")

        df = self.preprocessor.prepare_dataset(df)

        print("Starting semantic filtering...")

        # Apply semantic filtering
        results = []
        for _, row in df.iterrows():
            combined_text = f"{row['clean_title']} {row['clean_abstract']}"

            # Run classification
            result = await self.llm_classifier.classify(combined_text)

            if result.relevant:
                result_dict = {
                    'PMID':        row['PMID'],
                    'title':       row['Title'],
                    'abstract':    row['Abstract'],
                    'method_type': result.method_type,
                    'method_name': result.method_name,
                    'relevant':    result.relevant,
                    'reasoning':   result.reasoning
                }

                results.append(result_dict)

            print("=====================")
            print(f"Title:       {row['Title']}")
            print(f"Abstract:    {row['Abstract']}")
            print(f"Is relevant: {result.relevant}")
            print(f"Method Type: {result.method_type}")
            print(f"Method Name: {result.method_name}")
            print(f"Reasoning:   {result.reasoning}")
            print("")
            print("=====================")

        self.results = pd.DataFrame(results)

        print("Semantic filtering complete.")


    def save_results(self, output_dir, id):
        """Save results and statistics to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save main results
        if self.results is not None:
            # Save filtered papers
            self.results.to_csv(f"{output_dir}/{id}_filtered_papers.csv", index=False)

            stats = self.get_statistics()

            # Save statistics
            with open(f"{output_dir}/{id}_statistics.json", "w") as f:
                json.dump(stats, f, indent=2)

            print(f"\nResults saved to {output_dir}/")


    def get_statistics(self):
        """Get statistics about the processed results."""
        if self.results is None:
            raise ValueError("Run process() first")

        total_papers_processed = self.data_loader.get_basic_stats()['total_records']
        total_relevant_papers = len(self.results)

        # Method type distribution
        method_distribution = self.results['method_type'].value_counts()
        method_percentages = (method_distribution / total_relevant_papers * 100).round(2).to_dict()
        method_counts = method_distribution.to_dict()

        # Top method names
        top_methods = self.results['method_name'].value_counts().head(10).to_dict()

        stats = {
            'total_papers_processed': total_papers_processed,
            'total_relevant_papers': total_relevant_papers,
            'relevance_percentage': round((total_relevant_papers / total_papers_processed) * 100, 2),
            'method_counts': method_counts,
            'method_percentages': method_percentages,
            'top_method_names': top_methods,
        }

        return stats

    def plot_statistics(self, output_dir):
        statistics = self.get_statistics()

        # Bar chart for method counts
        method_counts = statistics['method_counts']
        methods = list(method_counts.keys())
        counts = list(method_counts.values())

        plt.figure(figsize=(8, 6))
        plt.bar(methods, counts, color='skyblue')
        plt.xlabel('Method Type')
        plt.ylabel('Number of Papers')
        plt.title('Distribution of Method Types')
        plt.savefig(f'{output_dir}/method_type_distribution.png')
        plt.close()

        # Pie chart for method percentages
        method_percentages = statistics['method_percentages']
        labels = list(method_percentages.keys())
        sizes = list(method_percentages.values())

        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Method Type Percentages')
        plt.savefig(f'{output_dir}/method_type_percentages.png')
        plt.close()

        # Print statistics
        print(f"Total papers processed: {statistics['total_papers_processed']}")
        print(f"Total relevant papers: {statistics['total_relevant_papers']}")
        print(f"Relevance percentage: {statistics['relevance_percentage']}%")
        print("\nMethod distribution:")
        for method, count in method_counts.items():
            percentage = method_percentages[method]
            print(f"- {method}: {count} papers ({percentage}%)")

        print("\nTop Method Names Extracted:")
        for method_name, count in statistics['top_method_names'].items():
            print(f"- {method_name}: {count} occurrences")