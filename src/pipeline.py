
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
                    'year':        row['Publication Year'],
                    'method_type': method_type,
                    'method_name': method_name,
                }

                results.append(result_dict)

            print("=====================")
            print(f"Title:       {row['Title']}")
            print(f"Abstract:    {row['Abstract']}")
            print(f'Year:        {row["Publication Year"]}')
            print(f"Is relevant: {is_relevant}")
            print(f"Scores:      {scores}")
            print(f"Method type: {method_type}")
            print(f"Method name: {method_name}")
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
                    'year':        pd.to_numeric(row['Publication Year'], errors='coerce').fillna(0).astype(int),
                    'relevant':    result.relevant,
                    'method_type': result.method_type,
                    'method_name': result.method_name,
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
        method_percentages = (method_distribution / total_relevant_papers * 100).round(2)
        method_counts = method_distribution

        # Convert keys to str and values to native types for JSON serialization
        method_counts = {str(k): int(v) for k, v in method_counts.items()}
        method_percentages = {str(k): float(v) for k, v in method_percentages.items()}

        # Top method names overall
        top_methods_overall = self.results['method_name'].value_counts().head(10)
        top_methods_overall = {str(k): int(v) for k, v in top_methods_overall.items()}

        # Top method names by year
        top_methods_by_year = {}
        years = self.results['year'].unique()
        for year in sorted(years):
            year_df = self.results[self.results['year'] == year]
            top_methods = year_df['method_name'].value_counts().head(5)
            # Convert method names and counts to native types
            top_methods = {str(k): int(v) for k, v in top_methods.items()}
            if top_methods:
                top_methods_by_year[int(year)] = top_methods  # Convert year to int

        stats = {
            'total_papers_processed': int(total_papers_processed),
            'total_relevant_papers': int(total_relevant_papers),
            'relevance_percentage': float(round((total_relevant_papers / total_papers_processed) * 100, 2)),
            'method_counts': method_counts,
            'method_percentages': method_percentages,
            'top_method_names_overall': top_methods_overall,
            'top_method_names_by_year': top_methods_by_year,
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
        plt.show()
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

        # Plot number of relevant papers per year
        papers_per_year = self.results.groupby('year').size()

        plt.figure(figsize=(10, 6))
        papers_per_year.plot(kind='bar', color='skyblue')
        plt.xlabel('Publication Year')
        plt.ylabel('Number of Relevant Papers')
        plt.title('Number of Relevant Papers per Year')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/papers_per_year.png')
        plt.close()

        # Plot method type distribution over time
        method_types_over_time = self.results.groupby(['year', 'method_type']).size().unstack(fill_value=0)
        method_types_over_time.plot(kind='bar', stacked=True, figsize=(12, 7))
        plt.xlabel('Publication Year')
        plt.ylabel('Number of Papers')
        plt.title('Method Type Distribution Over Time')
        plt.legend(title='Method Type')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/method_types_over_time.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        # Trend lines for top methods
        for method_name in statistics['top_method_names_overall'].keys():
            method_data = self.results[self.results['method_name'] == method_name]
            papers_per_year = method_data.groupby('year').size()
            plt.plot(papers_per_year.index, papers_per_year.values, marker='o', label=method_name)

        plt.xlabel('Publication Year')
        plt.ylabel('Number of Papers')
        plt.title('Trend of Top Method Names Over Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/method_trends_over_time.png')
        plt.close()

        # Print statistics
        print(f"Total papers processed: {statistics['total_papers_processed']}")
        print(f"Total relevant papers: {statistics['total_relevant_papers']}")
        print(f"Relevance percentage: {statistics['relevance_percentage']}%")
        print("\nMethod distribution:")
        for method, count in method_counts.items():
            percentage = method_percentages[method]
            print(f"- {method}: {count} papers ({percentage}%)")

        # Print top method names by year
        print("\nTop Method Names by Year:")
        for year, methods in statistics['top_method_names_by_year'].items():
            print(f"\nYear: {int(year)}")
            for method_name, count in methods.items():
                print(f"- {method_name}: {count} occurrences")
