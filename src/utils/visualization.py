
import os
import pandas as pd
import plotly.express as px


class StatsVisualizer:
    """
    Class for visualizing statistics.
    """

    @staticmethod
    def plot_method_type_distribution(method_counts, output_dir):
        """
        Plot the distribution of method types as a bar chart.
        """
        methods = list(method_counts.keys())
        counts = list(method_counts.values())

        fig = px.bar(
            x=methods,
            y=counts,
            labels={
                'x': 'Method Type',
                'y': 'Number of Papers'
            },
            title='Distribution of Method Types',
        )

        fig.update_layout(
            xaxis_title='Method Type',
            yaxis_title='Number of Papers',
            title={'x': 0.5},  # Center the title
        )

        # Save the plot as an HTML file
        output_path = os.path.join(output_dir, 'method_type_distribution.html')
        fig.write_html(output_path)

        fig.show()


    @staticmethod
    def plot_method_type_percentage(method_percentages, output_dir):
        """
        Plot the percentage of each method type as a pie chart.
        """
        labels = list(method_percentages.keys())
        sizes = list(method_percentages.values())

        fig = px.pie(
            names=labels,
            values=sizes,
            title='Method Type Percentages',
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )

        fig.update_layout(
            title={'x': 0.5},
        )

        # Save the plot as an HTML file
        output_path = os.path.join(output_dir, 'method_type_percentages.html')
        fig.write_html(output_path)

        fig.show()


    @staticmethod
    def plot_papers_per_year(results: pd.DataFrame, output_dir):
        """
        Plot the number of relevant papers per year.
        """
        papers_per_year = results.groupby('year').size().reset_index(name='counts')

        fig = px.bar(
            papers_per_year,
            x='year',
            y='counts',
            labels={
                'year': 'Publication Year',
                'counts': 'Number of Relevant Papers'
            },
            title='Number of Relevant Papers per Year',
        )

        fig.update_layout(
            xaxis_title='Publication Year',
            yaxis_title='Number of Relevant Papers',
            title={'x': 0.5},
        )

        fig.update_xaxes(
            dtick=1,
            tickmode='linear',
            tickformat='d'
        )

        output_path = os.path.join(output_dir, 'papers_per_year.html')
        fig.write_html(output_path)

        fig.show()


    @staticmethod
    def plot_method_type_distribution_over_time(results: pd.DataFrame, output_dir):
        """
        Plot the distribution of method types over time.
        """
        method_types_over_time = results.groupby(['year', 'Method Type']).size().reset_index(name='counts')

        fig = px.bar(
            method_types_over_time,
            x='year',
            y='counts',
            color='Method Type',
            labels={
                'year': 'Publication Year',
                'counts': 'Number of Papers',
                'Method Type': 'Method Type'
            },
            title='Method Type Distribution Over Time',
        )

        fig.update_layout(
            barmode='stack',
            xaxis_title='Publication Year',
            yaxis_title='Number of Papers',
            legend_title_text='Method Type',
            title={'x': 0.5},
        )

        fig.update_xaxes(
            dtick=1,
            tickmode='linear',
            tickformat='d'
        )

        output_path = os.path.join(output_dir, 'method_types_over_time.html')
        fig.write_html(output_path)

        fig.show()


    @staticmethod
    def plot_trend_of_top_method_names_over_time(top_method_names, results, output_dir):
        method_data_list = []

        for method_name in top_method_names.keys():
            method_data = results[results['Method Name'] == method_name]
            papers_per_year = method_data.groupby('year').size().reset_index(name='counts')
            papers_per_year['Method Name'] = method_name
            method_data_list.append(papers_per_year)

        if method_data_list:
            all_method_data = pd.concat(method_data_list)

            fig = px.line(
                all_method_data,
                x='year',
                y='counts',
                color='Method Name',
                markers=True,
                labels={
                    'year': 'Publication Year',
                    'counts': 'Number of Papers',
                    'Method Name': 'Method Name'
                },
                title='Trend of Top Method Names Over Time',
            )

            fig.update_layout(
                xaxis_title='Publication Year',
                yaxis_title='Number of Papers',
                legend_title_text='Method Name',
                title={'x': 0.5},
            )

            fig.update_xaxes(
                dtick=1,
                tickmode='linear',
                tickformat='d'
            )

            output_path = os.path.join(output_dir, 'method_trends_over_time.html')
            fig.write_html(output_path)
            fig.show()
        else:
            print("No method data available to plot trends.")


    @staticmethod
    def plot_top_journals_by_method_type(relevant_df: pd.DataFrame, output_dir, top_n=10):
        """
        Plot the top journals for each method type.
        """
        method_types = relevant_df['Method Type'].unique()

        for method in method_types:
            method_df = relevant_df[relevant_df['Method Type'] == method]
            top_journals = method_df['Journal/Book'].value_counts().head(top_n).reset_index()
            top_journals.columns = ['Journal/Book', 'Paper Count']

            fig = px.bar(
                top_journals,
                x='Journal/Book',
                y='Paper Count',
                labels={
                    'Journal/Book': 'Journal',
                    'Paper Count': 'Number of Papers'
                },
                title=f'Top {top_n} Journals for Method Type: {method.capitalize()}',
            )

            fig.update_layout(
                xaxis_title='Journal',
                yaxis_title='Number of Papers',
                title={'x': 0.5},
                xaxis_tickangle=-45
            )

            output_file = f'top_journals_{method.lower().replace(" ", "_")}.html'
            output_path = os.path.join(output_dir, output_file)
            fig.write_html(output_path)
            fig.show()


    @staticmethod
    def plot_top_authors(relevant_df: pd.DataFrame, output_dir, top_n=10):
        """
        Plot the top authors based on the number of papers published.
        """
        top_authors = relevant_df['First Author'].value_counts().head(top_n).reset_index()
        top_authors.columns = ['Author', 'Paper Count']

        fig = px.bar(
            top_authors,
            x='Author',
            y='Paper Count',
            labels={
                'Author': 'Author',
                'Paper Count': 'Number of Papers'
            },
            title=f'Top {top_n} Authors',
        )

        fig.update_layout(
            xaxis_title='Author',
            yaxis_title='Number of Papers',
            title={'x': 0.5},
            xaxis_tickangle=-45
        )

        output_path = os.path.join(output_dir, 'top_authors.html')
        fig.write_html(output_path)
        fig.show()


    @staticmethod
    def plot_publications_per_journal_over_time(relevant_df: pd.DataFrame, output_dir, top_n=5):
        """
        Plot the number of publications per year for the top journals.
        """
        # Identify top journals
        top_journals = relevant_df['Journal/Book'].value_counts().head(top_n).index.tolist()
        df_top_journals = relevant_df[relevant_df['Journal/Book'].isin(top_journals)]

        # Group by year and journal
        publications_over_time = df_top_journals.groupby(['year', 'Journal/Book']).size().reset_index(name='counts')

        fig = px.line(
            publications_over_time,
            x='year',
            y='counts',
            color='Journal/Book',
            markers=True,
            labels={
                'year': 'Publication Year',
                'counts': 'Number of Papers',
                'Journal/Book': 'Journal'
            },
            title=f'Publications Over Time for Top {top_n} Journals',
        )

        fig.update_layout(
            xaxis_title='Publication Year',
            yaxis_title='Number of Papers',
            legend_title_text='Journal',
            title={'x': 0.5},
        )

        fig.update_xaxes(
            dtick=1,
            tickmode='linear',
            tickformat='d'
        )

        output_path = os.path.join(output_dir, 'publications_per_journal_over_time.html')
        fig.write_html(output_path)
        fig.show()


    @staticmethod
    def plot_publication_distribution_per_journal(relevant_df: pd.DataFrame, output_dir, top_n=10):
        """
        Plot the distribution of publications across journals.

        Parameters:
            relevant_df: DataFrame containing relevant papers
            output_dir: Directory to save the plot
            top_n: Number of top journals to display
        """
        journal_counts = relevant_df['Journal/Book'].value_counts().head(top_n).reset_index()
        journal_counts.columns = ['Journal/Book', 'Paper Count']

        fig = px.pie(
            journal_counts,
            names='Journal/Book',
            values='Paper Count',
            title=f'Distribution of Publications Across Top {top_n} Journals',
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )

        fig.update_layout(
            title={'x': 0.5},
        )

        output_path = os.path.join(output_dir, 'publication_distribution_per_journal.html')
        fig.write_html(output_path)
        fig.show()
