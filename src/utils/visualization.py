
import os
import pandas as pd
import networkx as nx

import plotly.express as px
import plotly.graph_objects as go

from wordcloud import WordCloud


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


    @staticmethod
    def plot_journal_comparison(relevant_df: pd.DataFrame, irrelevant_df: pd.DataFrame, output_dir, top_n=10):
        """
        Compare the top journals in relevant and irrelevant papers.

        Parameters:
            relevant_df: DataFrame containing relevant papers
            irrelevant_df: DataFrame containing irrelevant papers
            output_dir: Directory to save the plot
            top_n: Number of top journals to display
        """
        relevant_journal_counts = relevant_df['Journal/Book'].value_counts().head(top_n).reset_index()
        relevant_journal_counts.columns = ['Journal/Book', 'Relevant Paper Count']

        irrelevant_journal_counts = irrelevant_df['Journal/Book'].value_counts().head(top_n).reset_index()
        irrelevant_journal_counts.columns = ['Journal/Book', 'Irrelevant Paper Count']

        # Merge the two DataFrames on 'Journal/Book'
        journal_comparison = pd.merge(
            relevant_journal_counts,
            irrelevant_journal_counts,
            on='Journal/Book',
            how='outer'
        ).fillna(0)

        # Melt the DataFrame for easier plotting
        journal_comparison_melted = journal_comparison.melt(
            id_vars='Journal/Book',
            value_vars=['Relevant Paper Count', 'Irrelevant Paper Count'],
            var_name='Paper Type',
            value_name='Count'
        )

        fig = px.bar(
            journal_comparison_melted,
            x='Journal/Book',
            y='Count',
            color='Paper Type',
            barmode='group',
            labels={
                'Journal/Book': 'Journal',
                'Count': 'Number of Papers',
                'Paper Type': 'Paper Type'
            },
            title=f'Comparison of Top Journals in Relevant and Irrelevant Papers',
        )

        fig.update_layout(
            xaxis_title='Journal',
            yaxis_title='Number of Papers',
            legend_title_text='Paper Type',
            title={'x': 0.5},
            xaxis_tickangle=-45
        )

        output_path = os.path.join(output_dir, 'journal_comparison.html')
        fig.write_html(output_path)
        fig.show()

    @staticmethod
    def plot_irrelevant_papers_per_year(irrelevant_df: pd.DataFrame, output_dir):
        """
        Plot the number of irrelevant papers per year.

        Parameters:
            irrelevant_df: DataFrame containing irrelevant papers
            output_dir: Directory to save the plot
        """
        irrelevant_df['year'] = pd.to_numeric(irrelevant_df['Publication Year'], errors='coerce')
        papers_per_year = irrelevant_df.groupby('year').size().reset_index(name='counts')

        fig = px.bar(
            papers_per_year,
            x='year',
            y='counts',
            labels={
                'year': 'Publication Year',
                'counts': 'Number of Irrelevant Papers'
            },
            title='Number of Irrelevant Papers per Year',
        )

        fig.update_layout(
            xaxis_title='Publication Year',
            yaxis_title='Number of Papers',
            title={'x': 0.5},
        )

        fig.update_xaxes(
            dtick=1,
            tickmode='linear',
            tickformat='d'
        )

        output_path = os.path.join(output_dir, 'irrelevant_papers_per_year.html')
        fig.write_html(output_path)
        fig.show()

    @staticmethod
    def plot_irrelevance_scores_distribution(irrelevant_df: pd.DataFrame, output_dir):
        """
        Plot the distribution of scores leading to irrelevance.

        Parameters:
            irrelevant_df: DataFrame containing irrelevant papers with scores
            output_dir: Directory to save the plot
        """
        # Assuming 'scores' column is a dictionary with keys like 'architecture', 'task', 'context'
        # We need to extract these into separate columns
        scores_df = pd.json_normalize(irrelevant_df['scores'])
        combined_df = pd.concat([irrelevant_df.reset_index(drop=True), scores_df], axis=1)

        # Melt the DataFrame to plot distributions
        scores_melted = combined_df.melt(
            id_vars=['PMID'],
            value_vars=['architecture', 'task', 'context'],
            var_name='Score Type',
            value_name='Score'
        )

        fig = px.histogram(
            scores_melted,
            x='Score',
            color='Score Type',
            barmode='overlay',
            histnorm='percent',
            nbins=20,
            labels={
                'Score': 'Score Value',
                'percent': 'Percentage of Papers',
                'Score Type': 'Score Type'
            },
            title='Distribution of Irrelevance Scores',
        )

        fig.update_layout(
            xaxis_title='Score Value',
            yaxis_title='Percentage of Papers',
            legend_title_text='Score Type',
            title={'x': 0.5},
        )

        output_path = os.path.join(output_dir, 'irrelevance_scores_distribution.html')
        fig.write_html(output_path)
        fig.show()


    @staticmethod
    def plot_method_occurrence_network(relevant_df: pd.DataFrame, output_dir, text_column: str = 'Method Name'):
        # Extract method names
        relevant_df[text_column] = relevant_df[text_column].fillna('Not specified')

        method_lists = relevant_df[text_column].apply(
            lambda x: [m.strip() for m in x.split(',') if m.strip() != 'Not specified'])

        # Build co-occurrence matrix
        cooccurrence = {}
        for methods in method_lists:
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    pair = tuple(sorted([methods[i], methods[j]]))
                    cooccurrence[pair] = cooccurrence.get(pair, 0) + 1

        # Create graph
        G = nx.Graph()
        for pair, weight in cooccurrence.items():
            G.add_edge(pair[0], pair[1], weight=weight)

        # Draw graph
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        edge_trace = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    line=dict(width=weight * 0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
            )

        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            text=[node for node in G.nodes()],
            mode='markers+text',
            textposition='top center',
            hoverinfo='text',
            marker=dict(
                size=10,
                color='#FFA07A',
            )
        )

        fig = go.Figure(data=edge_trace + [node_trace],
                        layout=go.Layout(
                            title='Method Co-occurrence Network',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                        ))

        output_path = os.path.join(output_dir, 'method_occurrence_network.html')
        fig.write_html(output_path)
        fig.show()

    @staticmethod
    def plot_word_cloud(relevant_df: pd.DataFrame, output_dir, text_column: str = 'Method Name'):
        """
        Generate a word cloud from the specified text column and display it using Plotly.

        Parameters:
            relevant_df: DataFrame containing relevant papers
            output_dir: Directory to save the plot
            text_column: Column from which to generate the word cloud
        """
        text_data = ' '.join(relevant_df[text_column].dropna().tolist())

        # Generate the word cloud image
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            # stopwords={'and', 'or', 'of'}
        ).generate(text_data)

        # Create a Plotly figure with the word cloud image
        fig = px.imshow(wordcloud.to_array(), title=f'Word Cloud of {text_column}')

        # Hide axes for a cleaner look
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(
            title={'x': 0.5},
            margin=dict(l=0, r=0, t=30, b=0)
        )

        # Save the figure as an HTML file
        output_path = os.path.join(output_dir, f'word_cloud_{text_column.replace(" ", "_").lower()}.html')
        fig.write_html(file=output_path, auto_open=False)

        # Show the figure
        fig.show()