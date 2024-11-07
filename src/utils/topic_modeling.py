


def perform_topic_modeling(relevant_df: pd.DataFrame, output_dir, num_topics=5):
    # Preprocessing
    texts = relevant_df['combined_text'].dropna().tolist()
    processed_texts = []

    for text in texts:
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words]
        processed_texts.append(tokens)

    # Create a dictionary and corpus
    dictionary = Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Build LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)

    # Visualize topics
    lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    output_path = os.path.join(output_dir, 'lda_topics.html')
    pyLDAvis.save_html(lda_vis, output_path)
    pyLDAvis.show(lda_vis)
