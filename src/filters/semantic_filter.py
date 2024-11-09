
import sys
import spacy
import pandas as pd
import subprocess

from typing import Dict, Tuple, Set
from collections import defaultdict
from spacy.matcher import Matcher, PhraseMatcher
from spacy.language import Language
from sentence_transformers import SentenceTransformer, util

from src.utils.constants import Constants


class SemanticFilter:
    def __init__(self, spacy_model: str = 'en_core_web_lg', transformer_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize SemanticFilter with specified spaCy model.

        Args: spacy_model: Name of the spaCy model to use
        """
        # Load spaCy model and SentenceTransformer
        self.nlp               = self.load_spacy_model(spacy_model)
        self.transformer       = SentenceTransformer(transformer_model)

        # Embed domain-specific sentences for semantic relevance
        self.dl_embeddings     = self.transformer.encode(Constants.DL_SENTENCES, convert_to_tensor=True)
        self.domain_embeddings = self.transformer.encode(Constants.DOMAIN_SENTENCES, convert_to_tensor=True)

        # Initialize matchers
        self.dl_matcher        = self.initialize_dl_matcher()
        self.domain_matchers   = self.initialize_domain_matchers()
        self.medical_matcher   = self.initialize_medical_matcher()
        self.method_matchers   = self.initialize_method_matchers()
        self.negative_matcher  = self.initialize_negative_matcher()


    @staticmethod
    def load_spacy_model(model_name) -> Language:
        """
        Load spaCy model with error handling.
        """
        try:
            spacy_model = spacy.load(model_name)

            return spacy_model
        except OSError:
            print(f"Installing spaCy model: {model_name}")

            try:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    check=True,
                    capture_output=True
                )
                return spacy.load(model_name)
            except subprocess.CalledProcessError as e:
                print(f"Failed to install spaCy model: {model_name}")
                raise


    def initialize_dl_matcher(self) -> PhraseMatcher:
        """
        Initialize and configure spaCy's PhraseMatcher for deep learning patterns.
        """
        # The PhraseMatcher matches exact phrases, which may miss variations.
        # Consider using Matcher with token-based patterns for more flexibility.
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        for category, terms in Constants.DL_PATTERNS.items():
            # Convert terms to spaCy docs
            patterns = [self.nlp.make_doc(term) for term in terms]
            # Add patterns to the matcher
            matcher.add(category, patterns)

        return matcher


    def initialize_domain_matchers(self) -> Dict[str, PhraseMatcher]:
        """
        Initialize matchers for domain-specific patterns.
        """
        domain_matchers = {}

        for domain, terms in Constants.DOMAIN_PATTERNS.items():
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = [self.nlp.make_doc(term) for term in terms]
            matcher.add(domain, patterns)
            domain_matchers[domain] = matcher

        return domain_matchers


    def initialize_medical_matcher(self) -> PhraseMatcher:
        """
        Initialize PhraseMatcher for medical terms to filter out irrelevant papers.
        """
        matcher = spacy.matcher.PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(term) for term in Constants.MEDICAL_TERMS]
        matcher.add("MEDICAL", patterns)

        return matcher


    def initialize_method_matchers(self) -> Dict[str, PhraseMatcher]:
        """
        Initialize matchers for method classification patterns.
        """
        method_matchers = {}
        for method_type, terms in Constants.METHOD_PATTERNS.items():
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

            patterns = [self.nlp.make_doc(term) for term in terms]

            matcher.add(method_type, patterns)

            method_matchers[method_type] = matcher

        return method_matchers


    def initialize_negative_matcher(self) -> PhraseMatcher:
        """
        Initialize PhraseMatcher for negative keywords to filter out irrelevant papers.
        """
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        negative_patterns = [self.nlp.make_doc(term) for term in Constants.NEGATIVE_KEYWORDS]

        matcher.add("NEGATIVE_KEYWORD", negative_patterns)

        return matcher


    def is_semantic_relevant_by_similarity(self, text: str) -> Tuple[bool, Dict[str, float], str]:
        """
        Check if text is semantically relevant using sentence embeddings.

        To use Gensim for similarity for topic modeling and semantic analysis or SentenceTransformer ?

        :param text: Input text to analyze containing title and abstract of the paper
        :return: is_relevant
        """
        # Embedding the text and convert to tensor for cosine similarity
        text_embedding = self.transformer.encode(text, convert_to_tensor=True)

        # Compute cosine similarities with Deep Learning sentences
        dl_cosine_scores = util.cos_sim(text_embedding, self.dl_embeddings)
        dl_max_similarity = dl_cosine_scores.max().item()

        # Compute cosine similarity with domain sentences
        domain_cosine_scores = util.cos_sim(text_embedding, self.domain_embeddings)
        domain_max_similarity = domain_cosine_scores.max().item()

        # Set thresholds
        dl_similarity_threshold = 0.39
        domain_similarity_threshold = 0.2

        is_dl_relevant = dl_max_similarity >= dl_similarity_threshold
        is_domain_relevant = domain_max_similarity >= domain_similarity_threshold

        is_relevant = is_dl_relevant and is_domain_relevant

        reason = 'Relevant' if is_relevant else 'Not relevant'

        scores = {
            'dl_similarity_score': dl_max_similarity,
            'domain_similarity_score': domain_max_similarity
        }

        return is_relevant, scores, reason


    def is_semantic_relevant_by_pattern_matching(self, text: str) -> Tuple[bool, Dict[str, float], str]:
        """
        Check if text is semantically relevant using spaCy's matcher.

        :param text: Input text to analyze containing title and abstract of the paper
        :return: Tuple of (is_relevant, confidence_scores, reason)
        """
        # Check for missing or invalid text
        if pd.isna(text) or not isinstance(text, str):
            return False, { 'architecture': 0.0, 'task': 0.0, 'context': 0.0 }, 'Invalid text'

        # Check if the text contains negative keywords and no medical context before proceeding
        # Do not count as negative if in positive context
        if self.contains_negative_keywords(text) and not self.contains_medical_terms(text):
            return False, { 'architecture': 0.0, 'task': 0.0, 'context': 0.0 }, 'Negative keywords found'

        # First check for medical context that would invalidate the paper
        if self.is_medical_context(text):
            return False, {'architecture': 0.0, 'task': 0.0, 'context': 0.0}, 'Only medical context found'

        try:
            doc = self.nlp(text.lower())

            # Use the matcher to find relevant terms in the text
            matches = self.dl_matcher(doc)

            # Count unique matches per category
            matched_terms: Dict[str, Set[str]] = defaultdict(set)
            for match_id, start, end in matches:
                category = self.nlp.vocab.strings[match_id]
                term = doc[start:end].text
                matched_terms[category].add(term)

            # Calculate scores based on matched terms
            scores = {
                category: min(len(terms) * Constants.WEIGHTS.get(category, 0.0), 1.0)
                for category, terms in matched_terms.items()
            }
            scores = defaultdict(float, scores)

            # Match domain-specific terms
            domain_matches = []
            for domain, matcher in self.domain_matchers.items():
                domain_matches.extend(matcher(doc))

            if not domain_matches:
                return False, dict(scores), 'irrelevant domain'

            is_relevant = (
                    (scores['architecture'] >= 0.3 and scores['task'] >= 0.3) or
                    (scores['architecture'] >= 0.6 and scores['context'] >= 0.1) or
                    (scores['task'] >= 0.6 and scores['context'] >= 0.1)
            )

            return is_relevant, dict(scores), 'relevant domain and patterns'

        except Exception as e:
            print(f"Error in is_semantic_relevant: {e}")
            return False, {'architecture': 0.0, 'task': 0.0, 'context': 0.0}, 'Error during processing'


    def is_medical_context(self, text: str) -> bool:
        """
        Check if the text is only about medical conditions.

        Args:
            text: Input text to analyze.
        Returns:
            True if the text is primarily medical, False otherwise.
        """
        doc = self.nlp(text)
        medical_matches = self.medical_matcher(doc)
        computational_matches = self.dl_matcher(doc)

        return len(medical_matches) > 0 and len(computational_matches) == 0


    def contains_medical_terms(self, text: str) -> bool:
        """
        Check if the text contains medical terms.

        Args:
            text: Input text to analyze.
        Returns:
            True if the text contains medical terms, False otherwise.
        """
        doc = self.nlp(text)

        matches = self.medical_matcher(doc)

        return len(matches) > 0


    def contains_negative_keywords(self, text: str) -> bool:
        """
        Check if the text contains negative keywords.

        :param text:
        :return:
        """
        doc = self.nlp(text.lower())

        matches = self.negative_matcher(doc)

        return len(matches) > 0


    def classify_method(self, text: str) -> str:
        """
        Classify the method using improved pattern matching and context.

        Args:text: Input text to classify
        Returns: Classification result: 'text_mining', 'computer_vision', 'both', or 'other'
        """
        if pd.isna(text) or not isinstance(text, str):
            return "other"

        try:
            doc = self.nlp(text)

            scores = {}
            for method_type, matcher in self.method_matchers.items():
                matches = matcher(doc)
                scores[method_type] = len(matches)

            # Determine method type based on the matches
            text_mining     = scores.get(Constants.TOPICS['text_mining'], 0) > 0
            computer_vision = scores.get(Constants.TOPICS['computer_vision'], 0) > 0

            if text_mining and computer_vision:
                return Constants.TOPICS['both']
            elif text_mining:
                return Constants.TOPICS['text_mining']
            elif computer_vision:
                return Constants.TOPICS['computer_vision']
            else:
                return Constants.TOPICS['other']

        except Exception as e:
            print(f"Error in classify_method: {e}")
            return "other"


    def extract_method_name(self, text: str) -> str:
        """
        Extract the method name from the text using NER and pattern matching.

        Args:
            text: Input text to extract method name from.

        Returns:
            Extracted method name or an empty string if none found.
        """
        if not text or not isinstance(text, str):
            return 'not specified'

        # Define relevant verbs and phrases indicating method usage
        relevant_verbs = {
            'use', 'implement', 'develop', 'propose',
            'apply', 'employ', 'utilize', 'design', 'train', 'introduce'
        }

        introduction_phrases = {'called', 'known as', 'referred to as', 'termed'}

        try:
            doc = self.nlp(text)
            method_names = set()

            # Use Matcher for pattern matching
            matcher = Matcher(self.nlp.vocab)

            # Pattern: Verb + Determiner (optional) + Adjective(s) + Noun (method name)
            pattern = [
                {'LEMMA': {'IN': list(relevant_verbs)}, 'POS': 'VERB'},
                {'OP': '*'},
                {'POS': 'DET', 'OP': '?'},
                {'POS': 'ADJ', 'OP': '*'},
                {'POS': 'NOUN', 'OP': '+'},
            ]
            matcher.add('METHOD_PATTERN', [pattern])

            # Pattern: Phrases indicating method introduction
            for phrase in introduction_phrases:
                pattern = [
                    {'LOWER': phrase.split()[0]},
                    {'LOWER': phrase.split()[1]} if len(phrase.split()) > 1 else {},
                    {'OP': '*'},
                    {'POS': 'PROPN', 'OP': '+'},
                ]
                matcher.add('INTRO_PATTERN', [pattern])

            matches = matcher(doc)

            for match_id, start, end in matches:
                span = doc[start:end]
                method_candidate = span.text

                # Refine method_candidate
                method_candidate = self.clean_method_name(method_candidate)

                if self.is_valid_method_name(method_candidate):
                    method_names.add(method_candidate)

            # Also look for specific architecture terms in the text
            for term in Constants.DL_PATTERNS['architecture']:
                if term.lower() in text.lower():
                    method_names.add(term)

            # Clean and deduplicate method names
            method_names = [method.strip() for method in method_names if len(method.strip()) > 2]

            return ', '.join(method_names) if method_names else 'Not specified'

        except Exception as e:
            print(f"Error in extract_method_name: {e}")
            return ""


    @staticmethod
    def clean_method_name(method_candidate: str) -> str:
        """
        Clean and normalize the extracted method name.

        Args:
            method_candidate: The raw method name extracted.

        Returns:
            Cleaned method name.
        """
        # Remove unnecessary characters and whitespace
        method_candidate = method_candidate.strip(' .,;-:"\'')
        return method_candidate


    @staticmethod
    def is_valid_method_name(text: str) -> bool:
        """
        Check if the extracted text is a valid method name.

        Args:
            text: Extracted text to validate.

        Returns:
            True if valid, False otherwise.
        """
        # Avoid common false positives
        if len(text.split()) > 7:
            return False

        # Check if the text contains any relevant terms
        text_lower = text.lower()
        return any(
            term.lower() in text_lower
            for terms in Constants.DL_PATTERNS.values()
            for term in terms
        ) or any(keyword in text_lower for keyword in ['model', 'network', 'algorithm', 'approach', 'method'])
