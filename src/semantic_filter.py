import sys
import spacy
import pandas as pd
import subprocess

from typing      import Dict, Tuple, Set
from collections import defaultdict

from src.constants import Constants


class SemanticFilter:
    def __init__(self):
        """
        Initialize SemanticFilter with specified spaCy model.

        Args:
            model_name: Name of the spaCy model to use
        """
        self.nlp             = self.load_spacy_model()
        self.matcher         = self.initialize_matcher()
        self.medical_matcher = self.initialize_medical_matcher()
        self.method_matchers = self.initialize_method_matchers()


    def load_spacy_model(self, model_name: str = 'en_core_web_lg') -> spacy.language.Language:
        """Load spaCy model with error handling."""
        try:
            return spacy.load(model_name)
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


    def initialize_matcher(self) -> spacy.matcher.PhraseMatcher:
        """
        Initialize and configure spaCy's PhraseMatcher for deep learning patterns.
        """
        matcher = spacy.matcher.PhraseMatcher(self.nlp.vocab, attr="LOWER")

        for category, terms in Constants.DL_PATTERNS.items():
            patterns = [self.nlp.make_doc(term) for term in terms]
            matcher.add(category, patterns)

        return matcher


    def initialize_medical_matcher(self) -> spacy.matcher.PhraseMatcher:
        """
        Initialize PhraseMatcher for medical terms to filter out irrelevant papers.
        """
        matcher = spacy.matcher.PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(term) for term in Constants.MEDICAL_TERMS]
        matcher.add("MEDICAL", patterns)

        return matcher


    def initialize_method_matchers(self) -> Dict[str, spacy.matcher.PhraseMatcher]:
        """
        Initialize matchers for method classification patterns.
        """
        method_matchers = {}
        for method_type, terms in Constants.METHOD_PATTERNS.items():
            matcher = spacy.matcher.PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = [self.nlp.make_doc(term) for term in terms]
            matcher.add(method_type, patterns)
            method_matchers[method_type] = matcher
        return method_matchers


    def is_semantic_relevant(self, text: str) -> Tuple[bool, Dict[str, float]]:
        """
        Check if text is semantically relevant using spaCy's matcher.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (is_relevant, confidence_scores)
        """
        if pd.isna(text) or not isinstance(text, str):
            return False, {'architecture': 0.0, 'task': 0.0, 'context': 0.0}

        # First check for medical context that would invalidate the paper
        if self.is_medical_context(text):
            return False, {'architecture': 0.0, 'task': 0.0, 'context': 0.0}

        try:
            doc = self.nlp(text.lower())
            matches = self.matcher(doc)

            # Count unique matches per category
            matched_terms: Dict[str, Set[str]] = defaultdict(set)
            for match_id, start, end in matches:
                category = self.nlp.vocab.strings[match_id]
                term = doc[start:end].text
                matched_terms[category].add(term)

            # Calculate scores based on matched terms
            scores = {
                category: min(len(terms) * Constants.WEIGHTS[category], 1.0)
                for category, terms in matched_terms.items()
            }

            # Set default scores for missing categories
            scores = defaultdict(float, scores)

            # Determine relevance based on the scores and criteria
            is_relevant = (
                    scores['architecture'] > 0 or
                    (scores['task'] >= 0.5 and scores['context'] > 0) or
                    ('algorithm' in text.lower() and scores['task'] > 0)
            )

            return is_relevant, dict(scores)
        except Exception as e:
            print(f"Error in is_semantic_relevant: {e}")
            return False, {'architecture': 0.0, 'task': 0.0, 'context': 0.0}


    def is_medical_context(self, text: str) -> bool:
        """
        Check if the text is primarily about medical conditions.

        Args:
            text: Input text to analyze.

        Returns:
            True if the text is primarily medical, False otherwise.
        """
        doc = self.nlp(text)
        medical_matches = self.medical_matcher(doc)
        computational_matches = self.matcher(doc)

        # If there are more medical matches than computational matches, it's likely medical context
        return len(medical_matches) > len(computational_matches)


    def classify_method(self, text: str) -> str:
        """
        Classify the method using improved pattern matching and context.

        Args:
            text: Input text to classify

        Returns:
            Classification result: 'text_mining', 'computer_vision', 'both', or 'other'
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
            text_mining = scores.get('text_mining', 0) > 0
            computer_vision = scores.get('computer_vision', 0) > 0

            if text_mining and computer_vision:
                return "both"
            elif text_mining:
                return "text_mining"
            elif computer_vision:
                return "computer_vision"
            else:
                return "other"

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
            return ""

        try:
            doc = self.nlp(text)
            method_names = set()

            # Use NER to find potential method names
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART', 'TECH']:
                    if self.is_valid_method_name(ent.text):
                        method_names.add(ent.text)

            # Also look for specific architecture terms in the text
            for term in Constants.DL_PATTERNS['architecture']:
                if term.lower() in text.lower():
                    method_names.add(term)

            return ', '.join(method_names)

        except Exception as e:
            print(f"Error in extract_method_name: {e}")
            return ""


    def is_valid_method_name(self, text: str) -> bool:
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
        )
