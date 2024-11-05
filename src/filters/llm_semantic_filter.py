import asyncio
import logging
import re
import yaml
from dataclasses import dataclass

from datetime import datetime

from typing import Dict

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama


@dataclass
class ClassificationResult:
    method_type: str
    method_name: str
    relevant: bool
    reasoning: str


class PaperClassifier:
    def __init__(self):
        """Initialize the paper classifier with configuration."""

        # Valid method types
        self.VALID_METHOD_TYPES = {"text_mining", "computer_vision", "both", "other"}

        # Initialize LLM
        self.model = self.initialize_llm_model()

        # Initialize the output parser
        self.output_parser = JsonOutputParser()

        # Create the classification chain
        self.classification_chain = self.create_chain()


    def initialize_llm_model(self):
        """
        Initialize the LLM model with the correct parameters.
        """
        return ChatOllama(
            model='llama3.2:3b',
            format='json',
            temperature=0.1,
            top_p=0.9,  # Slightly constrained sampling
            repeat_penalty=1.1  # Discourage repetition
        )

    def create_chain(self):
        """
        Create the classification chain with improved prompt.
        """

        CLASSIFICATION_TEMPLATE = """
        You are an expert for analyzing computer science research papers in Virology/Epidemiology. 
        Your task is to carefully analyze the input paper and classify its computational methods.

        Guidelines for classification:
        1. RELEVANCE criteria:
           - Clear use of  text mining, machine learning, deep learning, or AI methods
           
        2. METHOD TYPE criteria:
           - "text_mining": NLP, text analysis, content analysis, social media analysis, etc.
           - "computer_vision": Image processing, visual analysis, object detection, etc.
           - "both": When both text and image analysis are used
           - "other": When no clear method type is mentioned

        3. METHOD NAME criteria:
           - Specific approach or algorithm names (e.g., "BERT", "CNN", "Inception-v3")
           - For multiple methods, choose the primary one

        4. REASONING criteria:
              - Brief explanation of the classification decision (max 100 chars)
              
        IMPORTANT: 
        - Some papers may not use computational methods and 
        could be about cognitive science, psychology, medical research, syndrome like computer vision syndrome, etc.
        These are not relevant and should be filtered out as irrelevant.
        - Focus only on the computational methods used in the paper.
        
        Paper to analyze:
        {text_input}

        Provide your analysis in the following JSON format:
        {{
            "relevant": boolean,     // True if uses relevant computational methods
            "method_type": string,   // "text_mining", "computer_vision", "both", or "other"
            "method_name": string,   // Specific method/algorithm name or empty string
            "reasoning": string,     // Brief explanation (max 100 chars)
        }}
        """

        return PromptTemplate(
            template=CLASSIFICATION_TEMPLATE,
            input_variables=["text_input"]
        ) | self.model | self.output_parser


    def validate_result(self, result: Dict) -> None:
        """Validate classification results."""
        if "method_type" not in result or result["method_type"] not in self.VALID_METHOD_TYPES:
            raise ValueError(f"Invalid method_type: {result.get('method_type')}")

        if not isinstance(result.get("relevant"), bool):
            raise ValueError("Missing or invalid 'relevant' field")

        if not isinstance(result.get("reasoning"), str):
            raise ValueError("Missing or invalid 'reasoning' field")


    async def classify(self, text: str) -> ClassificationResult:
        """
        Classify a paper based on its text.
        """

        try:
            start_time = datetime.now()

            # Clean input text
            cleaned_text = self.preprocess_text(text)

            # Get classification
            result = await self.classification_chain.ainvoke({
                "text_input": cleaned_text
            })

            # Validate result format
            self.validate_result(result)

            # Log performance
            processing_time = (datetime.now() - start_time).total_seconds()

            return ClassificationResult(
                method_type=result["method_type"],
                method_name=result["method_name"],
                relevant=result["relevant"],
                reasoning=result["reasoning"]
            )

        except Exception as e:
            logging.error(f"Error in classify: {e}")
            raise


    def preprocess_text(self, text: str) -> str:
        """Preprocess the input text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-.,;:()\[\]"\'%]', '', text)
        return text


async def main():
    """
    Main function to run the classifier.
    """

    try:
        classifier = PaperClassifier()

        # Example paper text
        text = """
                Title: Health Warnings on Instagram Advertisements for Synthetic Nicotine E-Cigarettes and Engagement
                Abstract: IMPORTANCE: Synthetic nicotine is increasingly used in e-cigarette liquids along with flavors to appeal to youths. Regulatory loopholes have allowed tobacco manufacturers to use social media to target youths.
                OBJECTIVE: To analyze the extent to which synthetic nicotine e-cigarette brands have implemented US Food and Drug Administration (FDA) health warning requirements and to evaluate the association between health warnings and user engagement on Instagram.
                DESIGN, SETTING, AND PARTICIPANTS: In this cross-sectional study, posts from 25 brands were analyzed across a 14-month period (August 2021 to October 2022). A content analysis was paired with Warning Label Multi-Layer Image Identification, a computer vision algorithm designed to detect the presence of health warnings and whether the detected health warning complied with FDA guidelines by (1) appearing on the upper portion of the advertisement and (2) occupying at least 20% of the advertisement's area. Data analysis was performed from March to June 2024.
                EXPOSURE: Synthetic nicotine e-cigarette advertisement on Instagram.
                MAIN OUTCOMES AND MEASURES: The outcome variables were user engagement (number of likes and comments). Negative binomial regression analyses were used to evaluate the association between the presence and characteristics of health warnings and user engagement.
                RESULTS: Of a total of 2071 posts, only 263 (13%) complied with both FDA health warning requirements. Among 924 posts with health warnings, 732 (79%) displayed warnings in the upper image portion, and 270 (29%) had a warning covering at least 20% of the pixel area. Posts with warnings received fewer comments than posts without warnings (mean [SD], 1.8 [2.5] vs 5.4 [11.7] comments; adjusted incident rate ratio [aIRR], 0.70; 95% CI, 0.57-0.86; P < .001). For posts containing warnings, a larger percentage of the warning label's pixel area was associated with fewer comments (aIRR, 0.96; 95% CI, 0.93-0.99; P = .003). Flavored posts with health warnings placed in the upper image portion received more likes than posts with warnings in the lower portion (mean [SD], 34.6 [35.2] vs 19.9 [19.2] likes; aIRR, 1.48; 95% CI, 1.07-2.06; P = .02).
                CONCLUSIONS AND RELEVANCE: In this cross-sectional study of synthetic nicotine brand Instagram accounts, 87% of sampled posts did not adhere to FDA health warning requirements in tobacco promotions. Enforcement of FDA compliant health warnings on social media may reduce youth engagement with tobacco marketing.
            """

        # Run classification
        result = await classifier.classify(text)

        # Print results
        print("\nClassification Results:")
        print(f"Method Type: {result.method_type}")
        print(f"Method Name: {result.method_name}")
        print(f"Relevant: {result.relevant}")
        print(f"Reasoning: {result.reasoning}")

    except Exception as e:
        print(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
