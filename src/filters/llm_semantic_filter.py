
import asyncio
import logging
import re
from dataclasses import dataclass

from datetime import datetime

from typing import Dict

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

@dataclass
class ClassificationResult:
    method_type: str
    method_name: str
    relevant: bool
    reasoning: str


class LLMSemanticFilter:
    def __init__(self):
        """Initialize the paper classifier with configuration."""

        # Valid method types
        self.VALID_METHOD_TYPES = {"text_mining", "computer_vision", "both", "other", ""}

        # Initialize LLM
        self.model = self.initialize_llm_model()

        # Initialize the output parser
        self.output_parser = JsonOutputParser()

        # Create the classification chain
        self.classification_chain = self.create_chain()


    @staticmethod
    def initialize_llm_model():
        """
        Initialize the LLM model with the correct parameters.
        """
        return ChatOllama(
            model='llama3.1:8b',  # llama3.2:3b, llama3.1:8b,
            format='json',
        )


    def create_chain(self):
        """
        Create the classification chain with improved prompt.
        """

        CLASSIFICATION_TEMPLATE = """
        You are an expert in analyzing computer science research papers related to virology and epidemiology.
        Your task is to carefully analyze the input paper to identify if it discusses deep learning or artificial intelligence methods in the context of virology or epidemiology.

        **PRIMARY CRITERIA:**
        1. **Deep Learning Focus:**
           - Check if the paper discusses deep learning or AI methods.
           - Consider only papers that uses or discusses these methods in the context of virology or epidemiology.
           - Exclude papers that mention these terms only in passing, in the introduction, or as future work.
   
        2. **Method Classification:**
           - **text_mining:**
             * NLP for genomic sequences
             * Text analysis of clinical records
             * Language models for medical literature
             * Social media analysis for disease tracking

           - **computer_vision:**
             * Medical image analysis (X-rays, CT scans)
             * Microscopy image processing
             * Pathogen detection in images
             * Visual diagnosis systems

           - **both:** When both text and image analysis are central to the method

           - **other:** Novel deep learning approaches that don't fit above categories

        3. **Method Identification:**
           Look for specific architecture names like:
           - For Text: BERT, GPT, RNN, LSTM, Transformer
           - For Vision: CNN, ResNet, U-Net, YOLO
           - General: Neural Networks, Deep Learning, Transfer Learning

        4. **Reasoning:**
              Provide the main field of the paper followed by a brief explanation of why it is relevant or irrelevant (max 50).
        
        
        **EXCLUSION CRITERIA:**
        - Papers using deep learning or AI methods in domains **other than** virology or epidemiology (e.g., urban planning, energy efficiency).
        - Traditional statistical methods or machine learning without deep learning, neural networks or AI.
        - Pure epidemiological studies without computational methods.
        - Clinical trials or medical studies without AI or computational components.
        - Medical research not involving computational methods (e.g., purely laboratory-based studies).
        - Papers mentioning "computer vision syndrome" (a medical condition unrelated to computer vision techniques).
        - Future work or proposed applications without actual implementation.

        **EXAMPLE ANALYSIS:**
        Paper: "BERT-based analysis of viral mutation patterns from clinical records"
        {{
            "relevant": true,
            "method_type": "text_mining",
            "method_name": "BERT",
            "reasoning": "virology, uses BERT model to analyze clinical text data for viral mutations"
        }}

        Paper: "Deep learning-based analysis of urban traffic patterns for smart city planning"
        {{
            "relevant": false,
            "method_type": "",
            "method_name": "",
            "reasoning": "urban planning, uses deep learning but in urban planning context, not virology or epidemiology"
        }}

        Paper: "Clinical study of computer vision syndrome in hospital workers"
        {{
            "relevant": false,
            "method_type": "",
            "method_name": "",
            "reasoning": "ophthalmology, unrelated to computer vision techniques in AI or deep learning"
        }}

        Paper: "Convolutional Neural Networks for Breast Cancer Detection in Mammograms"
        {{
            "relevant": false,
            "method_type": "",
            "method_name": "",
            "reasoning": "oncology, uses CNNs but not in virology or epidemiology context"
        }}

        Important Notes:
        - A paper is considered **relevant** only if it **mentions** the use of deep learning or AI methods **applied specifically** in the context of **virology** or **epidemiology**.
        - If the paper uses these methods in **any other field** (e.g., urban planning, energy efficiency, oncology, cardiology), it should be marked as **irrelevant**, even if it involves medical imaging or biological data.
        - Do **not** assume relevance based on the use of general medical terms; focus on the specific domains of virology and epidemiology.

        Now analyze this paper:
        {text_input}

        Provide your analysis in the following JSON format:
        {{
            "relevant": boolean,     
            "method_type": string,   
            "method_name": string,   
            "reasoning": string
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
        classifier = LLMSemanticFilter()

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
