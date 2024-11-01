
from typing import Dict, Set, List


class Constants:
    # Medical terms to exclude/disambiguate
    MEDICAL_TERMS: Set[str] = {
        'computer vision syndrome',
        'cvs-q',
        'visual health',
        'visual symptoms',
        'visual fatigue',
        'eye strain',
        'ocular surface',
        'visual stress',
        'digital eye strain',
        'visual discomfort',
        'ophthalmology',
        'optometry',
        'eye disease',
        'retina',
        'glaucoma',
        'cataract'
    }

    # Method classification patterns with type hints
    METHOD_PATTERNS: Dict[str, Set[str]] = {
        'text_mining': {
            'nlp',
            'text mining',
            'natural language processing',
            'text analysis',
            'named entity recognition',
            'sentiment analysis',
            'text classification',
            'language model',
            'word embedding',
            'document classification',
            'topic modeling',
            'text extraction',
            'text generation',
            'text summarization',
            'information extraction',
            'text understanding',
            'transcriptomics',
            'content analysis',
            'post analysis',
            'comments',
            'engagement analysis',
            'social media',
            'posts',
            'biomarkers',
        },
        'computer_vision': {
            'computer vision',
            'computer-aided vision',
            'image processing',
            'image processing model',
            'image analysis',
            'detection'
            'detection strategy'
            'image detection',
            'image recognition',
            'visual recognition',
            'classification',
            'segmentation',
            'image classification',
            'image identification',
            'image segmentation',
            'image generation',
            'spatial inference',
            'image', 'imaging', 'scanning',
            'visual analysis', 'object detection',
            'scene understanding',
            'visual detection',
            'visual pattern recognition',
            'multi-layer image',
            'image identification',
            'visual inspection', 'image area', 'visual analysis',
        }
    }

    # Define semantic patterns for deep learning with type hints
    DL_PATTERNS: Dict[str, List[str]] = {
        'architecture': [
            'neural network', 'deep learning', 'deep-learning',
            'convolutional neural network', 'recurrent neural network',
            'lstm', 'transformer', 'bert', 'gpt', 'gru', 'encoder', 'decoder',
            'u-net', 'autoencoder', 'gan', 'generative adversarial network',
            'neural architecture', 'deep neural', 'cnn', 'rnn', 'gnn',
            'attention mechanism', 'transformer model', 'resnet', 'vgg',
            'artificial neural network', 'machine learning', 'artificial intelligence',
            'ai model', 'ml model', 'computational model', 'long short-term memory',
            'graph neural network', 'capsule network', 'self-attention'
        ],
        'task': [
            'classification', 'detection', 'prediction', 'recognition',
            'segmentation', 'generation', 'identification',
            'image identification', 'detect', 'analysis', 'forecasting',
            'clustering',
            'feature extraction',
            'pattern recognition',
            'anomaly detection',
            'automated analysis',
            'automated detection',
            'automated recognition',
            'automated identification'
        ],
        'context': [
            'train', 'predict', 'model', 'learn', 'optimize',
            'architecture', 'layer', 'network', 'deep', 'inference',
            'supervised', 'unsupervised', 'fine-tuning', 'embedding',
            'backpropagation', 'gradient descent', 'optimization',
            'algorithm', 'computational', 'automated', 'processing',
            'analysis', 'detection system'
        ]
    }

    # Scoring weights
    WEIGHTS: Dict[str, float] = {
        'architecture': 1.0,
        'task': 0.5,
        'context': 0.25
    }

    # Method indicators with weights
    METHOD_INDICATORS: Dict[str, float] = {
        "using": 1.5,
        "based on": 2.0,
        "through": 1.0,
        "implements": 2.0,
        "applying": 1.5,
        "utilizing": 1.5,
        "featuring": 1.0,
        "incorporating": 1.5
    }
