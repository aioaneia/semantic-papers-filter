from typing import Dict, Set, List


class Constants:
    # enum for topics
    TOPICS = {
        'text_mining': 'text_mining',
        "computer_vision": "computer_vision",
        'both': 'both',
        'other': 'other'
    }

    # Medical terms to exclude/disambiguate
    MEDICAL_TERMS: Set[str] = {
        'computer vision syndrome',
        'cvs-q',
        'digital eye strain',
        'eye strain',
        'epidemiological',
        'epidemiological data',
        'visual health',
        'visual symptoms',
        'visual fatigue',
        'visual stress',
        'visual discomfort',
        'ocular surface',
        'ophthalmology',
        'optometry',
        'eye disease',
        'retina',
        'glaucoma',
        'cataract',
        'epidemiological study',
        'clinical trial',
        'virus transmission',
        'vaccination',
        'vaccine development',
        'cell',
        'immunology',
        'pathogen',
        'infectious disease',
        'tumor',
        'healthcare',
        'transcriptomics',
    }

    NEGATIVE_KEYWORDS = [
        'urban planning', 'building design',
        'households', 'residential buildings', 'urban planners', 'smart cities', 'sustainable cities',
        'policymakers', 'external shading', 'operable window types',
        'Google Street View', 'demographic characteristics',
        'forest',
    ]

    DOMAIN_PATTERNS: Dict[str, Set[str]] = {
        'virology': {
            'virus', 'viral', 'viruses', 'virology', 'pathogen', 'pathogens',
            'infected', 'infection', 'infections', 'infectious disease', 'infectious diseases',
            'infected cells', 'viral infection', 'viral infections', 'viral disease',
            'antiviral', 'microbiology', 'viral load', 'viral replication',
            'host-virus interaction', 'immunology', 'vaccination', 'vaccine', 'vaccines',
            'epidemic', 'pandemic', 'outbreak', 'serology', 'virome', 'virion',
            'viral genome', 'host immune response', 'COVID-19', 'SARS-CoV-2', 'coronavirus',
            'e. coli', 'influenza', 'HIV', 'hepatitis', 'dengue', 'Zika', 'Ebola', 'measles',
            'viral transmission', 'viral replication', 'viral evolution', 'viral mutation', 'mutation',
            'viral pathogenesis', 'viral protein', 'viral vector', 'viral vector vaccine',
            'antibodies', 'antibody', 'antigen', 'antigenic', 'antigenic variation',
            'genetic recombination', 'genetic diversity', 'genetic variation', 'genetic mutation',
            'viral evolution', 'viral diversity', 'viral mutation', 'viral recombination',
            'genomic sequence', 'genomic data', 'genomic analysis', 'genomic epidemiology', 'genomic',
            'monkeypox',
        },
        'epidemiology': {
            'epidemiology', 'epidemiological', 'disease spread', 'disease transmission',
            'disease modeling', 'public health', 'contact tracing', 'disease surveillance',
            'incidence', 'prevalence', 'population health', 'health statistics',
            'risk factors', 'mortality', 'morbidity', 'case fatality rate',
            'disease outbreak', 'epidemic curve', 'reproduction number'
        }
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
            'textual metadata',
            'report Generation',
            'question Answering',
            'summarization',
            'text summarization',
            'information extraction',
            'text understanding',
            'content analysis',
            'post analysis',
            'comments',
            'engagement analysis',
            'social media',
            'posts',
            'biomarkers',
            'gene network analysis',
            'network structural information',
            'clinical notes',
            'clinical data',
            'clinical text',
            'medical records',
            'genomic sequence',
            'clinical text analysis',
            'genetic analysis',
            'gene expression',
            'variant analysis',
            'mutation detection',
            'phenotype analysis',
            'infodemiology',
            'language understanding',
            'semantic parsing',
        },
        'computer_vision': {
            '3D reconstruction',
            'classification',
            'computer vision',
            'computer-aided vision',
            'detection',
            'detection strategy',
            'detection model',
            'image',
            'imaging',
            'image processing',
            'image processing model',
            'image analysis',
            'image detection',
            'image recognition',
            'image identification',
            'image classification',
            'image segmentation',
            'image generation',
            'image harmonization',
            'image area',
            'pose estimation',
            'object detection',
            'object tracking',
            'segmentation',
            'spatial inference',
            'scene understanding',
            'vision-language',
            'vision-language model',
            'visual analysis',
            'visual inspection',
            'visual recognition',
            'visual pattern recognition',
            'visual Question Answering',
            'multi-layer image',
            'convolutional neural network',
            'ct scan',
            'mri',
            'x-ray',
            'microscopy image',
            'histopathology image',
            'medical image',
            'lesion detection',
            'lesion segmentation',
            'lesion classification',
            'lesion identification',
            'lesion analysis',
            'lesion recognition',
            'lesion localization',
            'tumor detection',
            'tumor segmentation',
            'tumor classification',
            'tumor identification',
            'tumor analysis',
            'cell detection',
            'cell counting',
        }
    }

    # Define semantic patterns for deep learning with type hints
    DL_PATTERNS: Dict[str, List[str]] = {
        'architecture': [
            'ai model',
            'artificial intelligence',
            'artificial neural network',
            'attention mechanism',
            'autoencoder',
            'alexnet',

            'bert',
            'bert-like models',

            'capsule network',
            'convolutional neural network',
            'cnn',

            'densenet',
            'deep learning',
            'deep-learning',
            'deep neural',
            'decoder',

            'encoder',
            'encoder-decoder',
            'efficientnet',

            'foundation model',
            'feedforward neural network',

            'gan',
            'gpt',
            'gru',
            'gnn',
            'generative adversarial network',
            'graph neural network',
            'graph convolutional network',

            'inceptionv3',
            'inception',
            'inception network',

            'lstm',
            'language model',
            'long short-term memory',

            'machine learning',
            'mobilenet', 'mobilenetv2', 'mobilenetv3', 'mobilenetv4', 'mobilenetv5',

            'neural network',
            'neural architecture',

            'pre-trained model',
            'pre-trained network',
            'pre-trained architecture',

            'rnn',
            'r-cnn',
            'residual network',
            'recurrent neural network',
            'resnet',

            'transformer',
            'transformer-based model',
            'transformer model',
            'transformer architecture',
            'transformer network',

            'self-attention',
            'self-attention mechanism',
            'self-attention layer',
            'seq2seq',
            'swin transformer',

            'u-net',
            'unet',
            'unet++',
            'unetr++',

            'variational autoencoder',
            'vgg',
            'vision transformer',
            'visual transformer',
            'vision-language model',
            'vae',

            'word embedding',
            'word2vec',

            'yolo',
        ],
        'task': [
            'classification',
            'detection',
            'detect',
            'prediction',
            'recognition',
            'segmentation',
            'generation',
            'identification',
            'image identification',
            'forecasting',
            'clustering',
            'feature extraction',
            'pattern recognition',
            'anomaly detection',
            'automated detection',
            'automated recognition',
            'automated identification',
            'sequence prediction',
            'mutation detection',
            'strain classification',
            'outbreak prediction',
            'genomic analysis',
        ],
        'context': [
            'model training',
            'feature learning',
            'data augmentation',
            'transfer learning',
            'layer',
            'network',
            'deep',
            'inference',
            'supervised',
            'unsupervised',
            'fine-tuning',
            'embedding',
            'backpropagation',
            'gradient descent',
            'multimodal',
        ]
    }

    # Scoring weights
    WEIGHTS: Dict[str, float] = {
        'architecture': 0.6,
        'task':         0.3,
        'context':      0.1
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

    # Domain-specific sentences for semantic similarity
    DOMAIN_SENTENCES = [
        # Deep Learning Applications in Virology and Epidemiology
        "We apply deep learning models to predict the spread of infectious diseases.",
        "Our research uses convolutional neural networks to detect viral infections from medical images.",
        "We develop machine learning algorithms for real-time epidemiological surveillance.",
        "The study employs deep learning to analyze genomic sequences of viruses.",
        "We utilize neural networks to model virus transmission dynamics.",
        "Our approach applies artificial intelligence to identify patterns in epidemiological data.",
        "We investigate the use of deep learning for vaccine efficacy prediction.",
        "Our model predicts viral mutations using recurrent neural networks.",
        "We leverage deep learning techniques for outbreak detection and response.",
        "The research focuses on applying machine learning to understand virus-host interactions.",
        "We use deep learning to analyze social media data for early detection of disease outbreaks.",
        "Our study employs graph neural networks to model disease transmission networks.",
        "We develop deep learning frameworks for the classification of pathogens from genomic data.",
        "Our approach integrates deep learning with epidemiological models to forecast pandemic spread.",
        "We use bioinformatics and deep learning tools to study viral genome evolution.",
        "Our research aims to enhance public health initiatives using AI and machine learning.",
        "We apply transformer-based models to analyze large-scale epidemiological datasets.",
        "Our system uses deep learning for automated contact tracing analysis.",
        "We utilize deep learning for drug discovery targeting viral infections.",
        "Our study focuses on deep learning applications in virology and epidemiology.",

        # Specific Applications and Methods
        "We develop a deep learning model to detect COVID-19 from chest X-ray images.",
        "Our research employs machine learning to predict disease outbreaks based on climate data.",
        "We use convolutional neural networks for segmentation of lesions in MRI scans of infected patients.",
        "Our approach applies deep learning to identify zoonotic spillover events.",
        "We utilize unsupervised learning to cluster viral strains based on genetic similarity.",
        "Our model uses attention mechanisms to improve prediction of infection rates.",
        "We implement a deep learning pipeline for high-throughput screening of antiviral compounds.",
        "Our study integrates multimodal data using deep learning for comprehensive epidemiological analysis.",
        "We employ deep learning to model the impact of interventions on disease spread.",
        "Our research uses deep reinforcement learning to optimize vaccination strategies.",

        # Combining Deep Learning with Other Biomedical Research
        "We apply deep learning to analyze microscopy images for rapid pathogen detection.",
        "Our study uses deep learning to predict protein structures of novel viruses.",
        "We develop neural network models to identify potential drug targets in viral genomes.",
        "Our approach utilizes deep learning for phylogenetic analysis of viral evolution.",
        "We employ deep learning to detect antimicrobial resistance patterns.",
        "Our research integrates deep learning with CRISPR technology for gene editing applications.",
        "We use deep learning to analyze single-cell RNA sequencing data in infected tissues.",
        "Our model predicts host immune response using deep learning algorithms.",
        "We leverage deep learning for personalized treatment strategies in infectious diseases.",
        "Our study applies deep learning to understand the spread of antimicrobial resistance.",

        # Public Health and Policy Applications
        "We develop AI models to support decision-making in public health policy.",
        "Our research uses deep learning to assess the effectiveness of quarantine measures.",
        "We apply machine learning to optimize resource allocation during pandemics.",
        "Our approach uses deep learning to analyze mobility data for infection risk assessment.",
        "We employ deep learning for sentiment analysis of public opinion on health interventions.",
        "Our system predicts hospital admission rates using deep learning models.",
        "We utilize deep learning to identify misinformation related to infectious diseases.",
        "Our study uses AI to enhance telemedicine services during outbreaks.",
        "We implement deep learning for real-time monitoring of disease spread using wearable devices.",
        "Our research applies deep learning to evaluate the socio-economic impact of epidemics.",
    ]

    # Define deep learning-related sentences for semantic similarity
    DL_SENTENCES = [
        "Deep learning models are used to classify viral sequences in genomic studies.",
        "We utilize machine learning to predict epidemic trends based on historical data.",
        "Our approach applies neural networks to detect viral mutations from sequence data.",
        "The study employs deep learning for automated detection of viral infections in medical images.",

        # General Deep Learning Applications
        "This study utilizes deep learning techniques to analyze medical data.",
        "We apply neural networks to predict disease outcomes.",
        "A convolutional neural network is developed for image classification tasks.",
        "Our research employs recurrent neural networks for time-series analysis.",
        "The model is based on transformer architecture for natural language processing.",
        "We implement deep learning algorithms to improve diagnostic accuracy.",
        "A deep neural network is trained to identify patterns in genomic data.",
        "Our approach leverages machine learning methods for predictive modeling.",
        "We use unsupervised learning to discover hidden structures in the data.",
        "The research focuses on supervised learning for classification problems.",
        "We introduce an attention mechanism to enhance model performance.",
        "Our method involves transfer learning to utilize pre-trained models.",
        "We adopt a generative adversarial network for data augmentation.",
        "The study applies autoencoders for feature extraction and dimensionality reduction.",
        "We propose a novel deep learning framework for medical image segmentation.",
        "Our system integrates reinforcement learning for decision-making processes.",
        "We utilize graph neural networks to model complex relationships in data.",
        "The deep learning model is optimized using backpropagation and gradient descent.",
        "We employ deep belief networks for hierarchical feature learning.",
        "The study uses ensemble learning to improve prediction accuracy.",

        # Applications in Virology and Epidemiology
        "Deep learning techniques are applied to predict viral mutations.",
        "We use neural networks to model the spread of infectious diseases.",
        "A machine learning approach is used for epidemic forecasting.",
        "Our study employs convolutional neural networks to detect COVID-19 from CT scans.",
        "We analyze genomic sequences using deep learning methods.",
        "The research applies recurrent neural networks to predict infection rates.",
        "We develop a transformer-based model for virus classification.",
        "Our model identifies potential antiviral compounds using deep learning.",
        "We utilize deep learning for automated contact tracing analysis.",
        "The study employs machine learning to understand virus-host interactions.",
        "We implement a deep learning framework to analyze social media data for outbreak detection.",
        "Our approach uses deep learning to predict vaccine efficacy.",
        "We apply unsupervised learning to cluster viral genome sequences.",
        "The model uses attention mechanisms to focus on key genomic features.",
        "We leverage deep learning for protein structure prediction in viruses.",
        "Our system detects zoonotic spillover events using machine learning.",
        "We utilize graph convolutional networks to model transmission networks.",
        "Deep learning models are trained to predict drug resistance mutations.",
        "We use deep neural networks to classify pathogens from metagenomic data.",
        "Our research employs deep learning for phylogenetic analysis.",

        # Deep Learning in Epidemiology and Public Health
        "We use deep reinforcement learning to optimize intervention strategies in epidemics.",
        "Our research applies deep learning to model human mobility and disease spread.",
        "We implement LSTM networks to predict future outbreak hotspots.",
        "Our approach uses deep learning to analyze social networks for contact tracing.",
        "We employ deep learning to detect early signals of outbreaks from web search data.",
        "Our model integrates satellite imagery and deep learning for environmental risk assessment.",
        "We use deep learning to analyze electronic health records for syndromic surveillance.",
        "Our study applies deep learning to model seasonal patterns in infectious diseases.",
        "We develop deep learning algorithms for real-time pandemic forecasting.",
        "Our research uses deep learning to evaluate the effectiveness of public health policies.",

        # Deep Learning in Microbiology and Microscopy
        "We apply deep learning techniques to analyze microscopy images for bacterial detection.",
        "A convolutional neural network is developed for identification of bacteria in microscopy images.",
        "Our research employs deep learning to enhance detection of pathogens in clinical samples.",
        "We implement CNN-based algorithms for quantification of bacterial cells using dark field microscopy.",
        "The study utilizes AI-enabled analytics to improve detection specificity in microbiological diagnostics.",
        "We use deep learning-powered computer vision to identify microorganisms in complex samples.",
        "Our approach integrates image processing models with deep learning for pathogen detection.",
        "We employ deep learning methods to analyze dark field microscopy images for bacterial identification.",
        "The research focuses on using neural networks for rapid detection of foodborne pathogens.",
        "We develop a deep learning framework for precise identification and quantification of bacterial cells.",

        # Medical Imaging and Computer Vision Applications
        "A convolutional neural network is used for lesion segmentation in medical images.",
        "We develop a deep learning model for tumor detection in MRI scans.",
        "Our method applies deep learning to classify skin lesions from images.",
        "We employ deep learning for retinal image analysis in diabetic patients.",
        "The study uses deep neural networks for automated cancer diagnosis.",
        "We utilize U-Net architecture for medical image segmentation tasks.",
        "Our approach enhances image resolution using super-resolution techniques.",
        "We implement a deep learning model for organ segmentation in CT scans.",
        "The research applies deep learning for fracture detection in X-ray images.",
        "We develop a deep learning-based system for histopathological image analysis.",
        "Our model detects pneumonia from chest X-ray images using CNNs.",
        "We use deep learning to analyze ultrasound images for fetal health assessment.",
        "The study employs deep learning for brain tumor classification.",
        "We apply transfer learning to improve medical image classification.",
        "Our approach uses 3D convolutional neural networks for volumetric data analysis.",
        "We implement a deep learning model for blood vessel segmentation.",
        "The research uses deep learning to detect Alzheimer’s disease from PET scans.",
        "We utilize deep learning for automated detection of COVID-19 pneumonia.",
        "Our system classifies breast cancer subtypes using deep learning algorithms.",
        "We apply deep learning to analyze microscopy images for cellular analysis.",
        "AI-based quantitative image analysis of screening CTs in osteoporosis screening",

        # Natural Language Processing and Text Mining Applications
        "We use natural language processing to extract information from clinical notes.",
        "Our study applies deep learning for sentiment analysis of health-related tweets.",
        "We implement a transformer-based model for biomedical text classification.",
        "The research employs BERT for named entity recognition in medical documents.",
        "We utilize deep learning to summarize scientific articles automatically.",
        "Our approach uses deep learning for question answering on medical datasets.",
        "We develop a language model to predict adverse drug reactions.",
        "The study applies topic modeling to identify trends in health records.",
        "We use word embeddings to represent biomedical terminology.",
        "Our method leverages deep learning for automated coding of medical procedures.",
        "We implement deep learning for language translation in medical contexts.",
        "The research employs sequence-to-sequence models for report generation.",
        "We use deep learning to classify patient feedback for quality improvement.",
        "Our system detects misinformation in health-related social media posts.",
        "We apply sentiment analysis to assess public opinion on vaccination.",
        "Our model predicts disease outbreaks by analyzing news articles.",
        "We utilize deep learning for automated chart review and data extraction.",
        "The study employs deep learning for speech recognition in telemedicine.",
        "We develop a chatbot using deep learning for patient engagement.",
        "Our approach uses deep learning for optical character recognition of medical documents.",

        # Genomics and Bioinformatics Applications
        "We apply deep learning to predict gene expression levels.",
        "Our study uses deep neural networks for SNP classification.",
        "We develop a deep learning model for protein function prediction.",
        "The research employs deep learning for enhancer-promoter interaction prediction.",
        "We use convolutional neural networks to analyze DNA methylation data.",
        "Our method predicts transcription factor binding sites using deep learning.",
        "We utilize deep learning for splice site prediction in genomics.",
        "The study applies deep learning to identify non-coding RNAs.",
        "We implement deep learning for chromatin state segmentation.",
        "Our model predicts gene-disease associations using deep learning.",
        "We employ deep learning for de novo genome assembly.",
        "The research uses deep learning to analyze single-cell RNA sequencing data.",
        "We develop a deep learning framework for epigenetic landscape modeling.",
        "Our approach applies deep learning to metabolic pathway prediction.",
        "We use deep learning for phylogenetic tree reconstruction.",
        "The study utilizes deep learning for microbiome data classification.",
        "We implement deep learning to predict protein-ligand interactions.",
        "Our model identifies genetic variants associated with diseases using deep learning.",
        "We employ deep learning for genotype imputation in population genetics.",
        "Our research uses deep learning to analyze CRISPR off-target effects.",

        # Drug Discovery and Computational Chemistry Applications
        "We use deep learning to predict drug-target interactions.",
        "Our study develops a deep learning model for virtual screening of compounds.",
        "We implement deep learning for quantitative structure-activity relationship modeling.",
        "The research employs deep learning for molecular property prediction.",
        "We utilize deep generative models to design novel drug molecules.",
        "Our approach uses deep learning to predict pharmacokinetic properties.",
        "We apply deep learning for toxicity prediction of chemical compounds.",
        "The study uses deep learning to model protein folding.",
        "We develop a deep learning framework for predicting bioactivity of molecules.",
        "Our model identifies potential drug repurposing opportunities using deep learning.",
        "We employ deep learning for side effect prediction in drug development.",
        "The research applies deep learning to analyze high-throughput screening data.",
        "We use deep learning to optimize drug delivery systems.",
        "Our method predicts molecular docking scores using deep learning.",
        "We utilize deep learning for cheminformatics and compound classification.",
        "The study develops a deep learning model for ADMET property prediction.",
        "We implement deep learning to understand molecular dynamics simulations.",
        "Our approach uses deep learning for antibody design and optimization.",
        "We apply deep learning for predicting chemical reaction outcomes.",
        "Our research uses deep learning to analyze nanomaterials for biomedical applications.",

        # Other Relevant Applications
        "We develop a deep learning model for speech emotion recognition in therapy sessions.",
        "Our study applies deep learning to predict patient readmission rates.",
        "We use deep learning to analyze electronic health records for disease prediction.",
        "The research employs deep learning for personalized treatment recommendation.",
        "We implement deep learning for anomaly detection in healthcare data.",
        "Our approach uses deep learning to model patient survival rates.",
        "We utilize deep learning for optimizing healthcare logistics and resource allocation.",
        "The study applies deep learning to monitor patient vital signs remotely.",
        "We develop a deep learning framework for fraud detection in medical insurance claims.",
        "Our model predicts patient adherence to medication using deep learning.",
        "We employ deep learning for health risk assessment based on lifestyle data.",
        "The research uses deep learning to analyze genetic networks and interactions.",
        "We apply deep learning for disease subtype classification.",
        "Our method utilizes deep learning for multimodal data integration in healthcare.",
        "We implement deep learning for real-time health monitoring using wearable devices.",
        "The study employs deep learning to enhance telemedicine services.",
        "We use deep learning to predict surgical outcomes and complications.",
        "Our approach applies deep learning for health informatics and data mining.",
        "We utilize deep learning for patient stratification in clinical trials.",
        "Our research employs deep learning to model the spread of antimicrobial resistance.",

        # Statements Reflecting Deep Learning Use
        "This paper introduces a deep learning approach for disease prediction.",
        "We propose a novel neural network architecture that outperforms existing models.",
        "Our deep learning model achieves state-of-the-art results on epidemiological data.",
        "We conduct extensive experiments using our deep learning framework.",
        "The results demonstrate the effectiveness of our deep learning techniques in this domain.",
        "We explore advanced deep learning applications in virology and epidemiology.",
        "Our study highlights the potential of deep learning in public health research.",
        "We compare various deep learning models to identify the best-performing one.",
        "The deep learning method significantly improves prediction accuracy over baselines.",
        "We investigate the impact of different deep learning architectures on our results.",
        "Our findings support the integration of deep learning into epidemiological modeling.",
        "We validate our deep learning model using cross-validation and external datasets.",
        "The deep learning approach is seamlessly integrated into our analytical pipeline.",
        "We discuss challenges and solutions related to deep learning implementation.",
        "Our work contributes by introducing novel deep learning methods to the field.",
        "We leverage deep learning to uncover complex patterns in biomedical data.",
        "The proposed model is based on cutting-edge deep learning architectures.",
        "We demonstrate how deep learning can be effectively applied to our problem.",
        "Our methodology involves training deep neural networks on large datasets.",
        "We show that deep learning provides superior performance in our application.",
    ]