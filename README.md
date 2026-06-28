# News-Sentiment-Analysis-RAG-Pipeline
An end-to-end NLP system that classifies sentiment in news articles and answers fact-based questions using Retrieval-Augmented Generation (RAG).

🔍 Overview

This project builds a complete Natural Language Processing pipeline applied to a large multi-source news dataset (CNN, BBC, Dawn, TRT, Fox News, Al Jazeera). It benchmarks three generations of NLP models — classical ML, deep learning, and Transformers — for sentiment classification, then extends into a RAG-based question answering system that retrieves relevant article chunks and generates grounded, citation-backed answers.


✨ Key Features


Multi-model benchmarking across ML, DL, and Transformer architectures
Automated sentiment annotation using a pretrained multilingual BERT model
Fine-tuned DistilBERT achieving the best classification performance
RAG pipeline with MPNet embeddings + cosine retrieval + Flan-T5 generation
Comprehensive evaluation with Accuracy, F1, AUC, Top-k Accuracy, Exact Match, and Confusion Matrix
Citation-grounded answers — the RAG system cites the source article for every response



🛠️ Tech Stack

ComponentTechnologyClassical MLScikit-learn · Logistic Regression · Linear SVM · TF-IDFDeep LearningPyTorch · Bi-LSTM · TextCNNTransformerHuggingFace · DistilBERT (fine-tuned)RAG EmbeddingsSentenceTransformers · all-mpnet-base-v2RAG GeneratorFlan-T5-Base (with beam search)Retrieval IndexScikit-learn · NearestNeighbors (cosine similarity)

Raw News Data (6 sources)
        │
        ▼
 Merging & Cleaning
        │
        ▼
Sentiment Annotation (BERT-based labeler → Negative / Neutral / Positive)
        │
        ▼
┌───────────────────────────────────────┐
│        Model Training & Evaluation    │
│  ┌─────────┐  ┌──────────┐  ┌──────┐ │
│  │ ML      │  │ DL       │  │ BERT │ │
│  │ LR/SVM  │  │ Bi-LSTM  │  │Distil│ │
│  │ TF-IDF  │  │ TextCNN  │  │BERT  │ │
│  └─────────┘  └──────────┘  └──────┘ │
└───────────────────────────────────────┘
        │
        ▼
  RAG Pipeline
  ┌──────────────────────────────────────┐
  │  Article Chunking (220w / 40w overlap)│
  │  → MPNet Embeddings                  │
  │  → Cosine Retrieval Index            │
  │  → Flan-T5 Generator + Citations     │
  └──────────────────────────────────────┘

  📊 Results Summary

ModelTypePerformanceLogistic RegressionClassical MLBaselineLinear SVMClassical MLBest classical baselineTextCNNDeep LearningSignificant improvement over MLBi-LSTMDeep LearningBetter than TextCNN (long-range dependencies)DistilBERTTransformerBest overall — highest Accuracy, F1, AUC

The RAG system substantially outperformed the standalone Flan-T5 baseline by producing context-grounded, citation-backed answers and significantly reducing hallucination.

👥 Authors

Hannan Asad 
Abdul Rehman Shahid
Shoaib Asif
