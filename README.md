# 📰 News Sentiment Analysis & RAG Pipeline

> An end-to-end NLP system that classifies sentiment in news articles and answers fact-based questions using Retrieval-Augmented Generation (RAG).

---

## 🔍 Overview

This project builds a complete Natural Language Processing pipeline applied to a large multi-source news dataset (CNN, BBC, Dawn, TRT, Fox News, Al Jazeera). It benchmarks **three generations of NLP models** — classical ML, deep learning, and Transformers — for sentiment classification, then extends into a **RAG-based question answering system** that retrieves relevant article chunks and generates grounded, citation-backed answers.

---

## ✨ Key Features

- **Multi-model benchmarking** across ML, DL, and Transformer architectures
- **Automated sentiment annotation** using a pretrained multilingual BERT model
- **Fine-tuned DistilBERT** achieving the best classification performance
- **RAG pipeline** with MPNet embeddings + cosine retrieval + Flan-T5 generation
- **Comprehensive evaluation** with Accuracy, F1, AUC, Top-k Accuracy, Exact Match, and Confusion Matrix
- **Citation-grounded answers** — the RAG system cites the source article for every response

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Classical ML | Scikit-learn · Logistic Regression · Linear SVM · TF-IDF |
| Deep Learning | PyTorch · Bi-LSTM · TextCNN |
| Transformer | HuggingFace · DistilBERT (fine-tuned) |
| RAG Embeddings | SentenceTransformers · `all-mpnet-base-v2` |
| RAG Generator | Flan-T5-Base (with beam search) |
| Retrieval Index | Scikit-learn · NearestNeighbors (cosine similarity) |

---

## 🗂️ Pipeline Architecture

```
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
```

---

## 📊 Results Summary

| Model | Type | Performance |
|---|---|---|
| Logistic Regression | Classical ML | Baseline |
| Linear SVM | Classical ML | Best classical baseline |
| TextCNN | Deep Learning | Significant improvement over ML |
| Bi-LSTM | Deep Learning | Better than TextCNN (long-range dependencies) |
| **DistilBERT** | **Transformer** | **Best overall — highest Accuracy, F1, AUC** |

The RAG system substantially outperformed the standalone Flan-T5 baseline by producing context-grounded, citation-backed answers and significantly reducing hallucination.

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/news-sentiment-rag.git
cd news-sentiment-rag

# Install dependencies
pip install -r requirements.txt

# Run sentiment classification
python train_classifier.py --model distilbert

# Run RAG pipeline
python rag_pipeline.py --query "What is the impact of AI on news reporting?"
```

---

## 📁 Project Structure

```
├── data/               # Dataset merging and preprocessing scripts
├── models/
│   ├── ml/             # Logistic Regression & SVM
│   ├── dl/             # Bi-LSTM & TextCNN
│   └── transformer/    # DistilBERT fine-tuning
├── rag/
│   ├── embeddings/     # MPNet chunk embeddings
│   ├── retrieval/      # NearestNeighbors cosine index
│   └── generation/     # Flan-T5 generator with prompting
├── evaluation/         # Metrics and result plots
└── report/             # Final project report
```

---

## 👥 Authors

- **Hannan Asad** (23L-0808)
- **Abdul Rehman Shahid** (23L-0720)
- **Shoaib Asif** (23L-0710)

---

## 🔮 Future Work

- Swap Flan-T5 for stronger generators (Phi-2, Gemma-2)
- GPU-accelerated retrieval with FAISS
- Cross-encoder reranking for better retrieval precision
- Web UI for real-time question answering
- Multilingual and multi-category expansion

---

## 📄 License

This project was developed as part of an academic NLP course. For reuse, please contact the authors.
