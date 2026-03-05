# 🛡️ CVE Intelligence Analyzer

An end-to-end NLP project that automatically analyzes CVE (Common Vulnerabilities and Exposures) descriptions and converts them into short, actionable security alerts.

## Overview

Security teams process hundreds of CVE reports daily. This project uses Natural Language Processing to:

- **Classify** vulnerability types (XSS, RCE, SQLi, etc.) using fine-tuned DistilBERT
- **Predict** severity levels using ML models trained on CVSS data
- **Extract** key security terms using TF-IDF analysis
- **Summarize** long CVE descriptions into concise alerts using BART
- **Search** for similar vulnerabilities using Sentence-BERT embeddings
- **Visualize** trends through an interactive Streamlit dashboard

### Example

**Input:**
> A buffer overflow vulnerability exists in Apache HTTP Server 2.4.58 that allows remote attackers to execute arbitrary code via crafted HTTP requests.

**Output:**
> 🔴 CVE Alert: Critical Remote Code Execution
>
> Summary: Attackers can execute arbitrary code through crafted HTTP requests.
>
> Recommended Action: Patch immediately. Isolate affected systems.

---

## Dataset

**Source:** [CVE 2024 Database - Kaggle](https://www.kaggle.com/datasets/manavkhambhayata/cve-2024-database-exploits-cvss-os)

| Column | Description |
|--------|-------------|
| CVE ID | Unique vulnerability identifier |
| Description | Full vulnerability description |
| CVSS Score | Severity score (0-10) |
| Attack Vector | CVSS attack vector string |
| Affected OS | Affected operating systems |

---

## Project Structure

```
CVE_NLP/
├── data/
│   └── cve_dataset.csv              # Raw dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and visualization
│   ├── 02_text_preprocessing.ipynb   # NLP text cleaning pipeline
│   ├── 03_tfidf_keyword_extraction.ipynb  # TF-IDF analysis
│   ├── 04_vulnerability_classification_bert.ipynb  # DistilBERT classifier
│   ├── 05_text_summarization.ipynb   # BART summarization
│   ├── 06_severity_prediction.ipynb  # ML severity prediction
│   └── 07_similarity_search.ipynb    # Sentence-BERT similarity
├── models/                           # Saved models and artifacts
├── app/
│   └── streamlit_app.py             # Interactive dashboard
├── utils/
│   ├── preprocessing.py             # Text preprocessing utilities
│   └── model_utils.py               # Model I/O and alert generation
├── requirements.txt
└── README.md
```

---

## NLP Pipeline

```
CVE Description
      │
      ▼
┌─────────────────┐
│ Text Preprocessing │  ← Cleaning, tokenization, lemmatization
└────────┬────────┘
         │
    ┌────┼────────────────┬──────────────────┐
    ▼    ▼                ▼                  ▼
┌──────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│TF-IDF│ │DistilBERT│ │  BART    │ │Sentence-BERT │
│Keywords│ │Classifier│ │Summarizer│ │ Embeddings   │
└──┬───┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘
   │          │             │              │
   ▼          ▼             ▼              ▼
Keywords  Vuln Type      Summary      Similar CVEs
              │
              ▼
        ┌───────────┐
        │ Severity  │  ← Random Forest / Logistic Regression
        │ Predictor │
        └───────────┘
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd CVE_NLP

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional)
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## Running the Notebooks

Run notebooks in order — each builds on the previous one's output:

```bash
cd notebooks
jupyter notebook
```

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | Data Exploration | EDA, visualizations, severity distribution |
| 02 | Text Preprocessing | Clean text, tokenize, lemmatize |
| 03 | TF-IDF Extraction | Keyword extraction, term analysis |
| 04 | BERT Classification | Fine-tune DistilBERT for vuln type classification |
| 05 | Text Summarization | Generate concise alerts with BART |
| 06 | Severity Prediction | Train RF/LR models for severity prediction |
| 07 | Similarity Search | Build Sentence-BERT embedding index |

---

## Running the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard has three sections:

1. **CVE Analyzer** — Paste a CVE description, get instant classification, severity, keywords, and summary
2. **Vulnerability Trends** — Interactive charts showing CVE distributions by severity, year, type, and vendor
3. **Similar CVE Search** — Semantic search to find CVEs with similar attack patterns

> **Note:** The dashboard works with or without trained models. Without models, it falls back to rule-based analysis. Run all notebooks first for full functionality.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| NLP | NLTK, spaCy |
| ML | scikit-learn |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Embeddings | Sentence-Transformers |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |

---

## Models Used

| Model | Task | Source |
|-------|------|--------|
| DistilBERT | Vulnerability classification | `distilbert-base-uncased` |
| BART | Text summarization | `facebook/bart-large-cnn` |
| Sentence-BERT | Similarity search | `all-MiniLM-L6-v2` |
| Random Forest / Logistic Regression | Severity prediction | scikit-learn |
| TF-IDF | Keyword extraction | scikit-learn |
