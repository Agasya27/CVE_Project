# Models Directory

This directory stores all trained models and artefacts used by the CVE Intelligence Analyzer.

## Files present after running the notebooks

| File / Folder | Notebook | Size (approx) | Description |
|---|---|---|---|
| `bert_classifier/` | `04_vulnerability_classification_bert.ipynb` | ~260 MB | Fine-tuned DistilBERT for vulnerability-type classification (9 classes, 79% accuracy) |
| `label_encoder.joblib` | 04 | ~1 KB | LabelEncoder mapping integer predictions to class names |
| `tfidf_vectorizer.joblib` | `03_tfidf_keyword_extraction.ipynb` | ~1 MB | TF-IDF vectorizer (5 000 features, unigrams + bigrams) |
| `tfidf_matrix.npz` | 03 | ~3 MB | Sparse TF-IDF matrix for all 1 314 CVEs |
| `severity_predictor.joblib` | `06_severity_prediction.ipynb` | ~2 MB | Logistic Regression severity classifier (70% accuracy) |
| `severity_tfidf.joblib` | 06 | ~1 MB | TF-IDF vectorizer used for severity feature extraction |
| `severity_encoder.joblib` | 06 | ~1 KB | LabelEncoder for severity labels |
| `severity_vuln_columns.joblib` | 06 | ~1 KB | Column list for vulnerability-type one-hot features |
| `cve_embeddings.npy` | `07_similarity_search.ipynb` | ~2 MB | Sentence-BERT embeddings (1 314 × 384 float32) |

## Models downloaded on first run (HuggingFace Hub)

These are downloaded automatically when the Streamlit app first calls the relevant
loader function.  They are cached in `~/.cache/huggingface/` and are **not** stored
inside this repository.

| Model | Used for | Approx size |
|---|---|---|
| `facebook/bart-large-cnn` | Text summarisation (Page 1 — AI Summary) | ~1.6 GB |
| `facebook/bart-large-mnli` | Zero-shot vulnerability classification (fallback) | ~1.6 GB |
| `sentence-transformers/all-MiniLM-L6-v2` | Semantic similarity search (Page 3) | ~90 MB |

**Total first-run download:** ~3.3 GB (models cached; only downloaded once)

## Running without downloading heavy models

All three HuggingFace models have graceful fallbacks:

- **Summariser** — Falls back to extractive (longest sentence) summarisation.
- **Zero-shot classifier** — Falls back to regex pattern matching from `utils/preprocessing.py`.
- **Sentence encoder** — Falls back to TF-IDF cosine similarity on-the-fly.

The app will show a warning in the sidebar instead of crashing.

## Regenerating artefacts

Run the notebooks in order to regenerate all local artefacts:

```bash
jupyter nbconvert --to notebook --execute notebooks/03_tfidf_keyword_extraction.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_vulnerability_classification_bert.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_severity_prediction.ipynb
jupyter nbconvert --to notebook --execute notebooks/07_similarity_search.ipynb
```
