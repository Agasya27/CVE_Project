"""
CVE Intelligence Analyzer - Streamlit Dashboard

Interactive dashboard for analyzing CVE vulnerability descriptions.
Features:
  1. CVE Analyzer: Classify, predict severity, extract keywords, summarize
  2. Vulnerability Trends: Charts and analytics
  3. Similar CVE Search: Semantic similarity search
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import os
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocessing import (
    preprocess_text, classify_vulnerability_type,
    get_severity_label
)
from utils.model_utils import generate_alert, format_alert_text

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CVE Intelligence Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Targets all background layers including the white header strip */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main, [data-testid="stHeader"] {
        background-color: #F7F6FF !important;
    }

    /* Removes the white background from the content block specifically */
    [data-testid="stAppViewBlockContainer"] {
        background-color: #F7F6FF !important;
        padding-top: 2rem !important; /* Adjusts top spacing */
    }

    html, body { 
        font-family: 'Inter', sans-serif;
        background-color: #F7F6FF !important; 
    }

         

    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px; }
            

    /* ── 4. Black Border Chart Cards (Replacement) ── */
    /* This targets the internal Streamlit block that holds your chart */
    [data-testid="stVerticalBlock"] > div:has(div.stPlotlyChart) {
        background-color: #ffffff !important;
        border: 2px solid #000000 !important; /* Pure Black Border */
        border-radius: 12px !important;
        padding: 20px !important;
        margin-bottom: 25px !important;
        box-shadow: 4px 4px 0px rgba(0,0,0,0.05) !important;
    }

    /* Styles the title inside the black-bordered box */
    [data-testid="stVerticalBlock"] > div:has(div.stPlotlyChart) .section-title {
        margin-top: 0 !important;
        color: #000000 !important;
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 8px;
        margin-bottom: 15px !important;
    }




    

    /* ── Hero header ────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.25);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.8;
        margin: 0.4rem 0 0;
    }

    /* ── Section headings ───────────────────────────────── */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #302b63;
        margin: 1.2rem 0 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    /* ── Cards ──────────────────────────────────────────── */
    .glass-card {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(230,230,250,0.6);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.10);
    }

    /* ── Metric pills ───────────────────────────────────── */
    .metric-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 0.8rem; }
    .metric-pill {
        flex: 1;
        min-width: 130px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(102,126,234,0.25);
    }
    .metric-pill .mp-value { font-size: 1.6rem; font-weight: 800; line-height: 1.2; }
    .metric-pill .mp-label { font-size: 0.75rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.6px; }

    /* alternate colors */
    .metric-pill.orange { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
    .metric-pill.red    { background: linear-gradient(135deg, #e53935 0%, #ef5350 100%); }
    .metric-pill.green  { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }

    /* ── Severity badges ────────────────────────────────── */
    .severity-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }
    .severity-critical { background: #ffcdd2; color: #b71c1c; }
    .severity-high     { background: #ffe0b2; color: #e65100; }
    .severity-medium   { background: #fff9c4; color: #f57f17; }
    .severity-low      { background: #c8e6c9; color: #1b5e20; }

    /* ── Keyword chips ──────────────────────────────────── */
    .chip-container { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    .chip {
        background: #ede7f6;
        color: #4527a0;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #d1c4e9;
        transition: background 0.15s;
    }
    .chip:hover { background: #d1c4e9; }

    /* ── Alert box ──────────────────────────────────────── */
    .alert-box {
        background: linear-gradient(135deg, #e8eaf6 0%, #f3e5f5 100%);
        border-left: 5px solid #5c6bc0;
        border-radius: 0 12px 12px 0;
        padding: 1.2rem 1.5rem;
        font-size: 0.92rem;
        line-height: 1.65;
        white-space: pre-wrap;
    }

    /* ── Similarity result cards ─────────────────────────── */
    .sim-card {
        background: #fafafe;
        border: 1px solid #e8eaf6;
        border-radius: 12px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.7rem;
        transition: border-color 0.15s;
    }
    .sim-card:hover { border-color: #7c4dff; }
    .sim-card .sim-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .sim-card .sim-cve {
        font-weight: 700;
        font-size: 1rem;
        color: #302b63;
    }
    .sim-card .sim-score {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .sim-card .sim-tags { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.45rem; }
    .sim-card .sim-tag {
        font-size: 0.72rem;
        padding: 0.15rem 0.55rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .sim-card .sim-desc { font-size: 0.85rem; color: #555; line-height: 1.55; }

    /* ── Sidebar styling (light mode) ─────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f0f8 0%, #e8eaf6 100%);
    }
    section[data-testid="stSidebar"] hr { border-color: rgba(0,0,0,0.08); }

    /* ── Model status dots ──────────────────────────────── */
    .status-row { display: flex; align-items: center; gap: 0.45rem; margin: 0.25rem 0; font-size: 0.82rem; }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%; display: inline-block;
    }
    .status-dot.green  { background: #4caf50; box-shadow: 0 0 6px #4caf50; }
    .status-dot.red    { background: #ef5350; }
    .status-row { color: #333 !important; }

    /* ── Streamlit overrides ────────────────────────────── */
    div[data-testid="stMetric"] {
        background: #fafafe;
        border: 1px solid #ede7f6;
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
        transition: border-color 0.2s;
    }
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.15) !important;
    }

    /* ── The Nuclear Neon Button Fix ── */
    body .stApp div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%) !important;
        background-color: #26de81 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        height: 3rem !important;
        box-shadow: 0 4px 15px rgba(38, 222, 129, 0.4) !important;
        text-transform: uppercase !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Hover State Force */
    body .stApp div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2ae386 0%, #25c975 100%) !important;
        box-shadow: 0 6px 20px rgba(38, 222, 129, 0.6) !important;
        transform: translateY(-1px) !important;
    }



      

    
    /* ── Footer Styling ── */
    .app-footer {
        text-align: center; /* This centers the text */
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e0e0e0;
        color: #666;
        font-size: 0.85rem;
        line-height: 1.6;
        width: 100%; /* Ensures it spans the full width to allow centering */
    }     
    
</style>
""", unsafe_allow_html=True)

# ─── Model Loading (cached) ──────────────────────────────────────────────────

MODELS_DIR = (PROJECT_ROOT / "models").resolve()
DATA_DIR = (PROJECT_ROOT / "data").resolve()


def _bert_available():
    return (MODELS_DIR / "bert_classifier" / "model.safetensors").is_file()

def _severity_available():
    return (MODELS_DIR / "severity_predictor.joblib").is_file()

def _tfidf_available():
    return (MODELS_DIR / "tfidf_vectorizer.joblib").is_file()

def _embeddings_available():
    return (MODELS_DIR / "cve_embeddings.npy").is_file()


@st.cache_resource
def load_bert_classifier():
    """Load the BERT vulnerability classifier."""
    model_path = MODELS_DIR / "bert_classifier"
    le_path = MODELS_DIR / "label_encoder.joblib"
    if not model_path.is_dir() or not le_path.is_file():
        return None, None, None

    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    import torch

    tokenizer = DistilBertTokenizer.from_pretrained(str(model_path))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    label_encoder = joblib.load(le_path)
    return model, tokenizer, label_encoder


@st.cache_resource
def load_severity_predictor():
    """Load the severity prediction model."""
    paths = [
        MODELS_DIR / "severity_predictor.joblib",
        MODELS_DIR / "severity_tfidf.joblib",
        MODELS_DIR / "severity_encoder.joblib",
        MODELS_DIR / "severity_vuln_columns.joblib",
    ]
    if not all(p.is_file() for p in paths):
        return None, None, None, None

    return tuple(joblib.load(p) for p in paths)


@st.cache_resource
def load_summarizer():
    """Load the BART summarization model."""
    try:
        from transformers import pipeline
        summarizer = pipeline('summarization', model='facebook/bart-large-cnn', device=-1)
        return summarizer
    except Exception:
        return None


@st.cache_resource
def load_sbert():
    """Load the Sentence-BERT model for similarity search."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception:
        return None


@st.cache_data
def load_embeddings():
    """Load precomputed CVE embeddings."""
    emb_path = MODELS_DIR / "cve_embeddings.npy"
    if emb_path.is_file():
        return np.load(emb_path)
    return None


@st.cache_data
def load_tfidf_vectorizer():
    """Load TF-IDF vectorizer for keyword extraction."""
    tfidf_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    if tfidf_path.is_file():
        return joblib.load(tfidf_path)
    return None


@st.cache_data
def load_dataset():
    """Load the CVE dataset."""
    # Try loading the most processed version first
    for fname in ['cve_with_keywords.csv', 'cve_preprocessed.csv', 'cve_explored.csv', 'cve_dataset.csv']:
        path = DATA_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            # Ensure required columns
            if 'Severity' not in df.columns and 'CVSS Score' in df.columns:
                df['Severity'] = df['CVSS Score'].apply(get_severity_label)
            if 'Vulnerability_Type' not in df.columns and 'Description' in df.columns:
                df['Vulnerability_Type'] = df['Description'].apply(classify_vulnerability_type)
            if 'CVE_Year' not in df.columns and 'CVE ID' in df.columns:
                df['CVE_Year'] = df['CVE ID'].str.extract(r'CVE-(\d{4})-').astype(float)
            return df
    return pd.DataFrame()


# ─── Helper Functions ─────────────────────────────────────────────────────────

def predict_vuln_type_bert(text, model, tokenizer, label_encoder):
    """Classify vulnerability type using BERT."""
    import torch
    encoding = tokenizer(
        text, add_special_tokens=True, max_length=256,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(input_ids=encoding['input_ids'],
                       attention_mask=encoding['attention_mask'])
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    return label_encoder.inverse_transform([pred_idx])[0], confidence


def predict_severity_ml(text, vuln_type, model, tfidf, encoder, vuln_cols):
    """Predict severity using the ML model."""
    from scipy.sparse import hstack, csr_matrix
    cleaned = preprocess_text(text)
    X_tfidf = tfidf.transform([cleaned])

    # One-hot encode vulnerability type
    vuln_dict = {col: 0 for col in vuln_cols}
    col_name = f"vuln_{vuln_type}"
    if col_name in vuln_dict:
        vuln_dict[col_name] = 1
    vuln_array = csr_matrix([list(vuln_dict.values())])

    X = hstack([X_tfidf, vuln_array])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return encoder.inverse_transform([pred])[0], max(proba)


def summarize_text(text, summarizer):
    """Summarize a CVE description."""
    if len(text.strip()) < 50:
        return text
    try:
        result = summarizer(text, max_length=80, min_length=20,
                          do_sample=False, truncation=True)
        return result[0]['summary_text']
    except Exception:
        return text[:200] + "..."


def extract_keywords_tfidf(text, vectorizer, n=8):
    """Extract top keywords from text using TF-IDF."""
    cleaned = preprocess_text(text)
    tfidf_vec = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_vec.toarray().flatten()
    top_indices = scores.argsort()[::-1][:n]
    return [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]


def find_similar_cves(query, embeddings, df, sbert_model, top_n=5):
    """Find similar CVEs using Sentence-BERT."""
    query_emb = sbert_model.encode([query])
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_emb, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    results = []
    for idx in top_indices:
        results.append({
            'CVE ID': df['CVE ID'].iloc[idx],
            'Description': df['Description'].iloc[idx][:200] + "...",
            'CVSS Score': df['CVSS Score'].iloc[idx],
            'Severity': df.get('Severity', pd.Series(['N/A'])).iloc[idx] if 'Severity' in df.columns else 'N/A',
            'Type': df.get('Vulnerability_Type', pd.Series(['N/A'])).iloc[idx] if 'Vulnerability_Type' in df.columns else 'N/A',
            'Similarity': f"{similarities[idx]:.4f}"
        })
    return pd.DataFrame(results)


# ─── Fallback Helpers (for Streamlit Cloud / no-model deployment) ─────────────

def _rule_based_severity(text):
    """Estimate severity from keywords when ML model is unavailable."""
    t = text.lower()
    critical_kw = ['remote code execution', 'arbitrary code', 'unauthenticated', 'rce', 'root access', 'pre-auth']
    high_kw = ['privilege escalation', 'authentication bypass', 'command injection', 'sql injection', 'buffer overflow']
    low_kw = ['low impact', 'minor', 'informational']
    if any(k in t for k in critical_kw):
        return 'Critical'
    if any(k in t for k in high_kw):
        return 'High'
    if any(k in t for k in low_kw):
        return 'Low'
    return 'Medium'


def _extract_keywords_simple(text, n=8):
    """Extract keywords using basic TF approach when TF-IDF model is unavailable."""
    import re
    from collections import Counter
    stop = {'the','a','an','is','are','was','were','be','been','being','have','has',
            'had','do','does','did','will','would','could','should','may','might',
            'shall','can','to','of','in','for','on','with','at','by','from','as',
            'into','through','during','before','after','above','below','between',
            'out','off','over','under','again','further','then','once','that','this',
            'these','those','it','its','and','but','or','nor','not','so','very','also',
            'just','than','too','only','own','same','no','all','each','every','both',
            'few','more','most','other','some','such','any','up','about','which','when',
            'where','who','whom','what','how','if','because','until','while','although',
            'allows','via','using','allows','used','use','due'}
    words = re.findall(r'[a-z]{3,}', text.lower())
    words = [w for w in words if w not in stop]
    counts = Counter(words)
    total = max(len(words), 1)
    return [(w, c / total) for w, c in counts.most_common(n)]


def _extractive_summary(text, max_len=200):
    """Simple extractive summary by picking the most informative sentence."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return text[:max_len]
    # Pick the longest sentence (usually most informative) up to max_len
    best = max(sentences, key=len)
    if len(best) > max_len:
        best = best[:max_len].rsplit(' ', 1)[0] + '…'
    return best


def _find_similar_tfidf(query, df, top_n=5):
    """TF-IDF-based similarity search fallback when SBERT is unavailable."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    descriptions = df['Description'].fillna('').tolist()
    corpus = [query] + descriptions
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    results = []
    for idx in top_indices:
        results.append({
            'CVE ID': df['CVE ID'].iloc[idx],
            'Description': df['Description'].iloc[idx][:200] + "...",
            'CVSS Score': df['CVSS Score'].iloc[idx],
            'Severity': df['Severity'].iloc[idx] if 'Severity' in df.columns else 'N/A',
            'Type': df['Vulnerability_Type'].iloc[idx] if 'Vulnerability_Type' in df.columns else 'N/A',
            'Similarity': f"{similarities[idx]:.4f}"
        })
    return pd.DataFrame(results)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 🛡️ CVE Intel Analyzer")
st.sidebar.caption("NLP-powered vulnerability intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🔍 CVE Analyzer",
    "📊 Vulnerability Trends",
    "🔎 Similar CVE Search"
], label_visibility="collapsed")

# Model status indicators
st.sidebar.markdown("---")
st.sidebar.markdown("#### Model Status")

_bert_ready = _bert_available()
_sev_ready = _severity_available()
_tfidf_ready = _tfidf_available()
_emb_ready = _embeddings_available()

for name, ready in [
    ("BERT Classifier", _bert_ready),
    ("Severity Predictor", _sev_ready),
    ("TF-IDF Keywords", _tfidf_ready),
    ("CVE Embeddings", _emb_ready),
]:
    dot_cls = "green" if ready else "red"
    label = "Ready" if ready else "Will use fallback"
    st.sidebar.markdown(
        f'<div class="status-row"><span class="status-dot {dot_cls}"></span>{name} — <em>{label}</em></div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · HuggingFace · scikit-learn")

# Load dataset
df = load_dataset()

# ─── Page 1: CVE Analyzer ────────────────────────────────────────────────────

if page == "🔍 CVE Analyzer":
    st.markdown(
        '<div class="main-header">'
        '<h1>🛡️ CVE Insight: NLP-Powered Short Alerts for Vulnerability Management</h1>'
        '<p>Paste a vulnerability description to get instant classification, severity, keywords &amp; an actionable alert.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Input
    st.markdown('<div class="section-title"> Input CVE Description</div>', unsafe_allow_html=True)
    default_text = (
        "A buffer overflow vulnerability exists in Apache HTTP Server 2.4.58 "
        "that allows remote attackers to execute arbitrary code via crafted HTTP requests."
    )
    cve_text = st.text_area("CVE Description", value=default_text, height=120, label_visibility="collapsed")

    if st.button("Analyze Vulnerability", type="primary", use_container_width=True):
        if not cve_text.strip():
            st.warning("Please enter a CVE description.")
        else:
            with st.spinner("Running NLP pipeline…"):
                # ── Classification & Severity side-by-side ─────────────
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<div class="section-title">🏷️ Vulnerability Type</div>', unsafe_allow_html=True)
                    bert_model, bert_tokenizer, bert_le = load_bert_classifier()
                    if bert_model is not None:
                        vuln_type, confidence = predict_vuln_type_bert(
                            cve_text, bert_model, bert_tokenizer, bert_le
                        )
                        st.markdown(
                            f'<div class="glass-card">'
                            f'<div style="font-size:1.35rem;font-weight:700;color:#302b63">{vuln_type}</div>'
                            f'<div style="margin-top:0.4rem;font-size:0.85rem;color:#777">'
                            f'BERT confidence: <strong>{confidence:.1%}</strong></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        vuln_type = classify_vulnerability_type(cve_text)
                        st.markdown(
                            f'<div class="glass-card">'
                            f'<div style="font-size:1.35rem;font-weight:700;color:#302b63">{vuln_type}</div>'
                            f'<div style="margin-top:0.4rem;font-size:0.85rem;color:#999">Rule-based (BERT not trained)</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                with col2:
                    st.markdown('<div class="section-title">⚠️ Predicted Severity</div>', unsafe_allow_html=True)
                    sev_model, sev_tfidf, sev_encoder, sev_vuln_cols = load_severity_predictor()
                    if sev_model is not None:
                        severity, sev_conf = predict_severity_ml(
                            cve_text, vuln_type, sev_model, sev_tfidf, sev_encoder, sev_vuln_cols
                        )
                        sev_class = f"severity-{severity.lower()}"
                        st.markdown(
                            f'<div class="glass-card">'
                            f'<span class="severity-badge {sev_class}">{severity}</span>'
                            f'<div style="margin-top:0.5rem;font-size:0.85rem;color:#777">'
                            f'Confidence: <strong>{sev_conf:.1%}</strong></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        # Fallback: rule-based severity from keywords
                        severity = _rule_based_severity(cve_text)
                        sev_class = f"severity-{severity.lower()}"
                        st.markdown(
                            f'<div class="glass-card">'
                            f'<span class="severity-badge {sev_class}">{severity}</span>'
                            f'<div style="margin-top:0.4rem;font-size:0.85rem;color:#999">Rule-based estimate</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # ── Keywords ───────────────────────────────────────────
                st.markdown('<div class="section-title">🔑 Extracted Keywords</div>', unsafe_allow_html=True)
                tfidf_vec = load_tfidf_vectorizer()
                if tfidf_vec is not None:
                    keywords = extract_keywords_tfidf(cve_text, tfidf_vec)
                else:
                    keywords = _extract_keywords_simple(cve_text)
                if keywords:
                    chips_html = "".join(
                        f'<span class="chip">{kw} ({score:.2f})</span>' for kw, score in keywords[:10]
                    )
                    st.markdown(f'<div class="chip-container">{chips_html}</div>', unsafe_allow_html=True)
                else:
                    st.info("No significant keywords extracted.")

                # ── Generated Alert ────────────────────────────────────
                st.markdown('<div class="section-title">📋 Security Alert</div>', unsafe_allow_html=True)
                summarizer = load_summarizer()
                severity_display = severity
                if summarizer is not None:
                    summary = summarize_text(cve_text, summarizer)
                else:
                    summary = _extractive_summary(cve_text)

                alert = generate_alert(cve_text, vuln_type, severity_display, summary)
                st.markdown(
                    f'<div class="alert-box">{format_alert_text(alert)}</div>',
                    unsafe_allow_html=True,
                )


# ─── Page 2: Vulnerability Trends ────────────────────────────────────────────

elif page == "📊 Vulnerability Trends":
    st.markdown(
        '<div class="main-header">'
        '<h1>📊 Vulnerability Trends</h1>'
        '<p>Explore distribution patterns in the CVE dataset.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if df.empty:
        st.error("Dataset not found. Ensure data/cve_dataset.csv exists.")
    else:
        # ── KPI metric pills ──────────────────────────────────
        total = len(df)
        avg_cvss = df['CVSS Score'].mean()
        critical_n = int((df['Severity'] == 'Critical').sum()) if 'Severity' in df.columns else 0
        high_n = int((df['Severity'] == 'High').sum()) if 'Severity' in df.columns else 0
        st.markdown(
            '<div class="metric-row">'
            f'<div class="metric-pill"><div class="mp-value">{total:,}</div><div class="mp-label">Total CVEs</div></div>'
            f'<div class="metric-pill orange"><div class="mp-value">{avg_cvss:.1f}</div><div class="mp-label">Avg CVSS</div></div>'
            f'<div class="metric-pill red"><div class="mp-value">{critical_n}</div><div class="mp-label">Critical</div></div>'
            f'<div class="metric-pill green"><div class="mp-value">{high_n}</div><div class="mp-label">High</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Row 1: Two charts
        chart_col1, chart_col2 = st.columns(2)

        _chart_template = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif'),
            margin=dict(l=20, r=10, t=20, b=20),
        )

        with chart_col1:
           
            st.markdown('<div class="section-title">Severity Breakdown</div>', unsafe_allow_html=True)
            if 'Severity' in df.columns:
                sev_counts = df['Severity'].value_counts()
                severity_order = ['Critical', 'High', 'Medium', 'Low', 'None', 'Unknown']
                sev_counts = sev_counts.reindex([s for s in severity_order if s in sev_counts.index])
                color_map = {'Critical': '#e53935', 'High': '#fb8c00', 'Medium': '#fdd835',
                            'Low': '#43a047', 'None': '#90a4ae', 'Unknown': '#bdbdbd'}
                fig = px.pie(values=sev_counts.values, names=sev_counts.index,
                           color=sev_counts.index, color_discrete_map=color_map,
                           hole=0.45)
                fig.update_traces(textinfo='percent+label', textfont_size=12)
                fig.update_layout(height=450, showlegend=False, **_chart_template)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_col2:
           
            st.markdown('<div class="section-title">CVEs by Year</div>', unsafe_allow_html=True)
            if 'CVE_Year' in df.columns:
                year_counts = df['CVE_Year'].dropna().value_counts().sort_index()
                fig = px.bar(x=year_counts.index.astype(int), y=year_counts.values,
                           labels={'x': 'Year', 'y': 'Count'},
                           color=year_counts.values,
                           color_continuous_scale=[[0, "#bdc4f4"], [1, "#100d2f"]])
                fig.update_layout(height=450, showlegend=False,
                                  coloraxis_showscale=False, **_chart_template)
                fig.update_traces(marker_line_width=0, marker_cornerradius=6)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Row 2: Vulnerability types
       
        st.markdown('<div class="section-title">Vulnerability Types</div>', unsafe_allow_html=True)
        if 'Vulnerability_Type' in df.columns:
            vuln_counts = df['Vulnerability_Type'].value_counts()
            fig = px.bar(x=vuln_counts.values, y=vuln_counts.index,
                        labels={'x': 'Count', 'y': ''},
                        color=vuln_counts.values,
                        color_continuous_scale=[[0, '#e8eaf6'], [1, '#302b63']],
                        orientation='h')
            fig.update_layout(height=400, showlegend=False,
                              coloraxis_showscale=False,
                              yaxis=dict(autorange='reversed'),
                              **_chart_template)
            fig.update_traces(marker_line_width=0, marker_cornerradius=5)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Row 3: Most affected platforms
        
        st.markdown('<div class="section-title">Most Affected Vendors / Platforms</div>', unsafe_allow_html=True)
        if 'Affected OS' in df.columns:
            from collections import Counter
            os_data = df[df['Affected OS'] != 'N/A']['Affected OS'].dropna()
            all_vendors = []
            for os_str in os_data:
                for os_name in str(os_str).split(','):
                    parts = os_name.strip().split()
                    if parts:
                        all_vendors.append(parts[0])
            vendor_counts = Counter(all_vendors).most_common(15)
            if vendor_counts:
                fig = px.bar(x=[v[0] for v in vendor_counts],
                           y=[v[1] for v in vendor_counts],
                           labels={'x': 'Vendor', 'y': 'Count'},
                           color=[v[1] for v in vendor_counts],
                           color_continuous_scale=[[0, "#b1a5ed"], [1, "#2A144A"]])
                fig.update_layout(xaxis_tickangle=-45, height=380, showlegend=False,
                                  coloraxis_showscale=False, **_chart_template)
                fig.update_traces(marker_line_width=0, marker_cornerradius=5)
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        # CVSS Score distribution
       
        st.markdown('<div class="section-title">CVSS Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='CVSS Score', nbins=25,
                         color_discrete_sequence=["#6A378E"])
        fig.update_layout(height=300, bargap=0.06, **_chart_template)
        fig.update_traces(marker_line_width=0, marker_cornerradius=4)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Page 3: Similar CVE Search ──────────────────────────────────────────────

elif page == "🔎 Similar CVE Search":
    st.markdown(
        '<div class="main-header">'
        '<h1>🔎 Similar CVE Search</h1>'
        '<p>Find vulnerabilities with similar attack patterns using semantic embeddings.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Describe a vulnerability</div>', unsafe_allow_html=True)
    query = st.text_area(
        "Query",
        value="SQL injection vulnerability in login page allowing authentication bypass",
        height=100,
        label_visibility="collapsed",
    )

    num_results = st.slider("Number of results", 3, 15, 5)

    if st.button("Find Similar CVEs", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a vulnerability description.")
        else:
            sbert_model = load_sbert()
            embeddings = load_embeddings()

            # Try SBERT first, fall back to TF-IDF similarity
            use_tfidf_fallback = False
            if sbert_model is not None:
                if embeddings is None:
                    with st.spinner("Generating embeddings on-the-fly…"):
                        descriptions = df['Description'].fillna('').tolist()
                        embeddings = sbert_model.encode(descriptions, show_progress_bar=False, batch_size=32)
            else:
                use_tfidf_fallback = True

            if not use_tfidf_fallback and sbert_model is not None and embeddings is not None:
                with st.spinner("Searching…"):
                    results = find_similar_cves(query, embeddings, df, sbert_model, top_n=num_results)
            else:
                with st.spinner("Searching (TF-IDF similarity)…"):
                    results = _find_similar_tfidf(query, df, top_n=num_results)

            st.markdown(f'<div class="section-title">Top {len(results)} Matches</div>', unsafe_allow_html=True)

            for _, row in results.iterrows():
                severity = row.get('Severity', 'N/A')
                sev_cls = f"severity-{severity.lower()}" if severity in ('Critical','High','Medium','Low') else ""
                sev_badge = f'<span class="sim-tag {sev_cls}" style="margin-right:2px">{severity}</span>'
                type_badge = (
                    f'<span class="sim-tag" style="background:#e8eaf6;color:#302b63">{row["Type"]}</span>'
                )
                cvss_badge = (
                    f'<span class="sim-tag" style="background:#fff3e0;color:#e65100">CVSS {row["CVSS Score"]}</span>'
                )

                st.markdown(
                    f'<div class="sim-card">'
                    f'  <div class="sim-header">'
                    f'    <span class="sim-cve">{row["CVE ID"]}</span>'
                    f'    <span class="sim-score">Similarity {row["Similarity"]}</span>'
                    f'  </div>'
                    f'  <div class="sim-tags">{sev_badge}{type_badge}{cvss_badge}</div>'
                    f'  <div class="sim-desc">{row["Description"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown('''
    <div class="app-footer">
        🛡️ CVE Intelligence Analyzer Dashboard • Built with Streamlit & NLP Tools<br>
        © 2026  | NLP models may not be 100% accurate. Always verify with official CVE sources. | Data last updated: 2024 </div>
''', unsafe_allow_html=True)
