"""
CVE Intelligence Analyzer — Streamlit Dashboard
================================================

Four pages:
  1. CVE Analyzer          — classify, predict severity, extract keywords, summarize
  2. Vulnerability Dashboard — charts, metrics, and searchable table
  3. Similar CVE Search    — semantic / TF-IDF similarity search
  4. Attack Surface Scanner — paste your tech stack, get a risk-scored patch queue
                              and an executive security-posture report

Run:
    streamlit run app/streamlit_app.py
"""

import re
import sys
import time
import joblib
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocessing import (
    classify_vulnerability_type,
    extract_keywords,
    get_severity_label,
    preprocess_text,
)
from utils.model_utils import (
    format_alert_text,
    generate_alert,
    generate_recommended_action,
    summarize_cve,
    classify_vulnerability,
    predict_severity,
    find_similar_cves,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CVE Intelligence Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = (PROJECT_ROOT / "models").resolve()
DATA_DIR   = (PROJECT_ROOT / "data").resolve()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px; }

  /* Hero */
  .main-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #fff; padding: 2rem 2.5rem; border-radius: 16px;
    margin-bottom: 1.5rem; text-align: center;
    box-shadow: 0 8px 32px rgba(48,43,99,0.25);
  }
  .main-header h1 { font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
  .main-header p  { font-size: 1rem; opacity: .8; margin: .4rem 0 0; }

  /* Section titles */
  .section-title {
    font-size: 1.05rem; font-weight: 700; color: #302b63;
    margin: 1.2rem 0 .6rem; display: flex; align-items: center; gap: .45rem;
  }

  /* Cards */
  .glass-card {
    background: rgba(255,255,255,.75); backdrop-filter: blur(12px);
    border: 1px solid rgba(230,230,250,.6); border-radius: 14px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,.06);
    transition: transform .15s ease, box-shadow .15s ease;
  }
  .glass-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,.10); }

  /* Metric pills */
  .metric-row { display: flex; gap: .75rem; flex-wrap: wrap; margin-bottom: .8rem; }
  .metric-pill {
    flex: 1; min-width: 130px;
    background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    color: #fff; padding: 1rem 1.2rem; border-radius: 12px; text-align: center;
    box-shadow: 0 4px 16px rgba(102,126,234,.25);
  }
  .metric-pill .mp-value { font-size: 1.6rem; font-weight: 800; line-height: 1.2; }
  .metric-pill .mp-label { font-size: .75rem; opacity: .85; text-transform: uppercase; letter-spacing: .6px; }
  .metric-pill.orange { background: linear-gradient(135deg,#f7971e 0%,#ffd200 100%); }
  .metric-pill.red    { background: linear-gradient(135deg,#e53935 0%,#ef5350 100%); }
  .metric-pill.green  { background: linear-gradient(135deg,#11998e 0%,#38ef7d 100%); }

  /* Severity badges */
  .severity-badge { display: inline-block; padding: .35rem 1rem; border-radius: 20px; font-weight: 700; font-size: .95rem; letter-spacing: .3px; }
  .severity-critical { background:#ffcdd2; color:#b71c1c; }
  .severity-high     { background:#ffe0b2; color:#e65100; }
  .severity-medium   { background:#fff9c4; color:#f57f17; }
  .severity-low      { background:#c8e6c9; color:#1b5e20; }

  /* Chips */
  .chip-container { display: flex; flex-wrap: wrap; gap: .5rem; }
  .chip {
    background:#ede7f6; color:#4527a0; padding:.3rem .85rem; border-radius:20px;
    font-size:.82rem; font-weight:600; border:1px solid #d1c4e9;
  }

  /* Alert box */
  .alert-box {
    background: linear-gradient(135deg,#e8eaf6 0%,#f3e5f5 100%);
    border-left: 5px solid #5c6bc0; border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem; font-size: .92rem; line-height: 1.65; white-space: pre-wrap;
  }

  /* Similarity cards */
  .sim-card {
    background:#fafafe; border:1px solid #e8eaf6; border-radius:12px;
    padding:1rem 1.3rem; margin-bottom:.7rem; transition:border-color .15s;
  }
  .sim-card:hover { border-color:#7c4dff; }
  .sim-card .sim-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:.5rem; }
  .sim-card .sim-cve   { font-weight:700; font-size:1rem; color:#302b63; }
  .sim-card .sim-score {
    background:linear-gradient(135deg,#667eea,#764ba2); color:#fff;
    padding:.2rem .7rem; border-radius:12px; font-size:.78rem; font-weight:700;
  }
  .sim-card .sim-tags  { display:flex; gap:.4rem; flex-wrap:wrap; margin-bottom:.45rem; }
  .sim-card .sim-tag   { font-size:.72rem; padding:.15rem .55rem; border-radius:8px; font-weight:600; }
  .sim-card .sim-desc  { font-size:.85rem; color:#555; line-height:1.55; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background:linear-gradient(180deg,#f0f0f8 0%,#e8eaf6 100%); }
  section[data-testid="stSidebar"] hr { border-color:rgba(0,0,0,.08); }

  /* Model status dots */
  .status-row { display:flex; align-items:center; gap:.45rem; margin:.25rem 0; font-size:.82rem; color:#333 !important; }
  .status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
  .status-dot.green { background:#4caf50; box-shadow:0 0 6px #4caf50; }
  .status-dot.red   { background:#ef5350; }

  /* Streamlit overrides */
  div[data-testid="stMetric"] { background:#fafafe; border:1px solid #ede7f6; border-radius:12px; padding:.8rem 1rem; }
  .stTextArea textarea { border-radius:12px !important; border:2px solid #e0e0e0 !important; }
  .stTextArea textarea:focus { border-color:#667eea !important; box-shadow:0 0 0 3px rgba(102,126,234,.15) !important; }
  button[kind="primary"] {
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important;
    border:none !important; border-radius:10px !important; font-weight:600 !important;
    box-shadow:0 4px 16px rgba(102,126,234,.3) !important;
  }
  button[kind="primary"]:hover { transform:translateY(-1px) !important; box-shadow:0 6px 24px rgba(102,126,234,.4) !important; }
  div.stSpinner > div { color:#667eea !important; }
</style>
""", unsafe_allow_html=True)


# ── Model availability checks ─────────────────────────────────────────────────

def _bert_available() -> bool:
    """Return True if the fine-tuned DistilBERT classifier is present on disk."""
    return (MODELS_DIR / "bert_classifier" / "model.safetensors").is_file()


def _severity_model_available() -> bool:
    """Return True if all four severity-model artefacts are present."""
    return all(
        (MODELS_DIR / f).is_file()
        for f in [
            "severity_predictor.joblib",
            "severity_tfidf.joblib",
            "severity_encoder.joblib",
            "severity_vuln_columns.joblib",
        ]
    )


def _tfidf_available() -> bool:
    """Return True if the TF-IDF vectorizer is present."""
    return (MODELS_DIR / "tfidf_vectorizer.joblib").is_file()


def _embeddings_available() -> bool:
    """Return True if precomputed CVE embeddings are present."""
    return (MODELS_DIR / "cve_embeddings.npy").is_file()


# ── Cached model loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading BERT classifier…")
def load_bert_classifier():
    """
    Load the fine-tuned DistilBERT vulnerability classifier.

    Returns ``(model, tokenizer, label_encoder)`` or ``(None, None, None)``
    if the model artefacts are not found.
    """
    model_path = MODELS_DIR / "bert_classifier"
    le_path    = MODELS_DIR / "label_encoder.joblib"
    if not model_path.is_dir() or not le_path.is_file():
        return None, None, None
    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        import torch
        tokenizer     = DistilBertTokenizer.from_pretrained(str(model_path))
        model         = DistilBertForSequenceClassification.from_pretrained(str(model_path))
        model.eval()
        label_encoder = joblib.load(le_path)
        return model, tokenizer, label_encoder
    except Exception as exc:
        st.warning(f"BERT classifier could not load: {exc}")
        return None, None, None


@st.cache_resource(show_spinner="Loading severity predictor…")
def load_severity_predictor():
    """
    Load the severity prediction model artefacts.

    Returns ``(model, tfidf, encoder, vuln_cols)`` or ``(None,…)`` on failure.
    """
    paths = [
        MODELS_DIR / "severity_predictor.joblib",
        MODELS_DIR / "severity_tfidf.joblib",
        MODELS_DIR / "severity_encoder.joblib",
        MODELS_DIR / "severity_vuln_columns.joblib",
    ]
    if not all(p.is_file() for p in paths):
        return None, None, None, None
    try:
        return tuple(joblib.load(p) for p in paths)
    except Exception as exc:
        st.warning(f"Severity model could not load: {exc}")
        return None, None, None, None


@st.cache_resource(show_spinner="Loading summarizer (first run downloads ~1.6 GB)…")
def _load_summarizer_cached():
    """
    Cached wrapper around :func:`utils.model_utils.load_summarizer`.

    Downloads ``facebook/bart-large-cnn`` on first call (~1.6 GB).
    Returns the pipeline or ``None`` on failure.
    """
    from utils.model_utils import load_summarizer
    return load_summarizer()


@st.cache_resource(show_spinner="Loading sentence encoder…")
def _load_sbert_cached():
    """
    Cached wrapper around :func:`utils.model_utils.load_sentence_encoder`.

    Downloads ``all-MiniLM-L6-v2`` on first call (~90 MB).
    Returns the model or ``None`` on failure.
    """
    from utils.model_utils import load_sentence_encoder
    return load_sentence_encoder()


@st.cache_data(show_spinner="Loading TF-IDF vectorizer…")
def load_tfidf_vectorizer():
    """Load and cache the saved TF-IDF vectorizer, or return ``None``."""
    p = MODELS_DIR / "tfidf_vectorizer.joblib"
    if p.is_file():
        try:
            return joblib.load(p)
        except Exception:
            pass
    return None


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset() -> pd.DataFrame:
    """
    Load the CVE dataset.

    Tries cleaned → with-keywords → preprocessed → raw versions in order.
    Derives ``Severity``, ``Vulnerability_Type``, and ``CVE_Year`` columns
    if absent.

    Returns:
        Loaded ``DataFrame``, or an empty ``DataFrame`` on failure.
    """
    for fname in [
        "cve_cleaned.csv",
        "cve_with_keywords.csv",
        "cve_preprocessed.csv",
        "cve_explored.csv",
        "cve_dataset.csv",
    ]:
        path = DATA_DIR / fname
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "Severity" not in df.columns and "CVSS Score" in df.columns:
                    df["Severity"] = df["CVSS Score"].apply(get_severity_label)
                if "Vulnerability_Type" not in df.columns and "Description" in df.columns:
                    df["Vulnerability_Type"] = df["Description"].apply(
                        classify_vulnerability_type
                    )
                if "CVE_Year" not in df.columns and "CVE ID" in df.columns:
                    df["CVE_Year"] = (
                        df["CVE ID"].str.extract(r"CVE-(\d{4})-").astype(float).squeeze()
                    )
                return df
            except Exception:
                continue
    return pd.DataFrame()


# ── BERT inference helper ─────────────────────────────────────────────────────

def _bert_predict(text: str, model, tokenizer, label_encoder) -> tuple:
    """
    Run inference with the fine-tuned DistilBERT classifier.

    Args:
        text:          CVE description.
        model:         Loaded DistilBERT model.
        tokenizer:     Matching tokenizer.
        label_encoder: Fitted LabelEncoder.

    Returns:
        ``(predicted_label, confidence)`` tuple.
    """
    import torch
    enc = tokenizer(
        text,
        add_special_tokens=True,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        probs    = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        conf     = probs[0][pred_idx].item()
    return label_encoder.inverse_transform([pred_idx])[0], conf


# ── Shared chart template ─────────────────────────────────────────────────────
_CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=45, b=40),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛡️ CVE Intel Analyzer")
st.sidebar.caption("NLP-powered vulnerability intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🔍 CVE Analyzer",
        "📊 Vulnerability Dashboard",
        "🔎 Similar CVE Search",
        "🎯 Attack Surface Scanner",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Model Status")

_bert_rdy  = _bert_available()
_sev_rdy   = _severity_model_available()
_tfidf_rdy = _tfidf_available()
_emb_rdy   = _embeddings_available()

for name, ready in [
    ("BERT Classifier",    _bert_rdy),
    ("Severity Predictor", _sev_rdy),
    ("TF-IDF Keywords",    _tfidf_rdy),
    ("CVE Embeddings",     _emb_rdy),
]:
    dot   = "green" if ready else "red"
    label = "Ready" if ready else "Fallback"
    st.sidebar.markdown(
        f'<div class="status-row">'
        f'<span class="status-dot {dot}"></span>'
        f'{name} — <em>{label}</em></div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · HuggingFace · scikit-learn")

# Pre-load dataset (used on multiple pages)
df = load_dataset()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CVE Analyzer
# ══════════════════════════════════════════════════════════════════════════════

# ── Attack Surface Scanner helpers ─────────────────────────────────────────

def _parse_cvss_vector(vector: str) -> dict:
    """
    Parse a CVSS 3.x vector string into a dict of metric → value.

    Example: ``'CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N'``
    Returns: ``{'AV': 'N', 'AC': 'L', 'PR': 'N', 'UI': 'R', ...}``
    """
    result = {}
    if not isinstance(vector, str):
        return result
    for part in vector.split("/"):
        if ":" in part:
            k, v = part.split(":", 1)
            result[k] = v
    return result


def _compute_risk_score(row: pd.Series) -> float:
    """
    Compute a contextual risk score (0–10) that goes beyond raw CVSS.

    Factors:
    - CVSS base score
    - Attack vector:   Network +2.0 | Adjacent +0.5 | Physical -1.0
    - Privileges req:  None +1.0 | Low +0.5
    - User interaction: None +0.5
    - Attack complexity: Low +0.5
    - Vulnerability-type multiplier (RCE ×1.5, SQLi ×1.3, …)
    """
    base = float(row.get("CVSS Score", 5.0))
    vec  = _parse_cvss_vector(str(row.get("Attack Vector", "")))

    av_bonus = {"N": 2.0, "A": 0.5, "L": 0.0, "P": -1.0}.get(vec.get("AV", "L"), 0.0)
    pr_bonus = {"N": 1.0, "L": 0.5, "H": 0.0}.get(vec.get("PR", "H"), 0.0)
    ui_bonus = {"N": 0.5, "R": 0.0}.get(vec.get("UI", "R"), 0.0)
    ac_bonus = {"L": 0.5, "H": 0.0}.get(vec.get("AC", "H"), 0.0)

    type_mult = {
        "Remote Code Execution":    1.5,
        "SQL Injection":            1.3,
        "Authentication Bypass":    1.3,
        "Command Injection":        1.2,
        "Privilege Escalation":     1.2,
        "Buffer Overflow":          1.2,
        "Information Disclosure":   1.0,
        "Path Traversal":           1.0,
        "SSRF":                     1.0,
        "Denial of Service":        0.9,
        "Cross-Site Scripting (XSS)": 0.9,
        "CSRF":                     0.8,
        "Other":                    1.0,
    }.get(str(row.get("Vulnerability_Type", "Other")), 1.0)

    raw = (base + av_bonus + pr_bonus + ui_bonus + ac_bonus) * type_mult
    return round(min(raw, 10.0), 2)


def _priority_label(score: float) -> str:
    """Map a risk score to a human-readable priority tier."""
    if score >= 9.0:  return "🔴 Patch Immediately"
    if score >= 7.0:  return "🟠 Patch within 24 h"
    if score >= 5.0:  return "🟡 Patch this week"
    return "🟢 Schedule maintenance"


def _scan_tech_stack(techs: list, df: pd.DataFrame, min_cvss: float) -> pd.DataFrame:
    """
    Find all CVEs in *df* whose description or Affected_Software mentions
    any technology in *techs*.

    Returns a deduplicated DataFrame with added columns:
    ``Matched_Technology``, ``Risk_Score``, ``Priority``.
    """
    frames = []
    for tech in techs:
        t = tech.strip()
        if not t:
            continue
        mask_desc = df["Description"].str.contains(t, case=False, na=False, regex=False)
        mask_sw   = (
            df["Affected_Software"].str.contains(t, case=False, na=False, regex=False)
            if "Affected_Software" in df.columns else pd.Series(False, index=df.index)
        )
        matched = df[mask_desc | mask_sw].copy()
        matched["Matched_Technology"] = t
        frames.append(matched)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    # Keep highest CVSS row per CVE (dedup across multiple tech matches)
    result = (
        result.sort_values("CVSS Score", ascending=False)
              .drop_duplicates(subset="CVE ID")
              .reset_index(drop=True)
    )
    result = result[result["CVSS Score"] >= min_cvss].copy()
    result["Risk_Score"] = result.apply(_compute_risk_score, axis=1)
    result["Priority"]   = result["Risk_Score"].apply(_priority_label)
    result = result.sort_values("Risk_Score", ascending=False).reset_index(drop=True)
    result.index += 1   # 1-based rank
    return result


def _executive_summary(scanned: pd.DataFrame, techs: list, timestamp: str) -> str:
    """
    Generate a professional executive security-posture report as plain text.
    """
    total      = len(scanned)
    crit_high  = int((scanned["Severity"].isin(["Critical", "High"])).sum())
    crit_only  = int((scanned["Severity"] == "Critical").sum())
    top5       = scanned.head(5)
    top_tech   = (
        scanned["Matched_Technology"].value_counts().index[0]
        if not scanned.empty else "N/A"
    )
    top_type   = (
        scanned["Vulnerability_Type"].value_counts().index[0]
        if "Vulnerability_Type" in scanned.columns and not scanned.empty
        else "N/A"
    )
    avg_score  = scanned["Risk_Score"].mean() if not scanned.empty else 0
    risk_level = (
        "CRITICAL" if avg_score >= 8 else
        "HIGH"     if avg_score >= 6 else
        "MEDIUM"   if avg_score >= 4 else
        "LOW"
    )
    av_dist    = {}
    for v in scanned.get("Attack Vector", pd.Series(dtype=str)).dropna():
        av = _parse_cvss_vector(v).get("AV", "?")
        av_dist[av] = av_dist.get(av, 0) + 1
    av_labels  = {"N": "Network", "A": "Adjacent", "L": "Local", "P": "Physical"}
    top_av     = max(av_dist, key=av_dist.get) if av_dist else "N"
    top_av_pct = (av_dist.get(top_av, 0) / max(total, 1)) * 100

    lines = [
        "=" * 64,
        "  SECURITY POSTURE ASSESSMENT REPORT",
        f"  Generated : {timestamp}",
        f"  Scope     : {', '.join(techs)}",
        "=" * 64,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        f"Overall Risk Level : {risk_level}",
        f"Total CVEs Found   : {total}",
        f"Critical / High    : {crit_high}  ({crit_high/max(total,1)*100:.0f}% of exposure)",
        f"Critical only      : {crit_only}",
        f"Average Risk Score : {avg_score:.1f} / 10",
        f"Most Exposed Tech  : {top_tech}",
        f"Dominant Vuln Type : {top_type}",
        f"Primary Attack Vec : {av_labels.get(top_av, top_av)} ({top_av_pct:.0f}%)",
        "",
    ]

    lines += ["TOP PRIORITY ACTIONS", "-" * 40]
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        action = generate_recommended_action(
            row.get("Severity", "Unknown"),
            row.get("Vulnerability_Type", "Other"),
        )
        lines += [
            f"{i}. [{row.get('Severity','?').upper()}] {row.get('CVE ID','?')}",
            f"   Technology : {row.get('Matched_Technology','?')}",
            f"   Type       : {row.get('Vulnerability_Type','?')}",
            f"   Risk Score : {row.get('Risk_Score', 0):.1f} / 10  (CVSS {row.get('CVSS Score',0):.1f})",
            f"   Action     : {action}",
            "",
        ]

    lines += [
        "SEVERITY BREAKDOWN",
        "-" * 40,
    ]
    for sev in ["Critical", "High", "Medium", "Low"]:
        n   = int((scanned["Severity"] == sev).sum())
        bar = "█" * int(n / max(total, 1) * 30)
        lines.append(f"  {sev:<10} {n:>4}  {bar}")

    lines += [
        "",
        "EXPOSURE BY TECHNOLOGY",
        "-" * 40,
    ]
    for tech, cnt in scanned["Matched_Technology"].value_counts().items():
        bar = "█" * int(cnt / max(total, 1) * 30)
        lines.append(f"  {tech:<22} {cnt:>4}  {bar}")

    lines += [
        "",
        "─" * 64,
        "Generated by CVE Intelligence Analyzer",
        "─" * 64,
    ]
    return "\n".join(lines)


# ── Live NVD API helpers ──────────────────────────────────────────────────────

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def _get_nvd_api_key() -> str:
    """
    Retrieve the NVD API key from Streamlit secrets.

    On Streamlit Cloud set it under Settings → Secrets as::

        NVD_API_KEY = "your-key-here"

    Falls back to an empty string (anonymous, lower rate limit) if not set.
    """
    try:
        return st.secrets["NVD_API_KEY"]
    except (KeyError, FileNotFoundError):
        return ""


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_nvd_keyword(keyword: str, max_results: int) -> pd.DataFrame:
    """
    Query the NVD REST API v2 for CVEs matching *keyword*.

    Results are cached for 1 hour per (keyword, max_results) combination
    so repeated scans of the same stack are instant.

    Args:
        keyword:     Technology name to search (e.g. ``'Apache'``).
        max_results: Maximum number of CVEs to retrieve (NVD cap: 2 000).

    Returns:
        DataFrame with columns matching the local dataset schema, or an
        empty DataFrame on network / parsing failure.
    """
    api_key = _get_nvd_api_key()
    headers = {"apiKey": api_key} if api_key.strip() else {}
    params  = {
        "keywordSearch":   keyword,
        "resultsPerPage":  min(max_results, 2000),
        "startIndex":      0,
    }
    try:
        resp = requests.get(NVD_API_URL, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return pd.DataFrame({"_error": [f"Timeout fetching '{keyword}' from NVD."]})
    except requests.exceptions.ConnectionError:
        return pd.DataFrame({"_error": [f"Could not reach NVD API — check your internet connection."]})
    except Exception as exc:
        return pd.DataFrame({"_error": [str(exc)]})

    rows = []
    for item in data.get("vulnerabilities", []):
        cve    = item.get("cve", {})
        cve_id = cve.get("id", "Unknown")

        # English description
        desc = next(
            (d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"),
            "No description available.",
        )

        # CVSS score + vector (prefer v3.1 > v3.0 > v2)
        cvss_score     = 0.0
        attack_vector  = ""
        for mkey in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            metrics = cve.get("metrics", {}).get(mkey, [])
            if metrics:
                cd            = metrics[0].get("cvssData", {})
                cvss_score    = float(cd.get("baseScore", 0.0))
                attack_vector = cd.get("vectorString", "")
                break

        # Year from CVE ID
        m    = re.search(r"CVE-(\d{4})-", cve_id)
        year = int(m.group(1)) if m else 0

        from utils.preprocessing import get_severity_label, classify_vulnerability_type
        rows.append({
            "CVE ID":             cve_id,
            "Description":        desc,
            "CVSS Score":         cvss_score,
            "Attack Vector":      attack_vector,
            "Severity":           get_severity_label(cvss_score),
            "CVE_Year":           year,
            "Vulnerability_Type": classify_vulnerability_type(desc),
            "Affected_Software":  keyword,
            "Affected OS":        "N/A",
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _scan_nvd_stack(
    techs: list,
    max_per_tech: int,
    min_cvss: float,
    status_container,
) -> pd.DataFrame:
    """
    Query the NVD API for each technology in *techs* with polite rate limiting.

    Rate limit is determined automatically from the stored API key:
    with key → 0.7 s gap, without → 6.2 s gap.

    Args:
        techs:            List of technology name strings.
        max_per_tech:     Max CVEs to fetch per technology.
        min_cvss:         Only keep CVEs at or above this score.
        status_container: ``st.status`` context used to stream progress messages.

    Returns:
        Deduplicated, risk-scored DataFrame sorted by Risk_Score descending.
    """
    api_key = _get_nvd_api_key()
    delay   = 0.7 if api_key.strip() else 6.2   # NVD rate-limit gap
    frames  = []
    errors  = []

    for i, tech in enumerate(techs):
        status_container.write(f"🔍 Fetching CVEs for **{tech}** ({i+1}/{len(techs)})…")
        result = _fetch_nvd_keyword(tech, max_per_tech)

        if "_error" in result.columns:
            errors.append(result["_error"].iloc[0])
            continue
        if not result.empty:
            result["Matched_Technology"] = tech
            frames.append(result)

        # Rate-limit pause (skip on last request)
        if i < len(techs) - 1:
            time.sleep(delay)

    if errors:
        status_container.write("⚠️ Errors: " + " | ".join(errors))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.sort_values("CVSS Score", ascending=False)
                .drop_duplicates(subset="CVE ID")
                .reset_index(drop=True)
    )
    combined = combined[combined["CVSS Score"] >= min_cvss].copy()
    combined["Risk_Score"] = combined.apply(_compute_risk_score, axis=1)
    combined["Priority"]   = combined["Risk_Score"].apply(_priority_label)
    combined = combined.sort_values("Risk_Score", ascending=False).reset_index(drop=True)
    combined.index += 1
    return combined


if page == "🔍 CVE Analyzer":
    st.markdown(
        '<div class="main-header">'
        "<h1>🛡️ CVE Intelligence Analyzer</h1>"
        "<p>Paste a vulnerability description for instant classification, severity assessment, "
        "keywords &amp; an actionable security alert.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Input section ─────────────────────────────────────────────
    st.markdown('<div class="section-title">📝 CVE Description</div>', unsafe_allow_html=True)
    default_text = (
        "A buffer overflow vulnerability exists in Apache HTTP Server 2.4.58 "
        "that allows remote attackers to execute arbitrary code via crafted HTTP requests."
    )
    cve_text = st.text_area(
        "CVE Description",
        value=default_text,
        height=130,
        label_visibility="collapsed",
        placeholder="Paste a CVE description here…",
    )

    cvss_col, _ = st.columns([1, 3])
    with cvss_col:
        cvss_input = st.number_input(
            "CVSS Score (optional, 0–10)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            format="%.1f",
            help="If provided, severity is derived directly from the CVSS score.",
        )
    cvss_score = cvss_input if cvss_input > 0.0 else None

    analyze_clicked = st.button(
        "Analyze Vulnerability", type="primary", use_container_width=True
    )

    if analyze_clicked:
        if not cve_text.strip():
            st.warning("Please enter a CVE description.")
        else:
            with st.spinner("Running NLP pipeline…"):

                # ── Classify vulnerability type ────────────────────
                bert_model, bert_tokenizer, bert_le = load_bert_classifier()
                if bert_model is not None:
                    try:
                        vuln_type, vt_confidence = _bert_predict(
                            cve_text, bert_model, bert_tokenizer, bert_le
                        )
                        vt_source = f"DistilBERT · {vt_confidence:.1%} confidence"
                    except Exception as exc:
                        st.warning(f"BERT inference failed: {exc}")
                        vuln_type, vt_confidence = classify_vulnerability(cve_text)
                        vt_source = "Rule-based fallback"
                else:
                    vuln_type, vt_confidence = classify_vulnerability(cve_text)
                    vt_source = "Rule-based classifier"

                # ── Predict severity ───────────────────────────────
                severity, sev_confidence = predict_severity(cve_text, cvss_score)
                sev_source = (
                    "CVSS direct mapping" if cvss_score is not None
                    else ("ML model" if _severity_model_available() else "Rule-based")
                )

                # ── Extract keywords ───────────────────────────────
                tfidf_vec = load_tfidf_vectorizer()
                if tfidf_vec is not None:
                    try:
                        cleaned = preprocess_text(cve_text)
                        tfidf_scores = tfidf_vec.transform([cleaned])
                        feat_names   = tfidf_vec.get_feature_names_out()
                        scores_flat  = tfidf_scores.toarray().flatten()
                        top_idx      = scores_flat.argsort()[::-1][:10]
                        keywords     = [
                            (feat_names[i], float(scores_flat[i]))
                            for i in top_idx
                            if scores_flat[i] > 0
                        ]
                    except Exception:
                        keywords = extract_keywords(cve_text, top_n=10)
                else:
                    keywords = extract_keywords(cve_text, top_n=10)

                # ── Summarise ──────────────────────────────────────
                summarizer = _load_summarizer_cached()
                summary    = summarize_cve(cve_text, summarizer)

                # ── Recommended action ─────────────────────────────
                action = generate_recommended_action(severity, vuln_type)

            # ── Results layout ─────────────────────────────────────
            left_col, right_col = st.columns(2, gap="large")

            # ── Left column ────────────────────────────────────────
            with left_col:
                # Vulnerability type
                st.markdown(
                    '<div class="section-title">🏷️ Vulnerability Type</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="glass-card">'
                    f'<div style="font-size:1.3rem;font-weight:700;color:#302b63">{vuln_type}</div>'
                    f'<div style="margin-top:.4rem;font-size:.82rem;color:#888">{vt_source}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Severity
                st.markdown(
                    '<div class="section-title">⚠️ Predicted Severity</div>',
                    unsafe_allow_html=True,
                )
                sev_cls = f"severity-{severity.lower()}"
                conf_pct = f"{sev_confidence:.1%}" if sev_confidence < 1.0 else "100% (CVSS)"
                st.markdown(
                    f'<div class="glass-card">'
                    f'<span class="severity-badge {sev_cls}">{severity}</span>'
                    f'<div style="margin-top:.5rem;font-size:.82rem;color:#888">'
                    f"Confidence: <strong>{conf_pct}</strong> · {sev_source}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Keywords
                st.markdown(
                    '<div class="section-title">🔑 Top Keywords</div>',
                    unsafe_allow_html=True,
                )
                if keywords:
                    chips = "".join(
                        f'<span class="chip">{kw}</span>'
                        for kw, _ in keywords[:5]
                    )
                    st.markdown(
                        f'<div class="chip-container">{chips}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No significant keywords extracted.")

            # ── Right column ───────────────────────────────────────
            with right_col:
                # AI Summary
                st.markdown(
                    '<div class="section-title">🤖 AI-Generated Summary</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="glass-card" style="font-size:.92rem;line-height:1.65">'
                    f"{summary}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Recommended action
                st.markdown(
                    '<div class="section-title">🛠️ Recommended Action</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="glass-card" style="font-size:.9rem;line-height:1.65;'
                    f'border-left:4px solid #e53935;">'
                    f"{action}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # CVE Alert (copy-paste text area)
                st.markdown(
                    '<div class="section-title">📋 CVE Alert (copy-paste ready)</div>',
                    unsafe_allow_html=True,
                )
                alert      = generate_alert(cve_text, vuln_type, severity, summary)
                alert_text = format_alert_text(alert)
                st.text_area(
                    "CVE Alert",
                    value=alert_text,
                    height=195,
                    label_visibility="collapsed",
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Vulnerability Dashboard
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Vulnerability Dashboard":
    st.markdown(
        '<div class="main-header">'
        "<h1>📊 Vulnerability Dashboard</h1>"
        "<p>Explore distribution patterns and trends across the CVE dataset.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if df.empty:
        st.error("Dataset not found. Ensure data/cve_dataset.csv exists.")
        st.stop()

    # ── 4 metric cards ─────────────────────────────────────────────
    total        = len(df)
    critical_pct = (
        f"{(df['Severity'] == 'Critical').mean() * 100:.1f}%"
        if "Severity" in df.columns else "N/A"
    )
    top_software = (
        df["Affected_Software"].value_counts().index[0]
        if "Affected_Software" in df.columns
        and df["Affected_Software"].nunique() > 1
        else "N/A"
    )
    top_type = (
        df["Vulnerability_Type"].value_counts().index[0]
        if "Vulnerability_Type" in df.columns
        else "N/A"
    )
    # Shorten long labels for the pill display
    top_software_short = (top_software[:12] + "…") if len(top_software) > 14 else top_software
    top_type_short     = (top_type[:14] + "…")     if len(top_type) > 16     else top_type

    st.markdown(
        '<div class="metric-row">'
        f'<div class="metric-pill"><div class="mp-value">{total:,}</div>'
        f'<div class="mp-label">Total CVEs</div></div>'
        f'<div class="metric-pill red"><div class="mp-value">{critical_pct}</div>'
        f'<div class="mp-label">Critical</div></div>'
        f'<div class="metric-pill orange"><div class="mp-value">{top_software_short}</div>'
        f'<div class="mp-label">Top Affected Software</div></div>'
        f'<div class="metric-pill green"><div class="mp-value">{top_type_short}</div>'
        f'<div class="mp-label">Most Common Type</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Row 1: Severity bar + Year line ───────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            '<div class="section-title">CVEs by Severity</div>',
            unsafe_allow_html=True,
        )
        if "Severity" in df.columns:
            sev_order  = ["Critical", "High", "Medium", "Low"]
            sev_counts = (
                df["Severity"]
                .value_counts()
                .reindex([s for s in sev_order if s in df["Severity"].unique()])
            )
            color_map  = {
                "Critical": "#e53935",
                "High":     "#fb8c00",
                "Medium":   "#fdd835",
                "Low":      "#43a047",
            }
            fig = px.bar(
                x=sev_counts.index,
                y=sev_counts.values,
                color=sev_counts.index,
                color_discrete_map=color_map,
                labels={"x": "Severity", "y": "Count"},
            )
            fig.update_layout(height=350, showlegend=False, **_CHART)
            fig.update_traces(marker_line_width=0, marker_cornerradius=6)
            st.plotly_chart(fig)

    with c2:
        st.markdown(
            '<div class="section-title">CVEs by Year</div>',
            unsafe_allow_html=True,
        )
        if "CVE_Year" in df.columns:
            year_counts = df["CVE_Year"].dropna().value_counts().sort_index()
            fig = px.line(
                x=year_counts.index.astype(int),
                y=year_counts.values,
                labels={"x": "Year", "y": "Count"},
                markers=True,
            )
            fig.update_traces(
                line_color="#667eea",
                line_width=2,
                marker=dict(size=8, color="#764ba2"),
            )
            fig.update_layout(height=350, **_CHART)
            st.plotly_chart(fig)

    # ── Row 2: Top 10 Software (h-bar) + Vuln types (pie) ─────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown(
            '<div class="section-title">Top 10 Most Vulnerable Software</div>',
            unsafe_allow_html=True,
        )
        if "Affected_Software" in df.columns:
            sw_counts = (
                df[df["Affected_Software"] != "Unknown"]["Affected_Software"]
                .value_counts()
                .head(10)
            )
            if not sw_counts.empty:
                fig = px.bar(
                    x=sw_counts.values,
                    y=sw_counts.index,
                    orientation="h",
                    labels={"x": "Count", "y": ""},
                    color=sw_counts.values,
                    color_continuous_scale=[[0, "#e8eaf6"], [1, "#302b63"]],
                )
                fig.update_layout(
                    height=370,
                    showlegend=False,
                    coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                    **_CHART,
                )
                fig.update_traces(marker_line_width=0, marker_cornerradius=4)
                st.plotly_chart(fig)

    with c4:
        st.markdown(
            '<div class="section-title">Vulnerability Type Distribution</div>',
            unsafe_allow_html=True,
        )
        if "Vulnerability_Type" in df.columns:
            vt_counts = df["Vulnerability_Type"].value_counts()
            fig = px.pie(
                values=vt_counts.values,
                names=vt_counts.index,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_traces(textinfo="percent+label", textfont_size=11)
            fig.update_layout(height=370, showlegend=False, **_CHART)
            st.plotly_chart(fig)

    # ── Searchable data table ──────────────────────────────────────
    st.markdown(
        '<div class="section-title">🔍 Searchable CVE Table</div>',
        unsafe_allow_html=True,
    )

    filter_cols = st.columns(3)
    with filter_cols[0]:
        sev_options  = ["All"] + sorted(df["Severity"].unique().tolist()) if "Severity" in df.columns else ["All"]
        sev_filter   = st.selectbox("Filter by Severity", sev_options)
    with filter_cols[1]:
        type_options = ["All"] + sorted(df["Vulnerability_Type"].unique().tolist()) if "Vulnerability_Type" in df.columns else ["All"]
        type_filter  = st.selectbox("Filter by Type", type_options)
    with filter_cols[2]:
        search_text = st.text_input("Search description", placeholder="e.g. Apache, WordPress…")

    display_df = df.copy()
    if sev_filter != "All" and "Severity" in display_df.columns:
        display_df = display_df[display_df["Severity"] == sev_filter]
    if type_filter != "All" and "Vulnerability_Type" in display_df.columns:
        display_df = display_df[display_df["Vulnerability_Type"] == type_filter]
    if search_text.strip() and "Description" in display_df.columns:
        mask       = display_df["Description"].str.contains(search_text, case=False, na=False)
        display_df = display_df[mask]

    # Choose columns to show
    show_cols = [c for c in ["CVE ID", "CVSS Score", "Severity", "Vulnerability_Type",
                              "Affected_Software", "Description"] if c in display_df.columns]
    st.dataframe(
        display_df[show_cols].reset_index(drop=True),
        use_container_width=True,
        height=400,
    )
    st.caption(f"Showing {len(display_df):,} of {len(df):,} records")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Similar CVE Search
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔎 Similar CVE Search":
    st.markdown(
        '<div class="main-header">'
        "<h1>🔎 Similar CVE Search</h1>"
        "<p>Find vulnerabilities with similar attack patterns using semantic embeddings.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Describe a vulnerability to search</div>',
        unsafe_allow_html=True,
    )
    query = st.text_area(
        "Query",
        value="SQL injection vulnerability in login page allowing authentication bypass",
        height=100,
        label_visibility="collapsed",
    )

    num_results = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if st.button("Find Similar CVEs", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a vulnerability description.")
        elif df.empty:
            st.error("Dataset not loaded — ensure data/cve_dataset.csv exists.")
        else:
            sbert_model = _load_sbert_cached()

            with st.spinner("Searching for similar CVEs…"):
                results = find_similar_cves(
                    query_text=query,
                    top_k=num_results,
                    encoder=sbert_model,
                    df=df,
                )

            if not results:
                st.info("No results found.")
            else:
                st.markdown(
                    f'<div class="section-title">Top {len(results)} Matches</div>',
                    unsafe_allow_html=True,
                )
                method = "Sentence-BERT embeddings" if sbert_model else "TF-IDF cosine similarity"
                st.caption(f"Search method: {method}")

                for r in results:
                    severity   = r.get("severity", "N/A")
                    sev_cls    = f"severity-{severity.lower()}" if severity in (
                        "Critical", "High", "Medium", "Low"
                    ) else ""
                    sim_score  = r["similarity_score"]
                    sim_pct    = int(sim_score * 100)

                    with st.expander(
                        f"🔹 {r['cve_id']}  ·  {severity}  ·  Similarity {sim_score:.3f}",
                        expanded=(sim_score > 0.5),
                    ):
                        badge_col, bar_col = st.columns([1, 3])
                        with badge_col:
                            st.markdown(
                                f'<span class="severity-badge {sev_cls}">{severity}</span>',
                                unsafe_allow_html=True,
                            )
                        with bar_col:
                            st.progress(min(sim_pct, 100), text=f"Similarity: {sim_score:.3f}")

                        st.write(r["description"])


elif page == "🎯 Attack Surface Scanner":
    from datetime import datetime

    st.markdown(
        '<div class="main-header">'
        "<h1>🎯 Attack Surface Scanner</h1>"
        "<p>Enter your technology stack to find every CVE that targets your infrastructure — "
        "risk-scored, ranked, and packaged into an executive report. "
        "Works with the local dataset or live NVD data for <em>any</em> technology.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Data source toggle ────────────────────────────────────────
    st.markdown(
        '<div class="section-title">📡 Data Source</div>',
        unsafe_allow_html=True,
    )
    data_source = st.radio(
        "Data source",
        [
            "📁 Local Dataset  (offline · 1,314 CVEs · instant)",
            "🌐 Live NVD API  (online · any tech · always current)",
        ],
        label_visibility="collapsed",
        horizontal=True,
    )
    live_mode = data_source.startswith("🌐")

    if live_mode:
        _key_active = bool(_get_nvd_api_key().strip())
        _rate_label = "~0.7 s / request" if _key_active else "~6 s / request (no key)"
        _key_badge  = "✅ API key active" if _key_active else "⚠️ No API key — slower rate limit"
        st.info(
            f"**Live NVD mode** queries the official National Vulnerability Database "
            f"in real-time. Works for any technology — Kubernetes, Rust crates, cloud "
            f"services, proprietary software — not just what's in the local dataset.  \n"
            f"**{_key_badge}** · Rate: {_rate_label}",
            icon="ℹ️",
        )
        max_per_tech = st.slider(
            "Max CVEs per technology",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="NVD returns up to 2,000 per query; keep this low for speed.",
        )
    else:
        max_per_tech = 50

    # ── Tech stack input ──────────────────────────────────────────
    st.markdown(
        '<div class="section-title">🖥️ Your Technology Stack</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "One technology per line or comma-separated. "
        "Local mode: use broad names like 'Apache', 'WordPress'. "
        "Live NVD mode: use exact product names — e.g. 'Kubernetes', 'log4j', 'OpenSSL 3.0', 'nginx'."
    )

    default_local  = "WordPress\nApache\nMySQL\nOpenSSL\nPHP"
    default_live   = "nginx\nKubernetes\nOpenSSL\nlog4j\nDocker"
    tech_input = st.text_area(
        "Tech Stack",
        value=default_live if live_mode else default_local,
        height=130,
        label_visibility="collapsed",
    )

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        min_cvss = st.slider(
            "Minimum CVSS score",
            min_value=0.0, max_value=10.0, value=4.0, step=0.1,
            help="4.0 = Medium and above.",
        )
    with opt_col2:
        show_top_n = st.slider(
            "Max results to display",
            min_value=10, max_value=500, value=50, step=10,
        )

    scan_clicked = st.button(
        "🔍 Scan My Attack Surface", type="primary", use_container_width=True
    )

    if scan_clicked:
        raw_lines = tech_input.replace(",", "\n").splitlines()
        techs     = [t.strip() for t in raw_lines if t.strip()]

        if not techs:
            st.warning("Please enter at least one technology.")
        else:
            # ── Fetch CVEs ─────────────────────────────────────────
            if live_mode:
                _has_key = bool(_get_nvd_api_key().strip())
                delay_s  = 0.7 if _has_key else 6.2
                est_s    = len(techs) * delay_s
                st.caption(
                    f"Querying NVD for {len(techs)} technologies "
                    f"(estimated {est_s:.0f} s — results cached for 1 h)."
                )
                with st.status("Fetching live CVE data from NVD…", expanded=True) as status:
                    scanned = _scan_nvd_stack(
                        techs, max_per_tech, min_cvss, status
                    )
                    if scanned.empty:
                        status.update(label="No CVEs found.", state="error")
                    else:
                        status.update(
                            label=f"✅ Found {len(scanned)} CVEs across {len(techs)} technologies.",
                            state="complete",
                        )
            else:
                if df.empty:
                    st.error("Dataset not loaded — ensure data/cve_dataset.csv exists.")
                    st.stop()
                with st.spinner(f"Scanning {len(df):,} CVEs for {len(techs)} technologies…"):
                    scanned = _scan_tech_stack(techs, df, min_cvss)

            # ── Results ────────────────────────────────────────────
            if scanned.empty:
                st.info(
                    "No CVEs found for those technologies at the selected CVSS threshold. "
                    + ("Try broadening the search terms or lowering the minimum CVSS."
                       if not live_mode else
                       "NVD may not index those exact names — try alternate spellings "
                       "or shorter product names.")
                )
            else:
                source_badge = "🌐 NVD Live" if live_mode else "📁 Local Dataset"
                total_found = len(scanned)
                crit_high_n = int(scanned["Severity"].isin(["Critical", "High"]).sum())
                avg_risk    = scanned["Risk_Score"].mean()
                top_tech_n  = scanned["Matched_Technology"].value_counts().index[0]

                risk_color = (
                    "red"    if avg_risk >= 8 else
                    "orange" if avg_risk >= 6 else
                    "green"
                )
                st.markdown(
                    '<div class="metric-row">'
                    f'<div class="metric-pill"><div class="mp-value">{total_found}</div>'
                    f'<div class="mp-label">CVEs Found · {source_badge}</div></div>'
                    f'<div class="metric-pill red"><div class="mp-value">{crit_high_n}</div>'
                    f'<div class="mp-label">Critical / High</div></div>'
                    f'<div class="metric-pill {risk_color}"><div class="mp-value">{avg_risk:.1f}</div>'
                    f'<div class="mp-label">Avg Risk Score</div></div>'
                    f'<div class="metric-pill orange"><div class="mp-value">{top_tech_n[:12]}</div>'
                    f'<div class="mp-label">Most Exposed</div></div>'
                    "</div>",
                    unsafe_allow_html=True,
                )

                # ── Priority patch queue ───────────────────────────
                st.markdown(
                    '<div class="section-title">🚨 Priority Patch Queue</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Risk Score = CVSS base + network/auth/complexity bonuses × vuln-type weight. "
                    "Sorted highest risk first."
                )
                display_cols = [
                    c for c in [
                        "CVE ID", "Matched_Technology", "CVSS Score", "Risk_Score",
                        "Priority", "Severity", "Vulnerability_Type",
                    ]
                    if c in scanned.columns
                ]
                st.dataframe(
                    scanned[display_cols].head(show_top_n),
                    use_container_width=True,
                    height=420,
                    column_config={
                        "Risk_Score":    st.column_config.ProgressColumn(
                            "Risk Score", min_value=0, max_value=10, format="%.2f"
                        ),
                        "CVSS Score":         st.column_config.NumberColumn(format="%.1f"),
                        "CVE ID":             st.column_config.TextColumn("CVE ID"),
                        "Priority":           st.column_config.TextColumn("Priority"),
                        "Severity":           st.column_config.TextColumn("Severity"),
                        "Vulnerability_Type": st.column_config.TextColumn("Type"),
                        "Matched_Technology": st.column_config.TextColumn("Technology"),
                    },
                )

                # ── Charts ─────────────────────────────────────────
                chart_l, chart_r = st.columns(2)

                with chart_l:
                    st.markdown(
                        '<div class="section-title">Exposure by Technology</div>',
                        unsafe_allow_html=True,
                    )
                    tech_counts = scanned["Matched_Technology"].value_counts()
                    fig = px.bar(
                        x=tech_counts.values, y=tech_counts.index,
                        orientation="h", labels={"x": "CVEs", "y": ""},
                        color=tech_counts.values,
                        color_continuous_scale=[[0, "#ffcdd2"], [0.5, "#fb8c00"], [1, "#b71c1c"]],
                    )
                    fig.update_layout(
                        height=320, showlegend=False, coloraxis_showscale=False,
                        yaxis=dict(autorange="reversed"), **_CHART,
                    )
                    fig.update_traces(marker_line_width=0, marker_cornerradius=4)
                    st.plotly_chart(fig)

                with chart_r:
                    st.markdown(
                        '<div class="section-title">Severity Breakdown</div>',
                        unsafe_allow_html=True,
                    )
                    sev_c = scanned["Severity"].value_counts()
                    fig = px.pie(
                        values=sev_c.values, names=sev_c.index, hole=0.45,
                        color=sev_c.index,
                        color_discrete_map={
                            "Critical": "#e53935", "High": "#fb8c00",
                            "Medium": "#fdd835", "Low": "#43a047",
                        },
                    )
                    fig.update_traces(textinfo="percent+label", textfont_size=11)
                    fig.update_layout(height=320, showlegend=False, **_CHART)
                    st.plotly_chart(fig)

                # ── Risk distribution ──────────────────────────────
                st.markdown(
                    '<div class="section-title">Risk Score Distribution</div>',
                    unsafe_allow_html=True,
                )
                fig = px.histogram(
                    scanned, x="Risk_Score", nbins=20,
                    color_discrete_sequence=["#e53935"],
                    labels={"Risk_Score": "Risk Score (0–10)"},
                )
                fig.add_vline(
                    x=avg_risk, line_dash="dash", line_color="#302b63",
                    annotation_text=f"  Avg {avg_risk:.1f}",
                    annotation_position="top right",
                )
                fig.update_layout(height=260, bargap=0.05, **_CHART)
                fig.update_traces(marker_line_width=0, marker_cornerradius=4)
                st.plotly_chart(fig)

                # ── Executive report ───────────────────────────────
                st.markdown(
                    '<div class="section-title">📄 Executive Security Report</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Select all → copy → paste into a ticket, email, or Slack.")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                report    = _executive_summary(scanned, techs, timestamp)
                st.text_area(
                    "Executive Report", value=report,
                    height=540, label_visibility="collapsed",
                )

                # ── CSV export ─────────────────────────────────────
                st.markdown(
                    '<div class="section-title">⬇️ Export</div>',
                    unsafe_allow_html=True,
                )
                csv_bytes = scanned.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Patch Queue as CSV",
                    data=csv_bytes,
                    file_name=f"patch_queue_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;margin-top:2.5rem;font-size:.78rem;color:#aaa;">'
    "CVE Intelligence Analyzer · Built with Streamlit, HuggingFace Transformers &amp; scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)
