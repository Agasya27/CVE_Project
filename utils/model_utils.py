"""
Model utility functions for CVE Intelligence Analyzer.

All heavy model-loading functions return ``None`` on failure so the Streamlit
app can gracefully degrade to rule-based fallbacks without crashing.

Public API
----------
load_summarizer()              → HuggingFace summarization pipeline or None
summarize_cve(text, ...)       → Summary string

load_classifier()              → HuggingFace zero-shot pipeline or None
classify_vulnerability(text, ...)  → (label, confidence)

predict_severity(text, ...)    → (severity_label, confidence)

load_sentence_encoder()        → SentenceTransformer or None
find_similar_cves(query, ...)  → list[dict]

generate_recommended_action(severity, vuln_type) → str

save_model / load_model / save_dataframe / load_dataframe  (I/O helpers)
generate_alert / format_alert_text  (formatting helpers)
"""

import re
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"

# Labels used by the zero-shot classifier
ZERO_SHOT_LABELS = [
    "Remote Code Execution",
    "SQL Injection",
    "Denial of Service",
    "Cross-Site Scripting",
    "Privilege Escalation",
    "Information Disclosure",
    "Buffer Overflow",
    "Authentication Bypass",
]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_model(model, filename: str) -> Path:
    """Save *model* to ``models/<filename>`` via joblib."""
    MODELS_DIR.mkdir(exist_ok=True)
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    logger.info("Model saved to %s", filepath)
    return filepath


def load_model(filename: str):
    """Load and return a joblib model from ``models/<filename>``."""
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    return joblib.load(filepath)


def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    """Save *df* to ``data/<filename>`` as CSV."""
    DATA_DIR.mkdir(exist_ok=True)
    filepath = DATA_DIR / filename
    df.to_csv(filepath, index=False)
    logger.info("DataFrame saved to %s", filepath)
    return filepath


def load_dataframe(filename: str) -> pd.DataFrame:
    """Load and return a CSV DataFrame from ``data/<filename>``."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    return pd.read_csv(filepath)


# ── Summarisation ─────────────────────────────────────────────────────────────

def load_summarizer():
    """
    Load the ``facebook/bart-large-cnn`` summarisation pipeline.

    Forces CPU (``device=-1``) so no GPU is required.  Downloads the model
    on first call (~1.6 GB); cached by HuggingFace thereafter.

    Returns:
        HuggingFace ``pipeline`` object, or ``None`` on failure.
    """
    try:
        from transformers import pipeline
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1,
        )
        return summarizer
    except Exception as exc:
        logger.warning("Could not load summarizer: %s", exc)
        return None


def summarize_cve(text, summarizer=None) -> str:
    """
    Summarise a CVE description into a concise alert sentence.

    If *summarizer* is provided the BART model is used.  Otherwise falls back
    to an extractive approach (longest sentence, truncated to 200 chars).

    Args:
        text:       Raw CVE description.
        summarizer: HuggingFace summarisation pipeline (optional).

    Returns:
        Summary string.
    """
    if not isinstance(text, str) or not text.strip():
        return "No description provided."
    text = text.strip()
    if len(text) < 50:
        return text

    if summarizer is not None:
        try:
            truncated  = text[:4000]
            # Avoid "max_length shorter than input" warning by capping to input word count
            word_count = len(truncated.split())
            max_len    = min(80, max(word_count - 2, 10))
            min_len    = min(20, max(max_len // 2, 5))
            result = summarizer(
                truncated,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
            return result[0]["summary_text"]
        except Exception as exc:
            logger.warning("Summarisation failed: %s", exc)

    # Extractive fallback: pick the longest (usually most informative) sentence
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text[:200]
    best = max(sentences, key=len)
    if len(best) > 200:
        best = best[:200].rsplit(" ", 1)[0] + "…"
    return best


# ── Vulnerability classification ──────────────────────────────────────────────

def load_classifier():
    """
    Load the ``facebook/bart-large-mnli`` zero-shot classification pipeline.

    Forces CPU.  Downloads the model on first call (~1.6 GB).

    Returns:
        HuggingFace ``pipeline`` object, or ``None`` on failure.
    """
    try:
        from transformers import pipeline
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
        return classifier
    except Exception as exc:
        logger.warning("Could not load zero-shot classifier: %s", exc)
        return None


def classify_vulnerability(text, classifier=None) -> tuple:
    """
    Classify vulnerability type.

    Uses zero-shot classification when *classifier* is provided, otherwise
    falls back to regex-based pattern matching.

    Args:
        text:       CVE description.
        classifier: HuggingFace zero-shot pipeline (optional).

    Returns:
        ``(label, confidence)`` tuple where *confidence* is 0.0 for the
        regex fallback.
    """
    if not isinstance(text, str) or not text.strip():
        return "Other", 0.0

    if classifier is not None:
        try:
            result = classifier(
                text[:512],
                candidate_labels=ZERO_SHOT_LABELS,
                hypothesis_template="This vulnerability is related to {}.",
            )
            return result["labels"][0], float(result["scores"][0])
        except Exception as exc:
            logger.warning("Zero-shot classification failed: %s", exc)

    # Regex fallback
    from utils.preprocessing import classify_vulnerability_type
    label = classify_vulnerability_type(text)
    return label, 0.0


# ── Severity prediction ───────────────────────────────────────────────────────

def predict_severity(text, cvss_score=None) -> tuple:
    """
    Predict severity from CVSS score or via the trained ML model.

    Resolution order:
    1. If *cvss_score* is given, map it directly:
       ``0–3.9 → Low``, ``4–6.9 → Medium``, ``7–8.9 → High``, ``9–10 → Critical``.
    2. Else load the saved ``severity_predictor.joblib`` LogisticRegression model.
    3. Else rule-based keyword fallback.

    Args:
        text:       CVE description string.
        cvss_score: Optional numeric CVSS score (0–10).

    Returns:
        ``(severity_label, confidence)`` where confidence is in ``[0, 1]``.
    """
    # ── Direct CVSS mapping ──────────────────────────────────────
    if cvss_score is not None:
        try:
            score = float(cvss_score)
            if score < 4.0:   label = "Low"
            elif score < 7.0: label = "Medium"
            elif score < 9.0: label = "High"
            else:              label = "Critical"
            return label, 1.0
        except (TypeError, ValueError):
            pass

    # ── Saved ML model ───────────────────────────────────────────
    paths = {
        "model":     MODELS_DIR / "severity_predictor.joblib",
        "tfidf":     MODELS_DIR / "severity_tfidf.joblib",
        "encoder":   MODELS_DIR / "severity_encoder.joblib",
        "vuln_cols": MODELS_DIR / "severity_vuln_columns.joblib",
    }
    if all(p.exists() for p in paths.values()):
        try:
            from scipy.sparse import hstack, csr_matrix
            from utils.preprocessing import preprocess_text, classify_vulnerability_type

            model     = joblib.load(paths["model"])
            tfidf     = joblib.load(paths["tfidf"])
            encoder   = joblib.load(paths["encoder"])
            vuln_cols = joblib.load(paths["vuln_cols"])

            cleaned   = preprocess_text(text)
            X_tfidf   = tfidf.transform([cleaned])

            vuln_type = classify_vulnerability_type(text)
            vuln_dict = {col: 0 for col in vuln_cols}
            col_name  = f"vuln_{vuln_type}"
            if col_name in vuln_dict:
                vuln_dict[col_name] = 1
            vuln_array = csr_matrix([list(vuln_dict.values())])

            X        = hstack([X_tfidf, vuln_array])
            pred     = model.predict(X)[0]
            proba    = model.predict_proba(X)[0]
            label    = encoder.inverse_transform([pred])[0]
            return label, float(max(proba))
        except Exception as exc:
            logger.warning("ML severity prediction failed: %s", exc)

    # ── Rule-based fallback ───────────────────────────────────────
    label = _rule_based_severity(text)
    return label, 0.5


def _rule_based_severity(text: str) -> str:
    """Keyword-based severity estimation used when no model is available."""
    t = text.lower()
    if any(k in t for k in [
        'remote code execution', 'arbitrary code', 'unauthenticated',
        ' rce ', 'root access', 'pre-auth', 'zero-day',
    ]):
        return "Critical"
    if any(k in t for k in [
        'privilege escalation', 'authentication bypass', 'command injection',
        'sql injection', 'buffer overflow',
    ]):
        return "High"
    if any(k in t for k in ['low impact', 'minor', 'informational']):
        return "Low"
    return "Medium"


# ── Sentence encoder & similarity search ─────────────────────────────────────

def load_sentence_encoder():
    """
    Load ``sentence-transformers/all-MiniLM-L6-v2`` for semantic similarity.

    Downloads the model (~90 MB) on first call; cached by the library thereafter.

    Returns:
        ``SentenceTransformer`` instance, or ``None`` on failure.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:
        logger.warning("Could not load sentence encoder: %s", exc)
        return None


def find_similar_cves(
    query_text: str,
    top_k: int = 5,
    encoder=None,
    df: pd.DataFrame = None,
) -> list:
    """
    Find CVEs similar to *query_text* by semantic or TF-IDF cosine similarity.

    On first call with *encoder* provided, embeddings are generated and cached
    to ``models/cve_embeddings.npy`` so subsequent searches are instant.
    Falls back to on-the-fly TF-IDF similarity if the encoder is unavailable.

    Args:
        query_text: Free-text vulnerability description.
        top_k:      Number of results to return.
        encoder:    ``SentenceTransformer`` instance (optional).
        df:         DataFrame with at minimum a ``Description`` column.
                    If ``None``, the function attempts to load
                    ``data/cve_cleaned.csv`` automatically.

    Returns:
        List of dicts with keys:
        ``cve_id``, ``description``, ``severity``, ``similarity_score``.
    """
    if not isinstance(query_text, str) or not query_text.strip():
        return []

    # ── Load dataset if not provided ─────────────────────────────
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        for fname in ["cve_cleaned.csv", "cve_with_keywords.csv",
                      "cve_preprocessed.csv", "cve_dataset.csv"]:
            p = DATA_DIR / fname
            if p.exists():
                df = pd.read_csv(p)
                break
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return []

    from sklearn.metrics.pairwise import cosine_similarity

    emb_path = MODELS_DIR / "cve_embeddings.npy"

    if encoder is not None:
        # Load cached embeddings or (re-)generate and cache them
        if emb_path.exists() and len(np.load(emb_path)) == len(df):
            embeddings = np.load(emb_path)
        else:
            descriptions = df["Description"].fillna("").tolist()
            embeddings   = encoder.encode(
                descriptions, show_progress_bar=False, batch_size=32
            )
            MODELS_DIR.mkdir(exist_ok=True)
            np.save(emb_path, embeddings)

        query_emb = encoder.encode([query_text])
        sims = cosine_similarity(query_emb, embeddings)[0]
    else:
        # TF-IDF cosine-similarity fallback
        import warnings
        from sklearn.feature_extraction.text import TfidfVectorizer
        descriptions = df["Description"].fillna("").tolist()
        vec          = TfidfVectorizer(max_features=5000, stop_words="english")
        matrix       = vec.fit_transform([query_text] + descriptions)
        # Suppress divide-by-zero/overflow that sklearn raises for zero-norm rows
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
        sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)

    top_indices = sims.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        row      = df.iloc[idx]
        severity = row.get("Severity", "N/A") if "Severity" in df.columns else "N/A"
        results.append({
            "cve_id":           str(row.get("CVE ID", "Unknown")),
            "description":      str(row.get("Description", ""))[:300],
            "severity":         str(severity),
            "similarity_score": float(sims[idx]),
        })
    return results


# ── Alert formatting ──────────────────────────────────────────────────────────

def generate_alert(
    description: str,
    vuln_type: str,
    severity: str,
    summary: str,
) -> dict:
    """
    Build a structured security alert dictionary.

    Args:
        description: Original CVE description.
        vuln_type:   Classified vulnerability type.
        severity:    Predicted severity label.
        summary:     AI- or extractive-generated summary.

    Returns:
        Dict with keys ``severity``, ``type``, ``summary``,
        ``recommended_action``, ``original_description``.
    """
    return {
        "severity":             severity,
        "type":                 vuln_type,
        "summary":              summary,
        "recommended_action":   generate_recommended_action(severity, vuln_type),
        "original_description": description,
    }


def format_alert_text(alert: dict) -> str:
    """
    Format an alert dict into a copy-paste-ready plain-text block.

    Args:
        alert: Dict produced by :func:`generate_alert`.

    Returns:
        Multi-line string suitable for display in a ``st.text_area``.
    """
    icons = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
    icon  = icons.get(alert.get("severity", ""), "⚪")
    lines = [
        f"{icon} CVE SECURITY ALERT",
        f"Severity : {alert.get('severity', 'Unknown')}",
        f"Type     : {alert.get('type', 'Unknown')}",
        "",
        f"Summary  : {alert.get('summary', '')}",
        "",
        "Recommended Action:",
        f"  {alert.get('recommended_action', '')}",
    ]
    return "\n".join(lines)


# ── Recommended-action matrix ─────────────────────────────────────────────────

def generate_recommended_action(severity: str, vuln_type: str) -> str:
    """
    Return specific, actionable remediation guidance for a
    severity × vulnerability-type combination.

    Covers all four severity levels (Critical / High / Medium / Low) crossed
    with the eight primary vulnerability types.  Unknown combinations fall
    back to a severity-level default.

    Args:
        severity: One of ``Critical``, ``High``, ``Medium``, ``Low``.
        vuln_type: Vulnerability class string (partial matches are accepted).

    Returns:
        Human-readable remediation guidance string.
    """
    sev = severity.strip().title()
    vt  = vuln_type.strip()

    # ── Full severity × type matrix ───────────────────────────────
    _MATRIX = {
        # ── Critical ──────────────────────────────────────────────
        ("Critical", "Remote Code Execution"): (
            "Apply emergency patch immediately. Isolate affected systems from the network. "
            "Check for active exploitation via threat-intelligence feeds. "
            "Enable application-level memory protections (ASLR, DEP/NX)."
        ),
        ("Critical", "SQL Injection"): (
            "Take the affected endpoint offline immediately. Deploy WAF rules to block "
            "malicious payloads. Audit all database access logs for data exfiltration. "
            "Rotate database credentials."
        ),
        ("Critical", "Buffer Overflow"): (
            "Apply vendor patch immediately. Enable ASLR/DEP system-wide. "
            "Monitor for anomalous process memory usage and consider isolating the service."
        ),
        ("Critical", "Authentication Bypass"): (
            "Revoke all active sessions and force a full password reset. "
            "Disable the vulnerable endpoint until patched. "
            "Audit authentication logs for unauthorized access."
        ),
        ("Critical", "Privilege Escalation"): (
            "Apply patch immediately. Audit all privilege assignments and "
            "review sudo/administrator group memberships for unauthorized changes."
        ),
        ("Critical", "Denial of Service"): (
            "Deploy rate-limiting and DDoS-mitigation controls immediately. "
            "Apply vendor patch. Activate redundancy/failover if available."
        ),
        ("Critical", "Cross-Site Scripting"): (
            "Deploy Content Security Policy headers immediately. "
            "Sanitize and encode all user-supplied output. "
            "Patch the affected component as an emergency."
        ),
        ("Critical", "Information Disclosure"): (
            "Restrict access to the affected endpoint immediately. "
            "Rotate exposed credentials, tokens, and API keys. "
            "Notify affected users and review data-retention policies."
        ),
        # ── High ──────────────────────────────────────────────────
        ("High", "Remote Code Execution"): (
            "Apply patch within 24 hours. Implement network segmentation around the "
            "affected service. Enable exploit-mitigation flags (ASLR, DEP/NX)."
        ),
        ("High", "SQL Injection"): (
            "Parameterize all database queries and sanitize inputs. "
            "Deploy WAF rules. Audit database logs for suspicious queries."
        ),
        ("High", "Buffer Overflow"): (
            "Apply vendor patch within 24 hours. Enable stack canaries. "
            "Restrict internet exposure of the affected service."
        ),
        ("High", "Authentication Bypass"): (
            "Patch within 24 hours. Enforce multi-factor authentication. "
            "Review authentication logs for unauthorized access attempts."
        ),
        ("High", "Privilege Escalation"): (
            "Apply patch within 24 hours. Audit user permissions. "
            "Review recent privilege changes in system logs."
        ),
        ("High", "Denial of Service"): (
            "Apply vendor patch within 24 hours. Implement rate-limiting. "
            "Monitor service availability and configure automated alerting."
        ),
        ("High", "Cross-Site Scripting"): (
            "Implement Content Security Policy headers and encode all user output. "
            "Apply patch within 24 hours."
        ),
        ("High", "Information Disclosure"): (
            "Restrict access to sensitive endpoints. Rotate any exposed credentials. "
            "Review and tighten access-control policies."
        ),
        # ── Medium ────────────────────────────────────────────────
        ("Medium", "Remote Code Execution"): (
            "Schedule patch in the next maintenance window. "
            "Apply network-level compensating controls in the interim."
        ),
        ("Medium", "SQL Injection"): (
            "Implement input validation and parameterized queries. "
            "Schedule patch deployment in the next sprint."
        ),
        ("Medium", "Cross-Site Scripting"): (
            "Implement Content Security Policy headers. "
            "Encode all user-supplied output in the next release cycle."
        ),
        ("Medium", "Privilege Escalation"): (
            "Apply patch in the next scheduled maintenance. "
            "Review and tighten privilege-assignment policies."
        ),
        ("Medium", "Denial of Service"): (
            "Apply patch in the next maintenance window. "
            "Implement basic rate-limiting as a compensating control."
        ),
        ("Medium", "Information Disclosure"): (
            "Review and restrict access to sensitive data endpoints. "
            "Apply patch in the next release cycle."
        ),
        ("Medium", "Authentication Bypass"): (
            "Enforce MFA where possible. Apply patch in next maintenance window."
        ),
        ("Medium", "Buffer Overflow"): (
            "Apply patch in the next maintenance window. "
            "Enable compiler-level protections (stack canaries, ASLR)."
        ),
        # ── Low ───────────────────────────────────────────────────
        ("Low", "Remote Code Execution"): (
            "Apply patch during routine maintenance. "
            "Confirm exploitation pre-conditions are absent in your environment."
        ),
        ("Low", "Information Disclosure"): (
            "Apply patch during routine maintenance. "
            "Review logging policies to limit sensitive-data exposure."
        ),
        ("Low", "Cross-Site Scripting"): (
            "Add Content Security Policy headers. Apply patch during routine maintenance."
        ),
        ("Low", "Denial of Service"): (
            "Apply patch during routine maintenance. Monitor service health dashboards."
        ),
        ("Low", "Privilege Escalation"): (
            "Apply patch during routine maintenance. "
            "Follow the principle of least privilege in user assignments."
        ),
        ("Low", "SQL Injection"): (
            "Apply patch during routine maintenance. "
            "Ensure all queries use parameterized statements."
        ),
        ("Low", "Buffer Overflow"): (
            "Apply patch during routine maintenance. "
            "Enable compile-time hardening flags."
        ),
        ("Low", "Authentication Bypass"): (
            "Apply patch during routine maintenance. "
            "Review authentication flows for weaker paths."
        ),
    }

    # ── Exact match ───────────────────────────────────────────────
    if (sev, vt) in _MATRIX:
        return _MATRIX[(sev, vt)]

    # ── Partial type match (handles aliases like 'XSS' → 'Cross-Site Scripting') ─
    for (s, v), action in _MATRIX.items():
        if s == sev and (v.lower() in vt.lower() or vt.lower() in v.lower()):
            return action

    # ── Severity-level defaults ───────────────────────────────────
    _DEFAULTS = {
        "Critical": (
            "Apply emergency patch immediately. Isolate affected systems. "
            "Escalate to the security team and monitor for active exploitation."
        ),
        "High": (
            "Apply patch within 24 hours. Implement compensating controls "
            "and monitor for exploitation attempts."
        ),
        "Medium": (
            "Schedule patching within the next maintenance window. "
            "Review compensating controls in the interim."
        ),
        "Low": (
            "Apply patch during routine maintenance. "
            "Assess impact in the context of your specific environment."
        ),
    }
    return _DEFAULTS.get(sev, "Assess impact and prioritize patching accordingly.")
