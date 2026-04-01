"""
Text preprocessing utilities for CVE descriptions.

Includes:
- clean_text()               Basic cleaning (lowercase, strip URLs, etc.)
- tokenize_text()            Word tokenisation
- remove_stopwords()         Remove stop-words while keeping security terms
- lemmatize_tokens()         WordNet lemmatisation
- preprocess_text()          Full pipeline (clean → tokenise → filter → lemmatise)
- extract_software_name()    Heuristic software-name extractor
- extract_keywords()         TF-IDF keyword extraction (top-N)
- get_vulnerability_type()   Regex-based vulnerability classifier (alias)
- classify_vulnerability_type()  Regex-based vulnerability classifier
- get_severity_label()       CVSS score → severity label
"""

import re
import logging
from pathlib import Path

import nltk

logger = logging.getLogger(__name__)

# Download required NLTK data silently on first import
for _resource in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4'):
    try:
        nltk.download(_resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords as _nltk_stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── Security-domain terms that must survive stop-word removal ─────────────────
SECURITY_TERMS_TO_KEEP = {
    'remote', 'local', 'arbitrary', 'execute', 'execution', 'overflow',
    'injection', 'denial', 'service', 'cross', 'site', 'scripting',
    'privilege', 'escalation', 'bypass', 'authentication', 'unauthorized',
    'access', 'disclosure', 'information', 'buffer', 'memory', 'corruption',
    'code', 'command', 'sql', 'xss', 'csrf', 'ssrf', 'rce', 'dos',
    'vulnerability', 'exploit', 'attack', 'attacker', 'malicious',
    'crafted', 'request', 'input', 'user', 'server', 'client',
    'sensitive', 'data', 'file', 'path', 'traversal', 'upload',
    'download', 'read', 'write', 'delete', 'modify', 'overwrite',
}

# ── Vulnerability type patterns (order matters — first match wins) ────────────
_VULN_TYPE_PATTERNS = {
    'Cross-Site Scripting (XSS)': [
        'cross-site scripting', ' xss ', 'stored xss', 'reflected xss', 'dom xss',
    ],
    'SQL Injection': ['sql injection', 'sqli'],
    'Remote Code Execution': [
        'remote code execution', ' rce ', 'execute arbitrary code', 'code execution',
        'arbitrary code execution',
    ],
    'Denial of Service': [
        'denial of service', ' dos ', 'denial-of-service', 'resource exhaustion',
    ],
    'Buffer Overflow': [
        'buffer overflow', 'heap overflow', 'stack overflow', 'out-of-bounds write',
        'heap-based buffer', 'stack-based buffer',
    ],
    'Privilege Escalation': [
        'privilege escalation', 'escalate privileges', 'gain elevated', 'local privilege',
    ],
    'Information Disclosure': [
        'information disclosure', 'information leak', 'sensitive information',
        'data exposure', 'data leak', 'memory disclosure',
    ],
    'Authentication Bypass': [
        'authentication bypass', 'bypass authentication', 'unauthorized access',
        'unauthenticated', 'improper authentication',
    ],
    'Path Traversal': ['path traversal', 'directory traversal', '../'],
    'CSRF': ['cross-site request forgery', 'csrf'],
    'SSRF': ['server-side request forgery', 'ssrf'],
    'Command Injection': [
        'command injection', 'os command', 'arbitrary command', 'shell injection',
    ],
}


# ── Core cleaning ─────────────────────────────────────────────────────────────

def clean_text(text) -> str:
    """
    Basic text cleaning for CVE descriptions.

    Handles None, NaN, and empty strings safely.  Lowercases, removes URLs,
    CVE identifiers, version numbers, and non-alphabetic characters.

    Args:
        text: Raw input — may be str, float (NaN), or None.

    Returns:
        Cleaned string, or "" if input is empty/null.
    """
    if text is None:
        return ""
    # Handle NaN / numeric types from pandas
    try:
        import math
        if isinstance(text, float) and math.isnan(text):
            return ""
    except Exception:
        pass
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove CVE identifiers from the body
    text = re.sub(r'cve-\d{4}-\d+', '', text, flags=re.IGNORECASE)
    # Anonymise version numbers
    text = re.sub(r'\b\d+\.\d+[\.\d]*\b', 'VERSION', text)
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_text(text: str) -> list:
    """Tokenise a cleaned string into a list of word tokens."""
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        return word_tokenize(text)
    except LookupError:
        # Fallback if punkt data is unavailable
        return text.split()


def remove_stopwords(tokens: list) -> list:
    """
    Remove English stop-words while preserving security-domain terms.

    Args:
        tokens: List of lowercase string tokens.

    Returns:
        Filtered token list.
    """
    try:
        stop_words = set(_nltk_stopwords.words('english'))
    except LookupError:
        stop_words = set()
    # Never remove security-relevant terms
    stop_words -= SECURITY_TERMS_TO_KEEP
    return [t for t in tokens if t not in stop_words and len(t) > 2]


def lemmatize_tokens(tokens: list) -> list:
    """Lemmatise a list of tokens using WordNetLemmatizer."""
    lem = WordNetLemmatizer()
    return [lem.lemmatize(t) for t in tokens]


def preprocess_text(text) -> str:
    """
    Full NLP preprocessing pipeline for CVE descriptions.

    Pipeline: clean → tokenise → remove stop-words → lemmatise → rejoin.

    Args:
        text: Raw CVE description (str, float, or None).

    Returns:
        Preprocessed string ready for vectorisation.
    """
    text = clean_text(text)
    if not text:
        return ""
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)


# ── Keyword extraction ────────────────────────────────────────────────────────

def extract_keywords(text, top_n: int = 10) -> list:
    """
    Extract the top-N keywords from *text* using TF-IDF scoring.

    If the project's saved TF-IDF vectorizer (``models/tfidf_vectorizer.joblib``)
    is available it is used for proper IDF weighting.  Otherwise falls back to a
    simple term-frequency approach so the function never raises.

    Args:
        text:   Raw CVE description.
        top_n:  Maximum number of keywords to return.

    Returns:
        List of ``(keyword, score)`` tuples, sorted descending by score.
        Returns an empty list if text is empty or null.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    vectorizer_path = Path(__file__).parent.parent / "models" / "tfidf_vectorizer.joblib"
    if vectorizer_path.exists():
        try:
            import joblib
            vectorizer = joblib.load(vectorizer_path)
            cleaned = preprocess_text(text)
            tfidf_vec = vectorizer.transform([cleaned])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_vec.toarray().flatten()
            top_indices = scores.argsort()[::-1][:top_n]
            return [
                (feature_names[i], float(scores[i]))
                for i in top_indices
                if scores[i] > 0
            ]
        except Exception as exc:
            logger.warning("TF-IDF vectorizer failed: %s — using term-frequency fallback", exc)

    # ── Term-frequency fallback ───────────────────────────────────
    from collections import Counter
    _STOP = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'before',
        'after', 'that', 'this', 'these', 'those', 'it', 'its', 'and', 'but',
        'or', 'not', 'also', 'just', 'only', 'all', 'some', 'any', 'via',
        'allows', 'using', 'used', 'use', 'due', 'which', 'when', 'where',
    }
    words = re.findall(r'[a-z]{3,}', text.lower())
    words = [w for w in words if w not in _STOP]
    total = max(len(words), 1)
    counts = Counter(words)
    return [(w, c / total) for w, c in counts.most_common(top_n)]


# ── Vulnerability type classification ─────────────────────────────────────────

def classify_vulnerability_type(description) -> str:
    """
    Rule-based vulnerability type classifier using regex pattern matching.

    Checks the description against known CVE keyword patterns.  First match
    wins.  Returns ``'Other'`` when no pattern matches.

    Args:
        description: Raw CVE description string.

    Returns:
        Vulnerability type label string.
    """
    if not isinstance(description, str) or not description.strip():
        return "Other"
    # Pad with spaces so boundary patterns work without \b issues
    d = ' ' + description.lower() + ' '
    for vuln_type, keywords in _VULN_TYPE_PATTERNS.items():
        if any(kw in d for kw in keywords):
            return vuln_type
    return "Other"


def get_vulnerability_type(text) -> str:
    """
    Alias for :func:`classify_vulnerability_type`.

    Provided for API consistency with ``model_utils.classify_vulnerability``.
    """
    return classify_vulnerability_type(text)


# ── Severity helper ───────────────────────────────────────────────────────────

def get_severity_label(cvss_score) -> str:
    """
    Convert a CVSS score to a human-readable severity category.

    Mapping: 0-3.9 → Low, 4-6.9 → Medium, 7-8.9 → High, 9-10 → Critical.

    Args:
        cvss_score: Numeric CVSS score (0–10) or string representation.

    Returns:
        Severity label string; ``'Unknown'`` if conversion fails.
    """
    try:
        score = float(cvss_score)
    except (ValueError, TypeError):
        return "Unknown"
    if score < 4.0:
        return "Low"
    if score < 7.0:
        return "Medium"
    if score < 9.0:
        return "High"
    return "Critical"


# ── Software name extractor ───────────────────────────────────────────────────

def extract_software_name(description) -> str:
    """
    Heuristically extract the primary affected software name from a CVE description.

    Uses regex patterns that look for capitalised product names adjacent to
    version qualifiers or product-type words (plugin, extension, module).

    Args:
        description: Raw CVE description string.

    Returns:
        Extracted software name, or ``'Unknown'``.
    """
    if not isinstance(description, str) or not description.strip():
        return "Unknown"
    patterns = [
        r'(?:in|of|for)\s+([A-Z][a-zA-Z0-9\s]+?)(?:\s+(?:before|through|prior|version|v\d))',
        r'^([A-Z][a-zA-Z0-9\s]+?)(?:\s+(?:before|through|prior|version|v\d))',
        r'(?:The\s+)?([A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)?)\s+(?:plugin|extension|module|component)',
    ]
    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            return match.group(1).strip()
    return "Unknown"
