"""
Text preprocessing utilities for CVE descriptions.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Custom security-domain stopwords to keep (these are meaningful in CVE context)
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


def clean_text(text):
    """Basic text cleaning for CVE descriptions."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove CVE IDs from text body
    text = re.sub(r'cve-\d{4}-\d+', '', text)
    # Remove version numbers but keep context
    text = re.sub(r'\b\d+\.\d+\.\d+[\.\d]*\b', 'VERSION', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_text(text):
    """Tokenize text into words."""
    return word_tokenize(text)


def remove_stopwords(tokens):
    """Remove stopwords while preserving security-relevant terms."""
    stop_words = set(stopwords.words('english'))
    # Remove security terms from stopwords so they are preserved
    stop_words -= SECURITY_TERMS_TO_KEEP
    return [t for t in tokens if t not in stop_words and len(t) > 2]


def lemmatize_tokens(tokens):
    """Lemmatize tokens."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(text):
    """Full preprocessing pipeline for CVE descriptions."""
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)


def extract_software_name(description):
    """Extract likely software names from CVE descriptions."""
    if not isinstance(description, str):
        return "Unknown"
    # Common patterns: "in <Software> <version>" or "<Software> plugin"
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


def classify_vulnerability_type(description):
    """Rule-based vulnerability type classification from description text."""
    if not isinstance(description, str):
        return "Other"
    desc_lower = description.lower()
    type_patterns = {
        'Cross-Site Scripting (XSS)': ['cross-site scripting', 'xss', 'stored xss', 'reflected xss'],
        'SQL Injection': ['sql injection', 'sqli'],
        'Remote Code Execution': ['remote code execution', 'rce', 'execute arbitrary code', 'code execution'],
        'Denial of Service': ['denial of service', 'dos', 'denial-of-service', 'crash', 'resource exhaustion'],
        'Buffer Overflow': ['buffer overflow', 'heap overflow', 'stack overflow', 'out-of-bounds write'],
        'Privilege Escalation': ['privilege escalation', 'escalate privileges', 'gain elevated'],
        'Information Disclosure': ['information disclosure', 'information leak', 'sensitive information',
                                    'data exposure', 'data leak'],
        'Authentication Bypass': ['authentication bypass', 'bypass authentication', 'unauthorized access'],
        'Path Traversal': ['path traversal', 'directory traversal', '../'],
        'CSRF': ['cross-site request forgery', 'csrf'],
        'SSRF': ['server-side request forgery', 'ssrf'],
        'Command Injection': ['command injection', 'os command', 'arbitrary command'],
    }
    for vuln_type, keywords in type_patterns.items():
        if any(kw in desc_lower for kw in keywords):
            return vuln_type
    return "Other"


def get_severity_label(cvss_score):
    """Convert CVSS score to severity category."""
    try:
        score = float(cvss_score)
    except (ValueError, TypeError):
        return "Unknown"
    if score == 0.0:
        return "None"
    elif score <= 3.9:
        return "Low"
    elif score <= 6.9:
        return "Medium"
    elif score <= 8.9:
        return "High"
    else:
        return "Critical"
