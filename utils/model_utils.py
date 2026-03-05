"""
Model utility functions for CVE Intelligence Analyzer.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


def save_model(model, filename):
    """Save a model to the models directory."""
    MODELS_DIR.mkdir(exist_ok=True)
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    return filepath


def load_model(filename):
    """Load a model from the models directory."""
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    return joblib.load(filepath)


def save_dataframe(df, filename):
    """Save a DataFrame to the data directory."""
    DATA_DIR.mkdir(exist_ok=True)
    filepath = DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"DataFrame saved to {filepath}")
    return filepath


def load_dataframe(filename):
    """Load a DataFrame from the data directory."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    return pd.read_csv(filepath)


def generate_alert(description, vuln_type, severity, summary):
    """Generate a structured security alert from analysis results."""
    severity_actions = {
        'Critical': 'Patch immediately. Isolate affected systems.',
        'High': 'Apply patches within 24 hours. Monitor for exploitation.',
        'Medium': 'Schedule patching within the next maintenance window.',
        'Low': 'Apply patches during routine maintenance.',
        'Unknown': 'Assess impact and prioritize accordingly.',
    }
    action = severity_actions.get(severity, severity_actions['Unknown'])

    alert = {
        'severity': severity,
        'type': vuln_type,
        'summary': summary,
        'recommended_action': action,
        'original_description': description,
    }
    return alert


def format_alert_text(alert):
    """Format an alert dictionary into readable text."""
    lines = [
        f"🔴 CVE Alert: {alert['severity']} {alert['type']}",
        f"",
        f"Summary: {alert['summary']}",
        f"",
        f"Recommended Action: {alert['recommended_action']}",
    ]
    return '\n'.join(lines)
