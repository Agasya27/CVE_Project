"""
Compute confusion matrices for vulnerability-type and severity models.

This uses the same model artifacts and fallbacks as the Streamlit app:
- Vulnerability type: DistilBERT fine-tuned model if present, else regex fallback
- Severity: saved ML model if present, else rule-based fallback

Outputs:
  eval_outputs/confusion_matrices/
    - vuln_type_confusion_matrix.png
    - vuln_type_confusion_matrix_normalized.png
    - vuln_type_classification_report.txt
    - severity_confusion_matrix.png
    - severity_confusion_matrix_normalized.png
    - severity_classification_report.txt
    - predictions.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Cols:
    cve_id: str = "CVE ID"
    description: str = "Description"
    severity: str = "Severity"
    vuln_type: str = "Vulnerability_Type"
    cvss: str = "CVSS Score"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_dataset() -> pd.DataFrame:
    root = _repo_root()
    for fname in ["cve_cleaned.csv", "cve_with_keywords.csv", "cve_dataset.csv"]:
        p = root / "data" / fname
        if p.is_file():
            return pd.read_csv(p)
    raise FileNotFoundError("No dataset found in data/. Expected cve_cleaned.csv etc.")


def _severity_from_cvss(cvss) -> str:
    from utils.preprocessing import get_severity_label

    return get_severity_label(cvss)


def _load_bert_artifacts():
    """
    Load fine-tuned DistilBERT classifier artifacts if present.
    Returns (model, tokenizer, label_encoder) or (None, None, None).
    """
    root = _repo_root()
    model_dir = root / "models" / "bert_classifier"
    le_path = root / "models" / "label_encoder.joblib"
    safetensors = model_dir / "model.safetensors"
    if not (model_dir.is_dir() and le_path.is_file() and safetensors.is_file()):
        return None, None, None

    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        import torch

        tokenizer = DistilBertTokenizer.from_pretrained(str(model_dir))
        model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        label_encoder = joblib.load(le_path)
        return model, tokenizer, label_encoder
    except Exception:
        return None, None, None


def _bert_predict_one(text: str, model, tokenizer, label_encoder) -> str:
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
        outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
    return str(label_encoder.inverse_transform([pred_idx])[0])


def _predict_vuln_type(texts: list[str], *, enable_bert: bool) -> list[str]:
    """
    Predict vulnerability type.

    Note: importing torch/transformers can crash in some environments; therefore
    BERT evaluation is opt-in via `--enable-bert`.
    """
    if enable_bert:
        model, tokenizer, le = _load_bert_artifacts()
        if model is not None:
            return [_bert_predict_one(t, model, tokenizer, le) for t in texts]

    from utils.preprocessing import classify_vulnerability_type

    return [classify_vulnerability_type(t) for t in texts]


def _predict_severity(texts: list[str]) -> list[str]:
    from utils.model_utils import predict_severity

    preds = []
    for t in texts:
        label, _conf = predict_severity(t, cvss_score=None)
        preds.append(str(label))
    return preds


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig_w = max(7, int(0.45 * len(labels)))
    fig_h = max(6, int(0.45 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells (avoid clutter for large matrices)
    if len(labels) <= 15:
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}" if cm.dtype.kind == "f" else str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="eval_outputs/confusion_matrices")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-bert",
        action="store_true",
        help="If set, evaluate vuln-type predictions using the fine-tuned DistilBERT artifacts when present.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for quick runs (0 means no cap).",
    )
    args = parser.parse_args()

    df = _load_dataset()
    cols = Cols()
    for req in [cols.cve_id, cols.description]:
        if req not in df.columns:
            raise ValueError(f"Dataset missing required column: {req}")

    work = df.copy()
    work[cols.description] = work[cols.description].fillna("").astype(str)

    # Ground truth labels (derive if missing)
    if cols.severity not in work.columns or work[cols.severity].isna().all():
        if cols.cvss not in work.columns:
            raise ValueError("Cannot derive Severity: missing `CVSS Score` column.")
        work[cols.severity] = work[cols.cvss].apply(_severity_from_cvss)
    if cols.vuln_type not in work.columns or work[cols.vuln_type].isna().all():
        from utils.preprocessing import classify_vulnerability_type

        work[cols.vuln_type] = work[cols.description].apply(classify_vulnerability_type)

    # Filter empty/unknown where evaluation would be meaningless
    work = work[work[cols.description].str.len() > 0].copy()
    work = work[work[cols.severity].astype(str).ne("Unknown")].copy()

    if args.max_rows and args.max_rows > 0:
        work = work.sample(n=min(args.max_rows, len(work)), random_state=args.seed).reset_index(
            drop=True
        )

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    out_dir = (_repo_root() / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Vulnerability Type ─────────────────────────────────────────────
    y_true_vt = work[cols.vuln_type].astype(str).tolist()
    X_text = work[cols.description].tolist()
    _, X_test, _, y_test = train_test_split(
        X_text,
        y_true_vt,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_true_vt,
    )
    y_pred = _predict_vuln_type(X_test, enable_bert=args.enable_bert)

    labels_vt = sorted(list(set(y_test) | set(y_pred)))
    cm_vt = confusion_matrix(y_test, y_pred, labels=labels_vt)
    cm_vt_norm = confusion_matrix(y_test, y_pred, labels=labels_vt, normalize="true")

    _plot_confusion_matrix(
        cm_vt, labels_vt, out_dir / "vuln_type_confusion_matrix.png", "Vulnerability Type"
    )
    _plot_confusion_matrix(
        cm_vt_norm,
        labels_vt,
        out_dir / "vuln_type_confusion_matrix_normalized.png",
        "Vulnerability Type (row-normalized)",
    )
    (out_dir / "vuln_type_classification_report.txt").write_text(
        classification_report(y_test, y_pred, labels=labels_vt, zero_division=0),
        encoding="utf-8",
    )

    # ── Severity ────────────────────────────────────────────────────
    y_true_sev = work[cols.severity].astype(str).tolist()
    _, X_test_s, _, y_test_s = train_test_split(
        X_text,
        y_true_sev,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_true_sev,
    )
    y_pred_s = _predict_severity(X_test_s)

    labels_sev = ["Low", "Medium", "High", "Critical"]
    cm_s = confusion_matrix(y_test_s, y_pred_s, labels=labels_sev)
    cm_s_norm = confusion_matrix(y_test_s, y_pred_s, labels=labels_sev, normalize="true")

    _plot_confusion_matrix(cm_s, labels_sev, out_dir / "severity_confusion_matrix.png", "Severity")
    _plot_confusion_matrix(
        cm_s_norm,
        labels_sev,
        out_dir / "severity_confusion_matrix_normalized.png",
        "Severity (row-normalized)",
    )
    (out_dir / "severity_classification_report.txt").write_text(
        classification_report(y_test_s, y_pred_s, labels=labels_sev, zero_division=0),
        encoding="utf-8",
    )

    # ── Save a compact predictions artifact for error analysis ──────
    pred_df = pd.DataFrame(
        {
            "task": (["vuln_type"] * len(y_test)) + (["severity"] * len(y_test_s)),
            "y_true": y_test + y_test_s,
            "y_pred": y_pred + y_pred_s,
        }
    )
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    print(f"Wrote confusion matrices + reports to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

