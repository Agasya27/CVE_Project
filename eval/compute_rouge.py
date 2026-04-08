"""
Compute ROUGE-1/2/L between model outputs and human references.

Typical usage:
  1) Create annotation template:
     python -m eval.make_reference_template --n 200 --stratify
  2) Fill `data/cve_references.csv` (Reference_Summary required)
  3) Compute ROUGE:
     python -m eval.compute_rouge --candidates data/cve_summarized.csv --references data/cve_references.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RougeResult:
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float


def _normalize_text(s: str) -> str:
    """Light normalization for ROUGE robustness (keep words, normalize whitespace)."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_keywords(s: str) -> str:
    """
    Turn a comma-separated keyword list into a whitespace-separated token string.
    This makes ROUGE-1/2 behave like keyword/phrase overlap.
    """
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    parts = [p.strip().lower() for p in s.split(",")]
    parts = [p for p in parts if p]
    # split multi-word phrases into tokens but keep order within phrase
    tokens: list[str] = []
    for p in parts:
        tokens.extend(p.split())
    return " ".join(tokens)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def _compute_rouge_scores(cands: list[str], refs: list[str]) -> tuple[pd.DataFrame, RougeResult]:
    try:
        from rouge_score import rouge_scorer
        from rouge_score import scoring
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `rouge-score`. Install with `pip install rouge-score`."
        ) from exc

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    rows = []
    for cand, ref in zip(cands, refs):
        scores = scorer.score(ref, cand)
        r1 = scores["rouge1"]
        r2 = scores["rouge2"]
        rl = scores["rougeL"]
        rows.append(
            {
                "rouge1_precision": r1.precision,
                "rouge1_recall": r1.recall,
                "rouge1_fmeasure": r1.fmeasure,
                "rouge2_precision": r2.precision,
                "rouge2_recall": r2.recall,
                "rouge2_fmeasure": r2.fmeasure,
                "rougeL_precision": rl.precision,
                "rougeL_recall": rl.recall,
                "rougeL_fmeasure": rl.fmeasure,
            }
        )
        aggregator.add_scores(scores)

    agg = aggregator.aggregate()
    corpus = RougeResult(
        rouge1_f=float(agg["rouge1"].mid.fmeasure),
        rouge2_f=float(agg["rouge2"].mid.fmeasure),
        rougeL_f=float(agg["rougeL"].mid.fmeasure),
    )
    return pd.DataFrame(rows), corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=str,
        default="data/cve_summarized.csv",
        help="CSV containing model outputs (expects `CVE ID` and `Summary` by default).",
    )
    parser.add_argument(
        "--references",
        type=str,
        default="data/cve_references.csv",
        help="CSV containing human references (expects `CVE ID` and `Reference_Summary`).",
    )
    parser.add_argument(
        "--candidate-col",
        type=str,
        default="Summary",
        help="Candidate text column to evaluate (e.g., Summary or Alert).",
    )
    parser.add_argument(
        "--reference-col",
        type=str,
        default="Reference_Summary",
        help="Reference text column to evaluate against.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["summary", "keywords"],
        default="summary",
        help="`summary` uses text normalization; `keywords` expects comma-separated keyword lists.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="eval_outputs/rouge",
        help="Directory to write outputs (per-sample CSV and summary txt).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cand_path = (repo_root / args.candidates).resolve()
    ref_path = (repo_root / args.references).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cands_df = _load_csv(cand_path)
    refs_df = _load_csv(ref_path)

    for req in ["CVE ID", args.candidate_col]:
        if req not in cands_df.columns:
            raise ValueError(f"Candidates file missing column: {req}")
    for req in ["CVE ID", args.reference_col]:
        if req not in refs_df.columns:
            raise ValueError(f"References file missing column: {req}")

    joined = refs_df[["CVE ID", args.reference_col]].merge(
        cands_df[["CVE ID", args.candidate_col]],
        on="CVE ID",
        how="inner",
    )
    joined = joined.dropna(subset=[args.reference_col]).copy()
    joined[args.reference_col] = joined[args.reference_col].astype(str)
    joined[args.candidate_col] = joined[args.candidate_col].fillna("").astype(str)

    if joined.empty:
        raise ValueError(
            "No rows to score after joining candidates and references. "
            "Check `CVE ID` overlap and that Reference_Summary is filled."
        )

    if args.mode == "keywords":
        ref_texts = [_normalize_keywords(x) for x in joined[args.reference_col].tolist()]
        cand_texts = [_normalize_keywords(x) for x in joined[args.candidate_col].tolist()]
    else:
        ref_texts = [_normalize_text(x) for x in joined[args.reference_col].tolist()]
        cand_texts = [_normalize_text(x) for x in joined[args.candidate_col].tolist()]

    per_df, corpus = _compute_rouge_scores(cand_texts, ref_texts)
    out = pd.concat([joined.reset_index(drop=True), per_df], axis=1)

    per_path = out_dir / "per_sample_rouge.csv"
    out.to_csv(per_path, index=False)

    summary = (
        f"Rows_scored: {len(out)}\n"
        f"ROUGE-1_F1 (keyword): {corpus.rouge1_f:.4f}\n"
        f"ROUGE-2_F1 (phrase):  {corpus.rouge2_f:.4f}\n"
        f"ROUGE-L_F1 (struct):  {corpus.rougeL_f:.4f}\n"
        f"Candidate_col: {args.candidate_col}\n"
        f"Reference_col: {args.reference_col}\n"
        f"Mode: {args.mode}\n"
    )
    summary_path = out_dir / "rouge_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print(summary.strip())
    print(f"Wrote: {per_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

