"""
Create a human-annotation template for ROUGE evaluation.

This script samples CVEs from the dataset (optionally stratified) and writes a
CSV you can fill with reference summaries/keywords.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Columns:
    cve_id: str = "CVE ID"
    description: str = "Description"
    severity: str = "Severity"
    vuln_type: str = "Vulnerability_Type"


def _read_any_csv(paths: list[Path]) -> pd.DataFrame:
    for p in paths:
        if p.is_file():
            return pd.read_csv(p)
    raise FileNotFoundError(
        "Could not find any dataset CSV. Tried: " + ", ".join(str(p) for p in paths)
    )


def _stratified_sample(
    df: pd.DataFrame, n: int, strata_cols: list[str], seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Allocate roughly proportional samples per stratum (minimum 1 if possible).
    strata = df.groupby(strata_cols, dropna=False)
    sizes = strata.size().reset_index(name="_count")
    sizes["_target"] = (sizes["_count"] / sizes["_count"].sum() * n).round().astype(int)
    sizes.loc[sizes["_target"] == 0, "_target"] = 1

    # If we overshoot due to rounding/min-1, trim targets deterministically.
    overshoot = int(sizes["_target"].sum() - n)
    if overshoot > 0:
        # Reduce from largest targets first.
        sizes = sizes.sort_values("_target", ascending=False).reset_index(drop=True)
        for i in range(len(sizes)):
            if overshoot <= 0:
                break
            if sizes.loc[i, "_target"] > 1:
                sizes.loc[i, "_target"] -= 1
                overshoot -= 1

    parts: list[pd.DataFrame] = []
    for _, row in sizes.iterrows():
        mask = pd.Series(True, index=df.index)
        for c in strata_cols:
            mask &= df[c].fillna("__nan__").astype(str).eq(
                str(row[c]) if pd.notna(row[c]) else "__nan__"
            )
        group = df[mask]
        k = int(min(row["_target"], len(group)))
        if k <= 0:
            continue
        take_idx = rng.choice(group.index.to_numpy(), size=k, replace=False)
        parts.append(df.loc[take_idx])

    sampled = pd.concat(parts, ignore_index=True).drop_duplicates(subset=[Columns().cve_id])
    # If duplicates/empty strata trimming undershot, top up uniformly.
    if len(sampled) < n:
        remaining = df[~df[Columns().cve_id].isin(sampled[Columns().cve_id])]
        topup = remaining.sample(n=n - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, topup], ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of CVEs to sample.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible sampling."
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratify by Severity and Vulnerability_Type (recommended).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/cve_references.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _read_any_csv(
        [
            repo_root / "data" / "cve_cleaned.csv",
            repo_root / "data" / "cve_with_keywords.csv",
            repo_root / "data" / "cve_dataset.csv",
        ]
    )

    cols = Columns()
    missing = [c for c in [cols.cve_id, cols.description] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    use = df.copy()
    for c in [cols.severity, cols.vuln_type]:
        if c not in use.columns:
            use[c] = np.nan

    if args.stratify:
        sampled = _stratified_sample(use, args.n, [cols.severity, cols.vuln_type], args.seed)
    else:
        sampled = use.sample(n=min(args.n, len(use)), random_state=args.seed).reset_index(
            drop=True
        )

    template = pd.DataFrame(
        {
            "CVE ID": sampled[cols.cve_id].astype(str),
            "Reference_Summary": "",
            "Reference_Keywords": "",
            "Notes": "",
            "Source_Description": sampled[cols.description].fillna("").astype(str),
            "Severity": sampled[cols.severity].astype(str),
            "Vulnerability_Type": sampled[cols.vuln_type].astype(str),
        }
    )

    template.to_csv(out_path, index=False)
    print(f"Wrote reference template: {out_path}  (rows={len(template)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

