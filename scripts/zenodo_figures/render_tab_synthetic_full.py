#!/usr/bin/env python
"""Emit the full synthetic-recovery supplementary table (all metrics) as LaTeX.

Complements Table~\\ref{tab:synthetic_conditions} (topography column only) with the
matched-source time-course correlation, Amari index, and planted-source MIR that
the Results text references. Medians (IQR) across the ten seeds, by regime x
condition x method, matching the topography table's aggregation.

Run:  python render_tab_synthetic_full.py [--csv PATH]
Prints the LaTeX table to stdout.
"""
import argparse
import csv
import statistics as st
from pathlib import Path

DEF_CSV = Path(__file__).resolve().parents[1] / "mne_synthetic" / "results" / "v1_full_analysis" / "synthetic_long_all_metrics.csv"
METHODS = ["AMICA@3k", "AMICA@10k", "Picard", "Infomax", "FastICA"]
CONDS = [("clean", "clean"), ("noise", "noise"), ("noise_eog", "noise+EOG"),
         ("noise_ecg", "noise+ECG"), ("full", "full")]
REGIMES = ["Laplacian", "Mixture"]
# (csv column, label, value format, lower-is-better?)
METRICS = [
    ("gt_r_source_median", r"Matched-source time-course $|r|$ (higher is better)", "{:.3f}", False),
    ("gt_amari_index",     r"Amari index (lower is better)",                       "{:.3f}", True),
    ("gt_mir_vs_truth_kbits_s", r"Planted-source MIR, kbits/s (higher is better)", "{:.1f}", False),
]


def med_iqr(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    med = st.median(v)
    q1 = v[len(v) // 4]
    q3 = v[(3 * len(v)) // 4]
    return med, q3 - q1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEF_CSV)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.csv.open()))

    def cell(reg, cond, method, col):
        vals = [float(r[col]) for r in rows
                if r["regime"] == reg and r["condition"] == cond
                and r["method_label"] == method and r[col] not in ("", "nan")]
        return med_iqr(vals)

    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Full synthetic recovery metrics (medians across the ten "
             r"seeds; IQR in parentheses), complementing the topography column of "
             r"Table~\ref{tab:synthetic_conditions}. AMICA is shown at both "
             r"iteration caps; comparators at their default budgets. The "
             r"topography and Amari-distance rankings agree, but the source "
             r"time-course correlation is more favourable to the "
             r"decorrelation-based estimators, and the planted-source MIR does "
             r"not separate \texttt{amica-python} from FastICA.}")
    L.append(r"\label{tab:synthetic_full}")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{llccccc}")
    L.append(r"\toprule")
    L.append(r"Regime & Condition & AMICA@3k & AMICA@10k & Picard & Infomax & FastICA \\")
    for col, label, fmt, _lower in METRICS:
        L.append(r"\midrule")
        L.append(r"\multicolumn{7}{l}{\emph{" + label + r"}}\\")
        for reg in REGIMES:
            for ckey, clabel in CONDS:
                cells = []
                for m in METHODS:
                    r = cell(reg, ckey, m, col)
                    cells.append("---" if r is None else f"${fmt.format(r[0])}$ (${fmt.format(r[1])}$)")
                L.append(f"{reg} & {clabel} & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}%")
    L.append(r"}")
    L.append(r"\end{table}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
