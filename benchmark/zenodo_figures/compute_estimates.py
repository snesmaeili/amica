"""Stage-2 contrast-level estimates for the real-EEG benchmark (preprint revision).

From the committed subject-level CSVs, compute the prespecified paired contrasts that feed
Main Figure 3 (MIR + remnant-PMI + dipolarity) and Main Table 3 (exact paired MIR effects):
for each of the 9 dataset x comparator cells, the paired mean difference (amica-python −
comparator), 95% bootstrap CI, Cohen's d_z (+CI), paired-t and Wilcoxon p, Holm-adjusted p
(MIR family across all 9), and the same-sign count. AMICA = JAX-GPU (the only AMICA backend
present in all three datasets; backends are numerically equivalent — backend-parity layer).

Writes `estimates_realeeg.csv` (long form: dataset, comparator, metric, ...). No new fits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

import style

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "cc_benchmark" / "results"
DATASETS = {
    "ds004505": RESULTS / "v3_paper_stage1_cluster" / "benchmark_results.csv",
    "ds004504": RESULTS / "ds004504_v3" / "benchmark_results.csv",
    "ds004621": RESULTS / "ds004621_v3" / "benchmark_results.csv",
}
AMICA = "AMICA-Python (JAX-GPU)"
COMPARATORS = ["Picard", "Infomax", "FastICA"]
# higher-is-better sign per metric (so the reported diff is amica − comparator in raw units;
# `favours_amica` flips it where lower is better, for interpretation only).
METRICS = {
    "mir_kbits_s":        +1,   # higher better
    "remnant_pmi_percent": -1,  # lower better (positive diff = amica worse)
    "nd_5_percent":       +1,   # higher better
    "nd_10_percent":      +1,
}


def _paired(amica, comp):
    """Align two subject-indexed Series; return paired diff (amica − comp) over common subjects."""
    j = pd.concat([amica.rename("a"), comp.rename("c")], axis=1).dropna()
    return (j["a"] - j["c"]).to_numpy(), len(j)


def main():
    rows = []
    for ds, path in DATASETS.items():
        df = pd.read_csv(path)
        df = df[["subject", "method", *METRICS]].copy()
        for metric, hib in METRICS.items():
            a = df[df.method == AMICA].set_index("subject")[metric]
            for comp in COMPARATORS:
                c = df[df.method == comp].set_index("subject")[metric]
                diff, n = _paired(a, c)
                if n < 3 or not np.isfinite(diff).any():
                    continue
                mean, lo, hi = style.bootstrap_ci(diff, statistic=np.mean)
                dz, dzlo, dzhi = style.cohens_dz(diff)
                t_p = float(sstats.ttest_rel(diff, np.zeros_like(diff)).pvalue)
                try:
                    w_p = float(sstats.wilcoxon(diff).pvalue)
                except ValueError:
                    w_p = np.nan
                favours = int(np.sum(np.sign(diff) == np.sign(hib)))  # subjects in the better direction
                rows.append(dict(dataset=ds, comparator=comp, metric=metric, n=n,
                                 mean_diff=mean, ci_lo=lo, ci_hi=hi, dz=dz, dz_lo=dzlo, dz_hi=dzhi,
                                 t_p=t_p, wilcoxon_p=w_p, favours_amica=favours, higher_is_better=hib))
    out = pd.DataFrame(rows)
    # Holm across the 9 MIR contrasts (one family); secondary metrics reported without
    # cross-metric correction (per-metric exploratory), Holm within each metric family of 9.
    out["holm_p"] = np.nan
    for metric in METRICS:
        m = out.metric == metric
        out.loc[m, "holm_p"] = style.holm(out.loc[m, "t_p"].to_numpy())
    out_path = HERE / "estimates_realeeg.csv"
    out.to_csv(out_path, index=False)

    # console summary of the headline MIR family
    mir = out[out.metric == "mir_kbits_s"].copy()
    print(f"wrote {out_path}  ({len(out)} contrast rows; {len(mir)} MIR)")
    print("\n=== MIR contrasts (amica − comparator, kbit/s) ===")
    for _, r in mir.iterrows():
        print(f"  {r.dataset} vs {r.comparator:8}: Δ={r.mean_diff:+.3f} "
              f"[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]  d_z={r.dz:.2f}  Holm p={r.holm_p:.1e}  "
              f"favours {int(r.favours_amica)}/{int(r.n)}")
    sec = out[out.metric != "mir_kbits_s"]
    print("\n=== secondary (mean paired diff) ===")
    for metric in ("remnant_pmi_percent", "nd_5_percent", "nd_10_percent"):
        s = sec[sec.metric == metric]
        print(f"  {metric:20}: " + "  ".join(f"{r.dataset}/{r.comparator[:4]} {r.mean_diff:+.2f}"
                                              for _, r in s.iterrows()))


if __name__ == "__main__":
    main()
