# Regenerate every zenodo-preprint figure that this directory owns.
#
# Reads:
#   ..\mne_synthetic\results\v1_full_analysis\synthetic_long_all_metrics.csv
#   ..\cc_benchmark\results\v3_paper_stage1_cluster\benchmark_results.csv
# Writes:
#   .\figures\fig_synthetic_recovery.pdf       (+ _stats.csv)
#   .\figures\fig_mir_combined.pdf             (+ _stats.csv)
#   .\figures\fig_quality_cost.pdf             (+ _stats.csv)
#
# To target a custom output dir (e.g. an Overleaf clone):
#   $env:OUT_DIR = "D:\overleaf_repos\<id>\figures"
#   .\make_all.ps1

$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $env:OUT_DIR) {
    $OutDir = Join-Path $Here 'figures'
} else {
    $OutDir = $env:OUT_DIR
}
if (-not $env:PYTHON) { $Python = 'python' } else { $Python = $env:PYTHON }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Output "[zenodo_figures] writing to $OutDir"

& $Python (Join-Path $Here 'render_fig_synthetic_recovery.py') `
    --out (Join-Path $OutDir 'fig_synthetic_recovery.pdf')
if (-not $?) { throw 'synthetic_recovery failed' }

& $Python (Join-Path $Here 'render_fig_mir_combined.py') `
    --out (Join-Path $OutDir 'fig_mir_combined.pdf')
if (-not $?) { throw 'mir_combined failed' }

& $Python (Join-Path $Here 'render_fig_runtime_combined.py') `
    --out (Join-Path $OutDir 'fig_quality_cost.pdf')
if (-not $?) { throw 'runtime_combined failed' }

Write-Output '[zenodo_figures] DONE'
Get-ChildItem (Join-Path $OutDir 'fig_synthetic_recovery.pdf'), `
              (Join-Path $OutDir 'fig_mir_combined.pdf'), `
              (Join-Path $OutDir 'fig_quality_cost.pdf')
