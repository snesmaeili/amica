#!/bin/bash
# Fetch the ds004505 EEG dataset (public, OpenNeuro, PDDL license) for the benchmark.
#
# Usage (on a node/login shell with internet; ~10 GB of EEG):
#   cp env.template env.local   # edit BIDS_ROOT_DS4505 to where you want the data
#   module load git-annex       # Alliance clusters; or have datalad/git-annex installed
#   bash download_ds004505.sh
#
# Default path (git-annex / GitHub mirror) gives the standard BIDS layout
# (sub-NN/eeg/*.set) -> keep AMICA_INPUT_LEVEL=bids (what the paper used).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -f env.local ] && source env.local
: "${BIDS_ROOT_DS4505:?set BIDS_ROOT_DS4505 in env.local}"
mkdir -p "$(dirname "$BIDS_ROOT_DS4505")"

echo "Fetching ds004505 -> $BIDS_ROOT_DS4505 (BIDS layout; use AMICA_INPUT_LEVEL=bids)"
if command -v datalad >/dev/null 2>&1; then
    datalad install -s https://github.com/OpenNeuroDatasets/ds004505.git "$BIDS_ROOT_DS4505"
    datalad get -d "$BIDS_ROOT_DS4505" "$BIDS_ROOT_DS4505"/sub-*/eeg/*_eeg.set "$BIDS_ROOT_DS4505"/sub-*/eeg/*_eeg.fdt
elif command -v git-annex >/dev/null 2>&1; then
    git clone https://github.com/OpenNeuroDatasets/ds004505.git "$BIDS_ROOT_DS4505"
    ( cd "$BIDS_ROOT_DS4505" && git annex get --jobs=4 sub-*/eeg/*_eeg.set sub-*/eeg/*_eeg.fdt )
else
    echo "datalad/git-annex not found. Fallback (gives sourcedata/Merged layout -> set AMICA_INPUT_LEVEL=merged):"
    echo "  pip install openneuro-py"
    echo "  python -c \"import openneuro; openneuro.download('ds004505', target_dir='$BIDS_ROOT_DS4505')\""
    exit 1
fi

n=$(find -L "$BIDS_ROOT_DS4505" -path '*/eeg/*_eeg.set' -type f 2>/dev/null | wc -l)
echo "Done. Materialized eeg .set files: $n (expect 25)."
