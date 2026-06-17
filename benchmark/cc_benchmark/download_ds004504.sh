#!/bin/bash
# Fetch the ds004504 EEG dataset (public, OpenNeuro, CC0) for the MIR replication.
# Miltiadous et al. 2023 (MDPI Data 8:95) -- AD/FTD/HC eyes-closed resting EEG; the
# benchmark uses the healthy-control subjects (sub-037..sub-065).
#
# Usage (login/compute shell with internet; small, a few hundred MB):
#   cp env.template env.local       # edit BIDS_ROOT_DS4504
#   module load git-annex           # Alliance clusters; or have datalad/git-annex
#   bash download_ds004504.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -f env.local ] && source env.local
: "${BIDS_ROOT_DS4504:?set BIDS_ROOT_DS4504 in env.local}"
mkdir -p "$(dirname "$BIDS_ROOT_DS4504")"

echo "Fetching ds004504 -> $BIDS_ROOT_DS4504 (EEGLAB .set; eyes-closed resting)"
if command -v datalad >/dev/null 2>&1; then
    datalad install -s https://github.com/OpenNeuroDatasets/ds004504.git "$BIDS_ROOT_DS4504"
    datalad get -d "$BIDS_ROOT_DS4504" "$BIDS_ROOT_DS4504"/sub-*/eeg/*_eeg.set "$BIDS_ROOT_DS4504"/sub-*/eeg/*_eeg.fdt
elif command -v git-annex >/dev/null 2>&1; then
    git clone https://github.com/OpenNeuroDatasets/ds004504.git "$BIDS_ROOT_DS4504"
    ( cd "$BIDS_ROOT_DS4504" && git annex get --jobs=4 sub-*/eeg/*_eeg.set sub-*/eeg/*_eeg.fdt )
else
    echo "datalad/git-annex not found. Install one, or use openneuro-py:"
    echo "  pip install openneuro-py"
    echo "  python -c \"import openneuro; openneuro.download('ds004504', target_dir='$BIDS_ROOT_DS4504')\""
    exit 1
fi

n=$(find -L "$BIDS_ROOT_DS4504" -path '*/eeg/*_eeg.set' -type f 2>/dev/null | wc -l)
echo "Done. Materialized eeg .set files: $n (expect 88; the benchmark uses the 29 HC sub-037..065)."
