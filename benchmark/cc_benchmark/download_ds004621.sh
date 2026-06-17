#!/bin/bash
# Fetch the ds004621 EEG dataset (public, OpenNeuro, CC0) for the MIR replication.
# Dzianok et al. 2022 (Nencki-Symfonia, GigaScience 11:giac015) -- 128-ch EEG/ERP; the
# benchmark uses the resting-state task (task-rest), 42 subjects (BrainVision format).
#
# Usage:
#   cp env.template env.local       # edit BIDS_ROOT_DS4621
#   module load git-annex
#   bash download_ds004621.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -f env.local ] && source env.local
: "${BIDS_ROOT_DS4621:?set BIDS_ROOT_DS4621 in env.local}"
mkdir -p "$(dirname "$BIDS_ROOT_DS4621")"

echo "Fetching ds004621 -> $BIDS_ROOT_DS4621 (task-rest; BrainVision .vhdr/.eeg/.vmrk)"
if command -v datalad >/dev/null 2>&1; then
    datalad install -s https://github.com/OpenNeuroDatasets/ds004621.git "$BIDS_ROOT_DS4621"
    datalad get -d "$BIDS_ROOT_DS4621" "$BIDS_ROOT_DS4621"/sub-*/eeg/*_task-rest_eeg.*
elif command -v git-annex >/dev/null 2>&1; then
    git clone https://github.com/OpenNeuroDatasets/ds004621.git "$BIDS_ROOT_DS4621"
    ( cd "$BIDS_ROOT_DS4621" && git annex get --jobs=4 sub-*/eeg/*_task-rest_eeg.* )
else
    echo "datalad/git-annex not found. Install one, or use openneuro-py:"
    echo "  pip install openneuro-py"
    echo "  python -c \"import openneuro; openneuro.download('ds004621', target_dir='$BIDS_ROOT_DS4621')\""
    exit 1
fi

n=$(find -L "$BIDS_ROOT_DS4621" -path '*/eeg/*_task-rest_eeg.vhdr' -type f 2>/dev/null | wc -l)
echo "Done. Materialized task-rest .vhdr files: $n (expect 42)."
