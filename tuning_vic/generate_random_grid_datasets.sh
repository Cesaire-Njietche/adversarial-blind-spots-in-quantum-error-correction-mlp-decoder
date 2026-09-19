#!/usr/bin/env bash
#
# Generate circuit-level datasets where each of the 4 noise parameters
# is drawn independently at random from {0.001, 0.002, ..., 0.010}.
#
# Usage:
#   ./generate_random_grid_datasets.sh X [DISTANCE] [SAMPLES] [OUTPUT_DIR] [SEED]
#
#   X           number of random datasets to generate (required)
#   DISTANCE    surface code distance, 3 or 5 (default: 3)
#   SAMPLES     number of shots per dataset (default: 1000000)
#   OUTPUT_DIR  where to write <label>/dataset.npz
#               (default: ./d<DISTANCE>/random_grid, next to this script)
#   SEED        optional seed for bash's RANDOM
#
# Each dataset is written to OUTPUT_DIR/rand_<i>/dataset.npz.
# The fixed reference point is written to OUTPUT_DIR/all_010/dataset.npz.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 X [DISTANCE] [SAMPLES] [OUTPUT_DIR] [SEED]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_DATASETS="$1"
DISTANCE="${2:-3}"
SAMPLES="${3:-1000000}"
OUTPUT_DIR="${4:-"$SCRIPT_DIR/d${DISTANCE}/random_grid"}"
SEED="${5:-}"
ROUNDS=9

if [[ -n "$SEED" ]]; then
    RANDOM="$SEED"
fi

mkdir -p "$OUTPUT_DIR"

# Draw one value uniformly from {0.001, 0.002, ..., 0.010}
# (10 possible values, 0.001 precision).
random_prob() {
    local n=$(( (RANDOM % 10) + 1 ))
    awk -v n="$n" 'BEGIN { printf "%.3f", n * 0.001 }'
}

manifest="$OUTPUT_DIR/manifest.csv"
echo "label,after_clifford_depolarization,after_reset_flip_probability,before_measure_flip_probability,before_round_data_depolarization" > "$manifest"

for i in $(seq 1 "$N_DATASETS"); do
    label="rand_$(printf '%03d' "$i")"
    dataset_dir="$OUTPUT_DIR/$label"
    mkdir -p "$dataset_dir"

    p_clifford=$(random_prob)
    p_reset=$(random_prob)
    p_measure=$(random_prob)
    p_round=$(random_prob)

    echo "########################################################"
    echo "[$label] d${DISTANCE} circuit-level: clifford=${p_clifford} reset=${p_reset} measure=${p_measure} round=${p_round}"
    echo "########################################################"

    python3 "$SCRIPT_DIR/../data/generate_datasets.py" \
        --distance "$DISTANCE" \
        --rounds "$ROUNDS" \
        --samples "$SAMPLES" \
        --noise circuit-level \
        --after-clifford-depolarization "$p_clifford" \
        --after-reset-flip-probability "$p_reset" \
        --before-measure-flip-probability "$p_measure" \
        --before-round-data-depolarization "$p_round" \
        --output "$dataset_dir/dataset.npz" \
        --verbose

    echo "${label},${p_clifford},${p_reset},${p_measure},${p_round}" >> "$manifest"
done

# # Reference dataset with all 4 parameters at the top of the range.
fixed_dir="$OUTPUT_DIR/all_005"
mkdir -p "$fixed_dir"

# echo "########################################################"
# echo "[all_010] d${DISTANCE} circuit-level: all 4 params = 0.010"
# echo "########################################################"

python3 "$SCRIPT_DIR/../data/generate_datasets.py" \
    --distance "$DISTANCE" \
    --rounds "$ROUNDS" \
    --samples "$SAMPLES" \
    --noise circuit-level \
    --after-clifford-depolarization 0.005 \
    --after-reset-flip-probability 0.005 \
    --before-measure-flip-probability 0.005 \
    --before-round-data-depolarization 0.005 \
    --output "$fixed_dir/dataset.npz" \
    --verbose

# echo "all_010,0.010,0.010,0.010,0.010" >> "$manifest"

# echo "Done. $((N_DATASETS + 1)) datasets written under: $OUTPUT_DIR"
# echo "Manifest: $manifest"
