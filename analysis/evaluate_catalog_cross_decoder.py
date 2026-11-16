"""
Cross-decoder transfer test: take the catalog of syndromes that fooled the
MLP (produced by attacks/simulated_annealing.py, which also stores the
ground-truth "catalog_labels" for each syndrome -- see that file's
docstring), decode those exact same syndromes with MWPM, and check whether
MWPM also gets them wrong.

This answers: "does an attack crafted against the MLP transfer to MWPM?",
directly comparable to Φ_MLP(W) (attack_rate already stored in the MLP's
own .jsonl header/result).

Arguments (CLI):
    --attacks_dir   directory of <NoiseModel>_d<D>_<param>.jsonl files
                     produced by attacks/simulated_annealing.py (MLP attack,
                     must have been (re-)run after catalog_labels was added)
    --rounds         number of rounds in the circuit (default: 9)
    --output         path to the .csv summary to write

OUTPUT: one row per (noise_model, distance, param, weight) with columns
    noise_model, distance, param, weight,
    n_catalog, attack_rate_mlp, transfer_rate_mlp_catalog_on_mwpm
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
import pymatching
import stim

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from noise_models import generate_noise_constants as nm

NOISE_MODELS = ["Depolarizing", "FT"]
DISTANCES = [3, 5, 7]
PARAMS = ["005", "01"]

NOISE_FOLDER_TO_TYPE = {
    "Depolarizing": "depolarizing",
    "FT": "circuit-level",
}

PARAM_FOLDER_TO_VALUE = {
    "005": 0.005,
    "01": 0.01,
}


def build_dem(distance, rounds, noise_type, noise_default):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        **nm(noise_type=noise_type, default=noise_default),
    )
    return circuit.detector_error_model(decompose_errors=True)


def read_jsonl_records(jsonl_path):
    """Return (header, {weight: result_record}) for one SA .jsonl file."""
    header = None
    results = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record["type"] == "header":
                header = record
            elif record["type"] == "result":
                results[record["weight"]] = record
    if header is None:
        raise ValueError(f"{jsonl_path} has no header record")
    return header, results


def transfer_rate_on_mwpm(matcher, catalog, catalog_labels):
    """Fraction of (syndrome, label) pairs in the catalog that MWPM ALSO decodes incorrectly."""
    if len(catalog) == 0:
        return float("nan")
    shots = np.array(catalog, dtype=np.uint8)
    labels = np.array(catalog_labels, dtype=np.int64)
    predictions = matcher.decode_batch(shots).reshape(-1).astype(np.int64)
    return float(np.mean(predictions != labels))


def parse_args():
    parser = argparse.ArgumentParser(description="Test the MLP's adversarial catalog against MWPM.")
    parser.add_argument("--attacks_dir", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_configs = list(itertools.product(NOISE_MODELS, DISTANCES, PARAMS))

    rows = []
    for noise_folder, distance, param_folder in all_configs:
        mlp_path = os.path.join(args.attacks_dir, f"{noise_folder}_d{distance}_{param_folder}.jsonl")
        if not os.path.exists(mlp_path):
            print(f"Skipping {noise_folder}/d{distance}/{param_folder}: no MLP attack file at {mlp_path}")
            continue

        mlp_header, mlp_results = read_jsonl_records(mlp_path)

        noise_type = NOISE_FOLDER_TO_TYPE[noise_folder]
        noise_default = PARAM_FOLDER_TO_VALUE[param_folder]
        dem = build_dem(distance, args.rounds, noise_type, noise_default)
        matcher = pymatching.Matching.from_detector_error_model(dem)

        for weight, result in sorted(mlp_results.items()):
            catalog = result["catalog"]
            catalog_labels = result.get("catalog_labels")
            if catalog_labels is None:
                print(f"Skipping {noise_folder}/d{distance}/{param_folder} W={weight}: "
                      f"no catalog_labels in {mlp_path} -- rerun attacks/simulated_annealing.py "
                      f"(or run_all_attacks.py) to regenerate it with labels.")
                continue

            tau = transfer_rate_on_mwpm(matcher, catalog, catalog_labels)

            rows.append({
                "noise_model": noise_folder,
                "distance": distance,
                "param": param_folder,
                "weight": weight,
                "n_catalog": len(catalog),
                "attack_rate_mlp": result["attack_rate"],
                "transfer_rate_mlp_catalog_on_mwpm": tau,
            })
            print(f"{noise_folder}/d{distance}/{param_folder} W={weight}: "
                  f"n_catalog={len(catalog)}, Phi_MLP={result['attack_rate']:.3f}, "
                  f"transfer_to_MWPM={tau:.3f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "noise_model", "distance", "param", "weight",
            "n_catalog", "attack_rate_mlp",
            "transfer_rate_mlp_catalog_on_mwpm",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.output}")
