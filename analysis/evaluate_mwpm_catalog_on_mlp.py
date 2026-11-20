"""
Cross-decoder transfer test (reverse direction of
analysis/evaluate_catalog_cross_decoder.py): take the catalog of syndromes
that fooled MWPM (produced by attacks/run_all_attacks_mwpm.py /
attacks/simulated_annealing_mwpm.py, which also stores "catalog_labels" --
see that file's docstring), decode those exact same syndromes with the
matching trained MLP checkpoint, and check whether the MLP also gets them
wrong.

This answers: "does an attack crafted against MWPM transfer to the MLP?",
directly comparable to Φ_MWPM(W) (attack_rate already stored in the MWPM
attack's own .jsonl header/result).

Kept as its own separate script on purpose, mirroring
evaluate_catalog_cross_decoder.py's structure rather than generalizing a
single "either direction" tool out of the two.

Arguments (CLI):
    --attacks_dir        directory of <NoiseModel>_d<D>_<param>.jsonl files
                          produced by attacks/run_all_attacks_mwpm.py (MWPM attack)
    --final-models-dir   root dir containing <NoiseModel>/d<D>/<param>/{config.json,model.pth}
                          (default: ../final_models)
    --rounds              number of rounds in the circuit (default: 9)
    --output              path to the .csv summary to write

OUTPUT: one row per (noise_model, distance, param, weight) with columns
    noise_model, distance, param, weight,
    n_catalog, attack_rate_mwpm, transfer_rate_mwpm_catalog_on_mlp
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "decoder"))
from model import build_model_from_checkpoint_as_eval

NOISE_MODELS = ["Depolarizing", "FT"]
DISTANCES = [3, 5, 7]
PARAMS = ["005", "01"]

DEFAULT_FINAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "final_models")


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


def transfer_rate_on_mlp(model, catalog, catalog_labels):
    """Fraction of (syndrome, label) pairs in the catalog that the MLP ALSO decodes incorrectly."""
    if len(catalog) == 0:
        return float("nan")
    shots = torch.from_numpy(np.array(catalog, dtype=np.float32))
    labels = np.array(catalog_labels, dtype=np.int64)
    with torch.no_grad():
        logits = model(shots)
    predictions = (logits.reshape(-1) > 0).numpy().astype(np.int64)
    return float(np.mean(predictions != labels))


def parse_args():
    parser = argparse.ArgumentParser(description="Test MWPM's adversarial catalog against the MLP.")
    parser.add_argument("--attacks_dir", type=str, required=True)
    parser.add_argument("--final-models-dir", type=str, default=DEFAULT_FINAL_MODELS_DIR)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_configs = list(itertools.product(NOISE_MODELS, DISTANCES, PARAMS))

    rows = []
    for noise_folder, distance, param_folder in all_configs:
        mwpm_path = os.path.join(args.attacks_dir, f"{noise_folder}_d{distance}_{param_folder}.jsonl")
        if not os.path.exists(mwpm_path):
            print(f"Skipping {noise_folder}/d{distance}/{param_folder}: no MWPM attack file at {mwpm_path}")
            continue

        model_dir = os.path.join(args.final_models_dir, noise_folder, f"d{distance}", param_folder)
        config_path = os.path.join(model_dir, "config.json")
        checkpoint_path = os.path.join(model_dir, "model.pth")
        if not (os.path.exists(config_path) and os.path.exists(checkpoint_path)):
            print(f"Skipping {noise_folder}/d{distance}/{param_folder}: no MLP checkpoint under {model_dir}")
            continue

        mwpm_header, mwpm_results = read_jsonl_records(mwpm_path)
        model = build_model_from_checkpoint_as_eval(config_path, checkpoint_path, distance, args.rounds)

        for weight, result in sorted(mwpm_results.items()):
            catalog = result["catalog"]
            catalog_labels = result.get("catalog_labels")
            if catalog_labels is None:
                print(f"Skipping {noise_folder}/d{distance}/{param_folder} W={weight}: "
                      f"no catalog_labels in {mwpm_path} -- rerun attacks/run_all_attacks_mwpm.py "
                      f"to regenerate it.")
                continue

            tau = transfer_rate_on_mlp(model, catalog, catalog_labels)

            rows.append({
                "noise_model": noise_folder,
                "distance": distance,
                "param": param_folder,
                "weight": weight,
                "n_catalog": len(catalog),
                "attack_rate_mwpm": result["attack_rate"],
                "transfer_rate_mwpm_catalog_on_mlp": tau,
            })
            print(f"{noise_folder}/d{distance}/{param_folder} W={weight}: "
                  f"n_catalog={len(catalog)}, Phi_MWPM={result['attack_rate']:.3f}, "
                  f"transfer_to_MLP={tau:.3f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "noise_model", "distance", "param", "weight",
            "n_catalog", "attack_rate_mwpm",
            "transfer_rate_mwpm_catalog_on_mlp",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.output}")
