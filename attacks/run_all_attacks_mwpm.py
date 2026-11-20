"""
Run the SA adversarial attack (see simulated_annealing_mwpm.py) against the
MWPM reference decoder for every <NoiseModel>/d<D>/<param> combination we
trained an MLP for -- no checkpoint is needed here (MWPM is rebuilt purely
from the circuit), the combos are just reused from final_models/ so the
output files line up 1:1 with attacks/run_all_attacks.py's MLP results.

Arguments (CLI):
    --output-dir   directory to write one <NoiseModel>_d<D>_<param>.jsonl file
                    per config into (same filename convention as run_all_attacks.py,
                    so pairing the two directories by filename gives you the MLP vs
                    MWPM attack for the same config), plus one combined
                    attack_rate_summary.csv across every config/weight
    --rounds       number of syndrome-measurement rounds (default: 9)
    --n-restarts   independent SA chains per weight budget (default: 50)
    --seed         random seed (default: 0)

OUTPUT:
    <output-dir>/<NoiseModel>_d<D>_<param>.jsonl  -- one per config (catalogs + labels,
                                                       see simulated_annealing_mwpm.py)
    <output-dir>/attack_rate_summary.csv          -- one row per (noise, d, p, weight):
                                                       noise_model, distance, param,
                                                       weight, n_catalog, attack_rate
"""
import argparse
import csv
import itertools
import json
import os

import pymatching
from tqdm import tqdm

from simulated_annealing_mwpm import build_dem, run_full_attack_for_model

# same convention as attacks/run_all_attacks.py / analysis/disagreement_map.py
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SA attack against MWPM for every (noise, distance, param) config.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--n-restarts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    combos = list(itertools.product(NOISE_MODELS, DISTANCES, PARAMS))
    summary_rows = []

    for noise_folder, distance, param_folder in tqdm(combos, desc="Attacking MWPM"):
        tqdm.write(f"#############################\nAttacking MWPM: {noise_folder}/d{distance}/{param_folder}\n#############################")

        noise_type = NOISE_FOLDER_TO_TYPE[noise_folder]
        noise_param = PARAM_FOLDER_TO_VALUE[param_folder]
        dem = build_dem(distance, args.rounds, noise_type, noise_param)
        matcher = pymatching.Matching.from_detector_error_model(dem)

        output_path = os.path.join(args.output_dir, f"{noise_folder}_d{distance}_{param_folder}.jsonl")
        run_full_attack_for_model(
            matcher, dem, distance, output_path,
            rounds=args.rounds, noise_type=noise_type, noise_default=noise_param,
            n_restarts=args.n_restarts, seed=args.seed,
        )

        with open(output_path) as f:
            for line in f:
                record = json.loads(line)
                if record["type"] == "result":
                    summary_rows.append({
                        "noise_model": noise_folder,
                        "distance": distance,
                        "param": param_folder,
                        "weight": record["weight"],
                        "n_catalog": len(record["catalog"]),
                        "attack_rate": record["attack_rate"],
                    })

    summary_path = os.path.join(args.output_dir, "attack_rate_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "noise_model", "distance", "param", "weight", "n_catalog", "attack_rate",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote {len(summary_rows)} rows to {summary_path}")
