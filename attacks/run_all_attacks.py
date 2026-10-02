"""
Run the SA adversarial attack (every weight budget, see
simulated_annealing.weight_budget_range) against every trained decoder
checkpoint under final_models/. It uses seed 0 and 50 restarts per weight budget.

Arguments (CLI):
    --final-models-dir   root dir containing <NoiseModel>/d<D>/<param>/{config.json,model.pth} (default: ../final_models)
    --output-dir          directory to write one <NoiseModel>_d<D>_<param>.jsonl file per model into
    --rounds               number of syndrome-measurement rounds the models were trained on (default: 9)

OUTPUT: one .jsonl file per model in --output-dir (see
simulated_annealing.py's module docstring for the line format). Models
with no checkpoint under --final-models-dir are skipped.
"""
import argparse
import itertools
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "decoder"))

from model import build_model_from_checkpoint_as_eval
from simulated_annealing import build_dem, run_full_attack_for_model

# the noise models and distances/params we trained a model for (matches the
# folder names under final_models/, same convention as analysis/disagreement_map.py)
NOISE_MODELS = ["Depolarizing", "FT"]
DISTANCES = [3, 5, 7]
PARAMS = ["005", "01"]

# final_models/ folder name -> noise_type expected by simulated_annealing.build_dem
NOISE_FOLDER_TO_TYPE = {
    "Depolarizing": "depolarizing",
    "FT": "circuit-level",
}

# final_models/.../<param folder> -> actual physical error rate
PARAM_FOLDER_TO_VALUE = {
    "005": 0.005,
    "01": 0.01,
}

DEFAULT_FINAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "final_models")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SA attack against every trained model.")
    parser.add_argument("--final-models-dir", default=DEFAULT_FINAL_MODELS_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--n-restarts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    combos = list(itertools.product(NOISE_MODELS, DISTANCES, PARAMS))
    
    for noise_folder, distance, param_folder in tqdm(combos, desc="Attacking models"):

        model_dir = os.path.join(args.final_models_dir, noise_folder, f"d{distance}", param_folder)
        config_path = os.path.join(model_dir, "config.json")
        checkpoint_path = os.path.join(model_dir, "model.pth")
        if not (os.path.exists(config_path) and os.path.exists(checkpoint_path)):
            tqdm.write(f"  no model found for {noise_folder}/d{distance}/{param_folder}, skipping")
            continue

        tqdm.write(f"#############################\n#############################\nAttacking {noise_folder}/d{distance}/{param_folder}\n#############################\n#############################\n")
        model = build_model_from_checkpoint_as_eval(config_path, checkpoint_path, distance, args.rounds)

        noise_type = NOISE_FOLDER_TO_TYPE[noise_folder]
        noise_param = PARAM_FOLDER_TO_VALUE[param_folder]
        dem = build_dem(distance, args.rounds, noise_type, noise_param)

        output_path = os.path.join(args.output_dir, f"{noise_folder}_d{distance}_{param_folder}.jsonl")
        run_full_attack_for_model(
            model, dem, distance, output_path,
            rounds=args.rounds, noise_type=noise_type, noise_default=noise_param,
            n_restarts=50, seed=0,
        )
