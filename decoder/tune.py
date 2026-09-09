"""
Author : Victor Fairon
Date : 2026-09-09

Hyperparameter search for the MLP decoder

Arguments (CLI):
    --dataset     path to the .npz to tune on
    --output-dir  directory to write one folder per trial + leaderboard.json into
    --trials      number of random configs to try (default: 20)
    --epochs      epochs per trial (default: 30)
    --seed        master seed controlling both the search and every trial (default: 42)
    --verbose

Outputted files :
    <output-dir>/trial_XXX/config.json / model.pth / metrics.json   -- written by train() itself
    <output-dir>/leaderboard.json                                  -- every trial's config + best_val_accuracy, ranked
"""
import argparse
import json
import os

import numpy as np

from train import train


# Hidden-layer shapes to try, and the discrete choices each layer's dropout
# and the optimizer's learning rate are drawn from.
SEARCH_SPACE = {
    "hidden_sizes": [(128, 64), (256, 128, 64), (256, 128, 64, 32)],
    "dropout_choices": [0.0, 0.1, 0.2],   
    "lr": [0.001, 0.003, 0.01],
}


def sample_config(rng):
    """Draw one random hyperparameter config from SEARCH_SPACE using `rng` (kept for replicability)."""
    hidden_sizes = SEARCH_SPACE["hidden_sizes"][rng.integers(len(SEARCH_SPACE["hidden_sizes"]))]
    # one dropout value per hidden layer, matching model.build_model's expectations
    dropout = tuple(float(rng.choice(SEARCH_SPACE["dropout_choices"])) for _ in hidden_sizes)
    lr = float(rng.choice(SEARCH_SPACE["lr"]))

    return {
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
        "lr": lr,
    }


def tune(dataset_path, output_dir, n_trials=20, epochs=30, seed=42, verbose=False):
    """
    Run a random hyperparameter search, training each config with train.train().

    Writes one folder per trial (config.json/model.pth/metrics.json, written
    by train() itself) plus a leaderboard.json ranking every trial by
    validation accuracy, best first.

    @returns:
        list of dicts, one per trial, sorted best_val_accuracy descending:
        {"trial_id", "trial_dir", "config", "seed", "best_val_accuracy"}
    """
    # A single master RNG drives both which configs get tried and which seed
    # each trial trains with 
    rng = np.random.default_rng(seed)
    leaderboard = []

    os.makedirs(output_dir, exist_ok=True)

    for trial_id in range(n_trials):
        config = sample_config(rng)
        trial_seed = int(rng.integers(0, 2**31 - 1))
        trial_dir = os.path.join(output_dir, f"trial_{trial_id:03d}")

        if verbose:
            print(f"[{trial_id + 1}/{n_trials}] config={config} seed={trial_seed}")

        metrics = train(config, dataset_path, trial_dir, epochs=epochs, seed=trial_seed, verbose=verbose)

        leaderboard.append({
            "trial_id": trial_id,
            "trial_dir": trial_dir,
            "config": config,
            "seed": trial_seed,
            "best_val_accuracy": metrics["best_val_accuracy"],
        })

    leaderboard.sort(key=lambda row: row["best_val_accuracy"], reverse=True)
    with open(os.path.join(output_dir, "leaderboard.json"), "w") as f:
        json.dump(leaderboard, f, indent=2)

    if verbose:
        best = leaderboard[0]
        print(f"Best trial: {best['trial_dir']} (val_accuracy={best['best_val_accuracy']:.4f})")

    return leaderboard


def parse_args():
    """Define and parse the CLI arguments controlling the hyperparameter search."""
    parser = argparse.ArgumentParser(description="Random hyperparameter search for the MLP decoder.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a .npz produced by generate_datasets.py")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write trial_XXX/ folders + leaderboard.json")
    parser.add_argument("--trials", type=int, default=20, help="Number of random configs to try (default: 20)")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per trial (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Master seed for the search (default: 42)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune(args.dataset, args.output_dir, n_trials=args.trials, epochs=args.epochs, seed=args.seed, verbose=args.verbose)
