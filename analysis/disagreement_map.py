"""
Date : 2026-09-30

Build a disagreement map between the MWPM (pymatching) baseline decoder and
a trained MLP decoder. The main does it for every one of our models but this
file contains functions to do it for only one model if desired.

We build the disagreement map by sampling a large number of syndromes from the noise model,
decoding them with both decoders, and recording the list of syndromes on which they disagree.

Arguments (CLI):
    --final-models-dir   root dir containing <NoiseModel>/d<D>/<param>/{config.json,model.pth} (default: ../final_models)
    --samples            number of syndromes to draw per config (default: 100_000)
    --rounds             number of syndrome-measurement rounds used to train the models (default: 9)
    --output             where to write the resulting HDF5 file

OUTPUT : the output is an HDF5 file (https://www.h5py.org/), 
The file is organized as nested groups. 
Every category is stored as a full array of syndromes :
    <NoiseModel>/d<distance>/<param>/
        both_correct     : array of syndromes where both decoders are correct
        both_wrong       : array of syndromes where both decoders are wrong
        mlp_blind_spot   : array of syndromes where MWPM is correct but MLP is wrong
        mwpm_blind_spot  : array of syndromes where MLP is correct but MWPM is wrong
"""

import argparse
import gc
import itertools
import os
import sys

import h5py
import numpy as np
import pymatching
import torch

#ajout des dossier decoder et data au path pour pouvoir importer les modules model et generate_datasets
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "decoder"))

from generate_datasets import create_circuit, create_sampler, sample as draw_samples
from model import build_model_from_checkpoint_as_eval

# the noise models and distances/params we trained a model for (matches the
# folder names under final_models/)
NOISE_MODELS = ["Depolarizing", "FT"]
DISTANCES = [3, 5, 7]
PARAMS = ["005", "01"]

# final_models/ folder name -> noise_type expected by generate_datasets.create_circuit
NOISE_FOLDER_TO_TYPE = {
    "Depolarizing": "depolarizing",
    "FT": "circuit-level",
}

# final_models/.../<param folder> -> actual physical error rate
PARAM_FOLDER_TO_VALUE = {
    "005": 0.005,
    "01": 0.01,
}



def sort_into_categories(features, true_labels, mwpm_predictions, mlp_predictions):
    """
    Compare both decoders' predictions to the true labels and sort every
    syndrome into one of 4 categories. Uses numpy boolean masks instead of
    a per-syndrome python loop, both for speed and because it keeps the
    syndromes as compact numpy arrays (ready to write to HDF5) instead of
    turning each one into a python list.

    @returns:
        dict with keys "both_correct"/"both_wrong"/"mlp_blind_spot"/"mwpm_blind_spot",
        each a numpy array of syndromes (one syndrome per row).
    """
    mwpm_is_correct = mwpm_predictions == true_labels
    mlp_is_correct = mlp_predictions == true_labels

    both_correct_mask = mwpm_is_correct & mlp_is_correct
    mlp_blind_spot_mask = mwpm_is_correct & ~mlp_is_correct
    mwpm_blind_spot_mask = ~mwpm_is_correct & mlp_is_correct
    both_wrong_mask = ~mwpm_is_correct & ~mlp_is_correct

    # features are 0/1 values but come in as float32; store as uint8 instead
    return {
        "both_correct": features[both_correct_mask].astype(np.uint8),
        "both_wrong": features[both_wrong_mask].astype(np.uint8),
        "mlp_blind_spot": features[mlp_blind_spot_mask].astype(np.uint8),
        "mwpm_blind_spot": features[mwpm_blind_spot_mask].astype(np.uint8),
    }


def build_disagreement_map_for_model(noise_folder, distance, param_folder, final_models_dir, rounds=9, samples=100_000):
    """
    Sample syndromes from one (noise_folder, distance, param_folder) config,
    decode them with MWPM and the matching MLP checkpoint, and sort every
    syndrome into a disagreement category.

    @returns:
        dict from sort_into_categories, or None if no checkpoint exists for
        this config (e.g. a final_models/.../d.../... folder that's empty)
    """
    # 1. Build MLP model from checkpoint
    model_dir = os.path.join(final_models_dir, noise_folder, f"d{distance}", param_folder)#for example ../final_models/Depolarizing/d3/005
    config_path = os.path.join(model_dir, "config.json")
    checkpoint_path = os.path.join(model_dir, "model.pth")
    if not (os.path.exists(config_path) and os.path.exists(checkpoint_path)):
        return None

    model = build_model_from_checkpoint_as_eval(config_path, checkpoint_path, distance, rounds)

    # 2. Build circuit and MWPM decoder from circuit
    noise_type = NOISE_FOLDER_TO_TYPE[noise_folder]
    noise_param = PARAM_FOLDER_TO_VALUE[param_folder]
    circuit = create_circuit(distance=distance, rounds=rounds, noise_type=noise_type, noise_default=noise_param)
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    # 3. Sample syndromes from the noise model
    sampler = create_sampler(circuit)
    true_labels, features = draw_samples(sampler, samples)
    true_labels = true_labels.reshape(-1)  # from shape (samples, 1) to shape (samples,)

    # 4. Decode with both decoders
    mwpm_predictions = np.array([int(matcher.decode(syndrome)[0]) for syndrome in features])

    features_tensor = torch.from_numpy(features)
    with torch.no_grad():
        logits = model(features_tensor)
    mlp_predictions = (logits.reshape(-1) > 0).numpy().astype(int)  # same as sigmoid(logit) > 0.5

    # 5. Sort into categories
    return sort_into_categories(features, true_labels, mwpm_predictions, mlp_predictions)


#our final models are in "final_models/"
DEFAULT_FINAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "final_models")


def parse_args():
    parser = argparse.ArgumentParser(description="Build an MWPM vs MLP disagreement map.")
    parser.add_argument("--final-models-dir", type=str, default=DEFAULT_FINAL_MODELS_DIR)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    #all 12 (noise_model, distance, param) combinations, flattened into one
    #simple list so we only need a single for loop below
    all_configs = itertools.product(NOISE_MODELS, DISTANCES, PARAMS)

    with h5py.File(args.output, "w") as f:
        for noise_folder, distance, param_folder in all_configs:
            print(f"Building disagreement map for {noise_folder}/d{distance}/{param_folder}")
            result = build_disagreement_map_for_model(
                noise_folder, distance, param_folder, args.final_models_dir, rounds=args.rounds, samples=args.samples
            )

            if result is None:
                print(f"  no model found, skipping")
                continue

            #write this config's result to disk straight away, as its own H5 group,
            #one dataset per category (all 4 kept as full syndrome arrays)
            group = f.create_group(f"{noise_folder}/d{distance}/{param_folder}")
            group.create_dataset("both_correct", data=result["both_correct"], compression="gzip")
            group.create_dataset("both_wrong", data=result["both_wrong"], compression="gzip")
            group.create_dataset("mlp_blind_spot", data=result["mlp_blind_spot"], compression="gzip")
            group.create_dataset("mwpm_blind_spot", data=result["mwpm_blind_spot"], compression="gzip")

            #drop the (potentially huge) syndrome arrays before moving to the next config
            del result, group
            gc.collect()
