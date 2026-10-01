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
    --output             where to write the resulting JSON

OUTPUT : the output JSON is a nested dictionary of the form:
{
    <NoiseModel> : {
        <distance> : {
            <param> : {
                "both_correct" : <number of syndromes where both decoders are correct>,
                "both_wrong"   : <number of syndromes where both decoders are wrong>,
                "blind"  : <LIST of syndromes where MLP is correct but MWPM is wrong>,
                "mlp" : <number of syndromes where MLP is wrong but MWPM is correct>
            }
        }
    }
}

"""

import argparse
import json
import os
import sys

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
    Compare both decoders' predictions to the true labels, syndrome by
    syndrome, and sort every syndrome into one of 4 categories.

    @returns:
        dict with keys "both_correct"/"both_wrong"/"blind"/"mlp", each a
        list of syndromes (each syndrome is a list of 0s and 1s)
    """
    categories = {
        "both_correct": [],
        "both_wrong": [],
        "blind": [],
        "mlp": [],
    }

    n_samples = len(true_labels)
    for i in range(n_samples):
        syndrome = features[i].astype(int).tolist()
        true_label = int(true_labels[i])

        mwpm_is_correct = mwpm_predictions[i] == true_label
        mlp_is_correct = mlp_predictions[i] == true_label

        if mwpm_is_correct and mlp_is_correct:
            categories["both_correct"].append(syndrome)
        elif mwpm_is_correct and not mlp_is_correct:
            categories["mlp"].append(syndrome)
        elif not mwpm_is_correct and mlp_is_correct:
            categories["blind"].append(syndrome)
        else:
            categories["both_wrong"].append(syndrome)

    return categories


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
    mwpm_predictions = []
    for syndrome in features:
        prediction = matcher.decode(syndrome)
        mwpm_predictions.append(int(prediction[0]))

    features_tensor = torch.from_numpy(features)
    with torch.no_grad():
        logits = model(features_tensor)
    mlp_predictions = []
    for logit in logits:
        if logit.item() > 0:  # same as sigmoid(logit) > 0.5
            mlp_predictions.append(1)
        else:
            mlp_predictions.append(0)

    # 5. Sort into categories
    return sort_into_categories(features, true_labels, mwpm_predictions, mlp_predictions)


#computed from this file's own location, so the default works no matter
#which directory you run the script from
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

    disagreement_map = {}

    #do it for every (noise_model, distance, param) combination (12 in total)
    for noise_folder in NOISE_MODELS:
        disagreement_map[noise_folder] = {}

        for distance in DISTANCES:
            disagreement_map[noise_folder][distance] = {}

            for param_folder in PARAMS:
                print(f"Building disagreement map for {noise_folder}/d{distance}/{param_folder}")
                result = build_disagreement_map_for_model(
                    noise_folder, distance, param_folder, args.final_models_dir, rounds=args.rounds, samples=args.samples
                )

                if result is None:
                    print(f"  no model found, skipping")
                    continue
                #modify result so that it contains counts instead of lists of syndromes except for the "blind" category
                result["both_correct"] = len(result["both_correct"])
                result["both_wrong"] = len(result["both_wrong"])
                result["mlp"] = len(result["mlp"])
                disagreement_map[noise_folder][distance][param_folder] = result


    with open(args.output, "w") as f:
        json.dump(disagreement_map, f)
