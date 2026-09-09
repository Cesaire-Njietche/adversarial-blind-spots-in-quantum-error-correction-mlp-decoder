"""
Author : Victor Fairon
Date : 2026-09-09

Minimum-Weight Perfect Matching (MWPM) reference/baseline decoder, built from the exact circuit that produced a given dataset. The circuit
is rebuilt via generate_datasets.create_circuit() from the .json file
exported by generate_datasets.save_metadata().

Arguments :
    --metadata          path to the circuit-parameters .json (from generate_datasets.py)
    --dataset           path to an existing .npz (features/labels) to decode
    --samples           number of shots to sample on the fly if --dataset is not given (can be useful when you want to build a new one)
    --prediction-path   where to save predictions/metrics as a .npz file 
    --verbose

Note on --dataset vs --samples : whenever this decoder needs to be compared
against another decoder (e.g. the MLP) on the same syndromes, always pass
--dataset pointing at that shared .npz. Sampling fresh shots here (via
--samples) draws an independent random syndrome set.
"""
import argparse
import json
import os
import sys

import numpy as np
import pymatching

# Reuse the exact same circuit-building/sampling code as dataset generation,
# so the matcher below is guaranteed to match the dataset it decodes.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from generate_datasets import create_circuit, create_sampler, sample as draw_samples


def load_metadata(metadata_path):
    """Load the circuit parameters (distance, rounds, noise_type, noise_default) saved alongside a dataset."""
    with open(metadata_path) as f:
        return json.load(f)


def build_matcher(circuit):
    """
    Turn the circuit into a PyMatching matcher.
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return matcher


def load_or_sample_syndromes(circuit, dataset_path, n_samples, verbose=False):
    """
    Get the (features, labels) to decode.

    If dataset_path is given, load the syndromes actually stored in that
    .npz -- (whenever the result needs to be comparable to another decoder run on the same data).

    Otherwise, draw a fresh sample straight from the circuit (standalone
    smoke test / quick baseline number only).
    """
    if dataset_path:
        if verbose:
            print(f"Loading syndromes to decode from {dataset_path}")
        data = np.load(dataset_path)
        return data["features"], data["labels"]
    else : 
        if verbose:
            print(f"No --dataset given: sampling {n_samples} fresh shots from the circuit instead.")
        sampler = create_sampler(circuit)
        labels, features = draw_samples(sampler, n_samples, verbose=verbose)
        return features, labels



def decode(matcher, features):
    """
    Run MWPM on a batch of syndromes.

    @args:
        features : np.ndarray, shape (n_shots, (d²-1)*rounds)
    @returns:
        np.ndarray, shape (n_shots, 1) of predicted logical observable flips (0/1)
    """
    predictions = matcher.decode_batch(features)
    return np.asarray(predictions, dtype=np.float32).reshape(-1, 1)


def evaluate(predictions, labels, verbose=False):
    """Compare MWPM predictions to ground-truth labels and return the logical error rate."""
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    n_errors = np.sum(predictions != labels)
    logical_error_rate = n_errors / labels.shape[0]

    if verbose:
        print("########################################################")
        print("MWPM baseline results")
        print(f"  Shots              : {labels.shape[0]}")
        print(f"  Logical error rate : {logical_error_rate:.6f}")
        print("########################################################")

    return logical_error_rate


def save_predictions(prediction_path, predictions, labels, logical_error_rate):
    """Write predictions, ground-truth labels and the summary metric to a .npz file."""
    np.savez(
        prediction_path,
        predictions=predictions,
        labels=labels,
        logical_error_rate=logical_error_rate,
    )


def parse_args():
    """Define and parse the CLI arguments controlling how the MWPM baseline is run."""
    parser = argparse.ArgumentParser(
        description="Decode syndromes with a PyMatching MWPM baseline decoder."
    )

    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to the circuit-parameters .json produced by generate_datasets.py.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to an existing .npz (features/labels) to decode. If omitted, "
             "a fresh test set is sampled from the circuit instead (see module docstring).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="Number of shots to sample when --dataset is not given (default: 100,000).",
    )
    parser.add_argument(
        "--prediction-path",
        type=str,
        required=True,
        help="Where to save predictions/metrics as a .npz file. (eg ./export.npz)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    metadata = load_metadata(args.metadata)
    if args.verbose:
        print("########################################################")
        print(f"Loaded circuit metadata from {args.metadata}: {metadata}")
        print("########################################################")

    circuit = create_circuit(**metadata, verbose=args.verbose)
    matcher = build_matcher(circuit)

    features, labels = load_or_sample_syndromes(
        circuit, args.dataset, args.samples, verbose=args.verbose
    )
    predictions = decode(matcher, features)
    logical_error_rate = evaluate(predictions, labels, verbose=args.verbose)

    if args.prediction_path:
        save_predictions(args.prediction_path, predictions, labels, logical_error_rate)
        if args.verbose:
            print(f"Saved predictions to {args.prediction_path}")
