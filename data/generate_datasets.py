"""
Author : Victor Fairon
Date : 2026-09-09

This file allows to create npz datasets from the stim circuit generation, with the noise model parameters imported from the noise_models.py file. 

Arguments : 
    --distance
    --rounds
    --samples
    --noise
    --output (the file path)


Generated files :
    A  .npz archive at --output containing:
        - "labels"   : shape (samples, 1), logical observable flips (0/1)
        - "features" : shape (samples, (d²-1)*rounds), detector (syndrome) bits
    A .json file containing data needed to re-create circuit used to generate the dataset (distance, rounds, noise model type and parameters). This file is saved next to the .npz file with the same name but with a .json extension.


With --verbose, a short dataset_analytics() summary (logical flip rate,
mean syndrome weight, fraction of trivial syndromes) is printed before saving.
"""
import argparse
import stim 
import numpy as np
from noise_models import generate_noise_constants as nm
import json
import os 


def create_circuit(
        distance, 
        rounds, 
        noise_type, 
        noise_default, 
        verbose=False
        ):
    """
    Build the stim.Circuit used to generate the dataset.

    Uses stim's built-in "surface_code:rotated_memory_z" generator, which
    protects a logical qubit stored in the Z basis (so it is sensitive to
    logical X errors caused by bit flips). Noise parameters are looked up
    from noise_models.py based on the --noise CLI argument, keeping the
    actual probabilities out of this file.


    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", # Protect a qubit stored in the z basis to detect a bit flip (logical x error)
        rounds = rounds,
        distance = distance,

        #Note that following line can be changed to be more specific (see noise_models.py for more details)
        **nm(
            noise_type=noise_type,
            default=noise_default
        )

    )
    if verbose:
        print(f"Generated circuit with distance={distance}, rounds={rounds}, noise={noise_type} and default noise value={noise_default}")
        print("########################################################")
    return circuit

def create_sampler(circuit, seed=None) :
    """
    Compile a detector sampler from the circuit (used to draw syndrome/observable samples).

    @args:
        seed : int, optional
            Pass an explicit seed to get reproducible sampling (e.g. drawing
            several independent evaluation sets from the same circuit, one
            seed each). Default None lets stim pick its own randomness.
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    return sampler


def sample(sampler, n_samples, verbose=False) :
    """
    Draw n_samples shots from the sampler.

    Returns:
        labels   : np.ndarray, shape (samples, 1)              — logical observable flips (0/1)
        features : np.ndarray, shape (samples, (d²-1)*rounds)  — detector (syndrome) bits
    """
    #syndromes shape : (samples, (d²-1)*rounds)
    #obs shape : (samples, 1)
    syndromes, obs = sampler.sample(n_samples, separate_observables=True)
    if verbose:
        print(f"Shape of the syndromes : {syndromes.shape}")
        print(f"Shape of the observables : {obs.shape}")
    labels = obs.astype(np.float32)
    features = syndromes.astype(np.float32)

    return labels, features

def create_dataset(labels, features):
    """Package labels/features into the dict layout expected by save_dataset."""
    return {
        "labels": labels,
        "features": features,
    }


def save_dataset(dataset):
    """
    Write the dataset dict to args.output as a compressed-free .npz archive.
    """
    np.savez(
        args.output,
        labels=dataset["labels"],
        features=dataset["features"],
    )

def save_metadata(dataset_path, distance, rounds, noise_type, noise_default=0.005):
    """
    Write a small JSON sidecar next to the dataset with the parameters used
    to build the circuit that generated it. Anything that needs to rebuild
    an identical circuit later (e.g. an MWPM baseline decoder) reads this
    file instead of guessing/hardcoding the settings.
    """
    metadata = {
        "distance": distance,
        "rounds": rounds,
        "noise_type": noise_type,
        "noise_default": noise_default,
    }

    #remove extension
    metadata_path = os.path.splitext(dataset_path)[0] + ".json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


def dataset_analytics(dataset):
    """
    Print a few quick sanity-check statistics about a generated dataset.

    @args:
        dataset : dict
            Dict with "labels" (shape (samples, 1)) and "features"
            (shape (samples, (d²-1)*rounds)) as returned by create_dataset.
    """
    labels = dataset["labels"]
    features = dataset["features"]
    n_samples = labels.shape[0]

    # Fraction of shots where the logical observable flipped (label == 1).
    flip_rate = labels.mean()

    # Mean and standard deviation of the number of detectors that fired in each shot.

    #summing per shot the number of detectors that fired (1) or not (0) (agglo horizontally)
    syndrome_weight = features.sum(axis=1)
    mean_weight = syndrome_weight.mean()
    std_weight = syndrome_weight.std()

    # Fraction of shots with a completely silent syndrome (no detector
    # fired at all) -- these carry no information for a decoder.
    trivial_fraction = np.mean(syndrome_weight == 0)

    print("########################################################")
    print("Dataset analytics")
    print(f"  Samples                     : {n_samples}")
    print(f"  Logical flip rate           : {flip_rate:.4f}")
    print(f"  Mean syndrome weight        : {mean_weight:.4f} (+/- {std_weight:.4f})")
    print(f"  Trivial (all-zero) syndromes: {trivial_fraction:.2%}")
    print("########################################################")

def parse_args():
    """Define and parse the CLI arguments controlling circuit/sampling/output settings."""
    parser = argparse.ArgumentParser(
    description="Generate a noisy quantum error-correction dataset."
    )

    parser.add_argument(
        "--distance", 
        type=int, 
        help="Surface code distance (default : 3)", 
        default=3)
    
    parser.add_argument(
        "--rounds", 
        type=int, 
        help="Number of syndrome-measurement rounds (default: 9).", 
        default=9)
    
    parser.add_argument("--samples",
        type=int, 
        help="Number of samples to collect (default: 100,000).",
        default=100_000)
    
    parser.add_argument(
        "--noise",
        choices=["depolarizing", "circuit-level"],
        default="depolarizing",
        help="Noise model to use either \"depolarizing\" or \"circuit-level\" (default: depolarizing)."
    )
    parser.add_argument(
        "--noise-default",
        type=float,
        default=0.005,
        help="Physical error rate applied to any noise parameter not explicitly overridden "
             "(see noise_models.py). Default: 0.005 (0.5%%)."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (e.g., ./export/dataset.npz) (required)",
        required=True)

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during dataset generation."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        print("########################################################")
        print(
            f"Generating dataset with distance={args.distance}, rounds={args.rounds}, "
            f"samples={args.samples}, noise={args.noise} on file {args.output}"
        )
        print("########################################################")



    circuit = create_circuit(args.distance, args.rounds, args.noise, noise_default=args.noise_default, verbose=args.verbose)
    sampler = create_sampler(circuit)
    labels, features = sample(sampler, args.samples, verbose=args.verbose)
    dataset = create_dataset(labels, features)

    if args.verbose:
        dataset_analytics(dataset)

    save_dataset(dataset)
    save_metadata(args.output, args.distance, args.rounds, args.noise, noise_default=args.noise_default)
