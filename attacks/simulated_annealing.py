"""
Run the SA adversarial attack against ONE trained decoder, sweeping every
weight budget W from 1 up to distance//2 + 1. --output is overwritten
from scratch at the start of the run (not resumed), and each weight
budget's result is written to it, as soon as it's ready, as one JSON
Lines record -- so re-running this script against the same model starts
that model's file clean rather than appending/duplicating.

Arguments (CLI):
    --config          path to the trained decoder's config.json
    --checkpoint      path to the trained decoder's model.pth
    --distance        surface code distance the model was trained on
    --rounds          number of syndrome-measurement rounds (default: 9)
    --noise-type      "depolarizing" or "circuit-level"
    --noise-default   noise parameter value used to build the DEM
    --n-restarts      independent SA chains per weight budget (default: 50)
    --seed            random seed (default: 0)
    --output          path to the .jsonl file to write results to

Output: the first line of --output is a "header" record describing the
model/noise config this file is for (so the file is self-describing even
without its name); every following line is a "result" record for one
weight budget W:
    header: {"type": "header", "distance", "rounds", "noise_type",
              "noise_default", "n_restarts", "seed"}
    result: {"type": "result", "weight", "attack_rate", "catalog",
              "best_S", "best_label"}
    "catalog" is the list of syndromes that fooled the decoder for that W
    (each a list of 0/1 ints); "best_S"/"best_label" are the single
    highest-BCE-loss syndrome found across all restarts for that W.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn
import stim

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "decoder"))

from model import build_model_from_checkpoint_as_eval
from noise_models import generate_noise_constants as nm


def extract_error_instructions_from_dem(dem):
    """
    Extract error instructions from dem. 

    The detector error model (dem) is a list of all the independent error mechanisms in a circuit
    as well as their symptoms (which detectors the fire/set off).

    A detector is a comparison between 2 ancillas measurment.

    Parameters
        ----------
        dem              : the detector error model
    
        Returns
        -------
        e_instructions : error instructions list
    """
    e_instructions = []

    for instr in dem.flattened(): # flattened because we need to extract error from within the circuit rounds
        if instr.type == "error":
             e_instructions.append(instr)

    return e_instructions

def E_to_synd_and_label(dem, E, e_instructions=None):
    """
    Given a binary error pattern E over DEM error mechanisms,
    compute the resulting syndrome and logical observable by XOR-ing
    the effects of all active error mechanisms.

    Parameters
        ----------
        dem              : the detector error model
        E                : binary vector, shape (L,), one entry per DEM error mechanism
        e_instructions   : optional, the (expensive-to-compute) result of
                            extract_error_instructions_from_dem(dem). Pass
                            it in explicitly when calling this many times
                            against the same dem (e.g. in a hot loop) to
                            avoid re-extracting it from scratch every call;
                            left as None it's derived from dem as before.

        Returns
        -------
        syndrome : binary array, shape (num_detectors,)
        label    : int in {0, 1}  -- 1 if logical X error occurred
        L        : total number of error locations (len of E which is the len of dem error instructions )
    """
    num_detectors = dem.num_detectors # N_ancillas X N_rounds
    num_observables = dem.num_observables # 1

    synd = np.zeros(num_detectors, dtype=np.int8)
    logical = np.zeros(num_observables, dtype=np.int8)

    if e_instructions is None:
        e_instructions = extract_error_instructions_from_dem(dem)

    for i, e_instr in enumerate(e_instructions):
        if E[i] == 1:
            # XOR the detectors this error flips. A target is a detector
            for target in e_instr.targets_copy():
                if target.is_relative_detector_id(): 
                    synd[target.val] ^= 1
                elif target.is_logical_observable_id():
                    
                    logical[target.val] ^= 1

    label = float(logical[0])  # 1 = logical X error occurred
    return synd.astype(np.float32), label


def attack(
    model,
    dem,
    criterion,
    W: int,                    # error budget. Hamming weight of E, fixed
    T0: float     = 2.0,       # initial annealing temperature
    T_min: float  = 0.1,       # stopping temperature
    cooling: float = 0.995,    # geometric cooling rate gamma
    steps_per_T: int = 20,     # metropolis steps per temperature level
    n_restarts: int  = 5,      # independent SA chains
    seed: int        = 0,
):
    """
    Finds the error pattern E* in A_W = {E : |E| = W} that maximises
    the Binary Cross Entropy (BCE) loss of the decoder, i.e. the most adversarial X-error pattern
    within the given budget.

    Move set: swap (remove one active fault, add one inactive fault)
    → preserves |E| = W exactly throughout.

    Returns
    -------
    adversarial_catalog   : array of synd   — worst-case error pattern
    attack_rate   : float                   — Attack success rate
    best_S      : float32 array             — syndrome produced by the best E
    best_label  : int                       — ground-truth label at best E
    """
    rng = np.random.default_rng(seed)

    adversarial_catalog = []
    attack_rate  = 0
    best_S     = None
    best_label = None
    best_loss = -np.inf
    n_detectors = dem.num_detectors # N_ancillas X N_rounds

    # extracted once per attack() call (not once per restart/step): it only
    # depends on dem, so re-deriving it from dem.flattened() on every
    # Metropolis step (as E_to_synd_and_label used to do internally) was
    # pure wasted work, and the dominant cost of the whole attack.
    e_instructions = extract_error_instructions_from_dem(dem)
    L = len(e_instructions)

    for restart in range(n_restarts):

        # -- initialise: random pattern of exact weight W
        E = np.zeros(L, dtype=np.int8)
        init_idx = rng.choice(L, size=W, replace=False)
        E[init_idx] = 1

        S, label = E_to_synd_and_label(dem, E, e_instructions)

        # transform the syndrome vector S and the ground truth label to fit in the model
        label = torch.tensor([label])
        S = torch.from_numpy(S)
        S = S.view(n_detectors)
        label = label.view(1)

        # compute the BCE loss
        curr_loss  = criterion(model(S), label)

        local_best_loss = curr_loss
        local_best_S    = S.clone()
        local_best_label = label

        T = T0

         # -- annealing loop 
        while T > T_min:
            for _ in range(steps_per_T):

                # swap move: keeps |E| = W
                on_idx  = np.flatnonzero(E == 1)
                off_idx = np.flatnonzero(E == 0)
                i_remove = rng.choice(on_idx)
                i_add    = rng.choice(off_idx)

                E_new = E.copy()
                E_new[i_remove] = 0
                E_new[i_add]    = 1

                S_new, label_new = E_to_synd_and_label(
                    dem, E_new, e_instructions)

                # transform the syndrome vector S and the ground truth label to fit in the model
                label_new = torch.tensor([label_new])
                S_new = torch.from_numpy(S_new)
                S_new = S_new.view(n_detectors)
                label_new = label_new.view(1)

                new_loss = criterion(model(S_new), label_new)

                # metropolis–Hastings acceptance
                delta = new_loss - curr_loss
                delta = delta.item()
                if delta > 0 or rng.random() < np.exp(delta / T):
                    E, curr_loss = E_new, new_loss
                    S, label     = S_new, label_new

                    if curr_loss > local_best_loss:
                        local_best_loss  = curr_loss
                        local_best_S     = S.clone()
                        local_best_label = label

            T *= cooling   # geometric cooling

        # update best_S and best_label across restarts
        if local_best_loss > best_loss:
            best_S  = local_best_S.clone()
            best_label = local_best_label
            best_loss = local_best_loss
            
        # update adversarial catalog accross restarts
        # compute attack success rate across restarts
        pred = (model(local_best_S) > 0.0).float()
        if pred != local_best_label:
            adversarial_catalog.append(local_best_S)
            attack_rate += 1

        if (restart + 1) % 1 == 0:
            print(f"  restart {restart+1}/{n_restarts} | "
                  f"best BCE loss so far: {best_loss:.4f}")

    attack_rate /= n_restarts

    return adversarial_catalog, attack_rate, best_S, best_label






def build_dem(distance, rounds, noise_type, noise_default):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        **nm(noise_type=noise_type, default=noise_default),
    )
    return circuit.detector_error_model(decompose_errors=True)


def run_full_attack_for_model(model, dem, distance, output_path, rounds=9, noise_type=None, noise_default=None, n_restarts=50, seed=0):
    """
    Attack `model` for every weight budget in [1, distance//2 + 1], writing
    one JSON line to output_path as soon as each W finishes. output_path is
    overwritten from scratch (not resumed/appended to) -- re-running this
    model always starts its file clean.
    """
    criterion = nn.BCEWithLogitsLoss()

    with open(output_path, "w") as f:
        header = {
            "type": "header",
            "distance": distance,
            "rounds": rounds,
            "noise_type": noise_type,
            "noise_default": noise_default,
            "n_restarts": n_restarts,
            "seed": seed,
        }
        f.write(json.dumps(header) + "\n")
        f.flush()

        for w in range(1, distance//2 + 1 + 1):
            print(f"####   Weight (W = {w}) for distance {distance}    ####\n")
            catalog, attack_rate, best_S, best_label = attack(
                model, dem, criterion, w,
                n_restarts=n_restarts, seed=seed,
            )
            record = {
                "type": "result",
                "weight": w,
                "attack_rate": attack_rate,
                "catalog": [s.tolist() for s in catalog],
                "best_S": best_S.tolist(),
                "best_label": int(best_label.item()),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SA attack against one trained decoder, sweeping every weight budget.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--noise-type", default="depolarizing")
    parser.add_argument("--noise-default", type=float, required=True)
    parser.add_argument("--n-restarts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    model = build_model_from_checkpoint_as_eval(
        args.config, args.checkpoint, args.distance, args.rounds
    )
    dem = build_dem(args.distance, args.rounds, args.noise_type, args.noise_default)

    run_full_attack_for_model(
        model, dem, args.distance, args.output,
        rounds=args.rounds, noise_type=args.noise_type, noise_default=args.noise_default,
        n_restarts=args.n_restarts, seed=args.seed,
    )


if __name__ == "__main__":
    main()