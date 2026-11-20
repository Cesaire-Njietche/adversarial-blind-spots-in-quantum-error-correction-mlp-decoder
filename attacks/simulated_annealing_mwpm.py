"""
Run the SA adversarial attack against the MWPM (PyMatching) reference
decoder itself, sweeping every weight budget W from 1 up to
distance//2 + 1. Mirrors simulated_annealing.py (the MLP attack) as
closely as possible -- same move set, same per-W sweep, same .jsonl
schema -- but targets MWPM instead of a trained MLP checkpoint, so there
is no --config/--checkpoint and no BCE loss.

MWPM has no logits/BCE loss to climb, so the SA objective here is a
"regret" in the matching's own cost units: PyMatching assigns each DEM
error mechanism i a weight w_i = -log(p_i / (1-p_i)) and returns, for a
given syndrome, the minimum-weight correction consistent with it
(`Matching.decode(..., return_weight=True)`). For an injected error
pattern E (used to produce the syndrome being attacked), define

    regret(E) = sum(w_i for i active in E) - weight_of_MWPM's_correction(S)

regret(E) is always >= 0 (MWPM's correction is the minimum-weight one,
and E is itself a valid, generally sub-optimal, explanation of the same
syndrome). The larger regret(E), the more strongly MWPM prefers a
cheaper alternative explanation over the truth E -- exactly the
condition under which it is likely to end up in the wrong logical
sector. The SA climbs regret(E) with the same swap move set as the MLP
attack; whether the attack actually succeeded for a given restart is
still checked exactly (MWPM's predicted logical bit vs. the true label).

Arguments (CLI):
    --distance        surface code distance to attack MWPM at
    --rounds          number of syndrome-measurement rounds (default: 9)
    --noise-type      "depolarizing" or "circuit-level"
    --noise-default   noise parameter value used to build the DEM
    --n-restarts      independent SA chains per weight budget (default: 50)
    --seed            random seed (default: 0)
    --output          path to the .jsonl file to write results to

Output: same schema as simulated_annealing.py --
    header: {"type": "header", "distance", "rounds", "noise_type",
              "noise_default", "n_restarts", "seed"}
    result: {"type": "result", "weight", "attack_rate", "catalog",
              "catalog_labels", "best_S", "best_label"}
    "catalog" is the list of syndromes that fooled MWPM for that W (each a
    list of 0/1 ints); "catalog_labels" is the matching list of
    ground-truth labels, same order -- lets another decoder (e.g. the MLP)
    be evaluated against these exact (syndrome, label) pairs later, see
    analysis/evaluate_mwpm_catalog_on_mlp.py. "best_S"/"best_label" are the
    single highest-regret syndrome found across all restarts for that W.
"""
import argparse
import json
import os
import sys

import numpy as np
import pymatching
import stim

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
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


def edge_weights_of(e_instructions):
    """-log(p/(1-p)) per DEM error mechanism -- the same per-edge weight PyMatching
    uses internally, so it's directly comparable to Matching.decode's return_weight."""
    probs = np.array([instr.args_copy()[0] for instr in e_instructions])
    return -np.log(probs / (1.0 - probs))


def attack(
    matcher,
    dem,
    W: int,                    # error budget. Hamming weight of E, fixed
    T0: float      = 2.0,      # initial annealing temperature
    T_min: float   = 0.1,      # stopping temperature
    cooling: float = 0.995,    # geometric cooling rate gamma
    steps_per_T: int = 20,     # metropolis steps per temperature level
    n_restarts: int  = 5,      # independent SA chains
    seed: int        = 0,
):
    """
    Finds the error pattern E* in A_W = {E : |E| = W} that maximises
    regret(E) (see module docstring), i.e. the pattern for which MWPM's
    chosen correction looks cheapest relative to the truth.

    Move set: swap (remove one active fault, add one inactive fault)
    → preserves |E| = W exactly throughout.

    regret(E) is only a noisy proxy for "does this syndrome actually fool
    MWPM" (unlike the MLP's BCE loss, which tracks its own decision
    boundary almost exactly): the single highest-regret state visited in a
    restart is not guaranteed to be a real failure, and a real failure
    visited earlier in the same restart is not guaranteed to have the
    highest regret. So every visited state is checked for an actual
    failure (MWPM's prediction vs. the true label) as the search goes,
    instead of only checking the final highest-regret state -- otherwise
    genuine failures found along the way get silently dropped. (Confirmed
    empirically: plain random weight-2 sampling under circuit-level noise
    found MWPM failures at ~5% while only checking the regret-maximizer
    per restart found none.)

    Returns
    -------
    adversarial_catalog   : list of (S, label) pairs -- syndromes that fooled MWPM
    attack_rate           : float -- attack success rate across restarts
    best_S / best_label   : highest-regret CONFIRMED failure found across all
                             restarts, falling back to the plain highest-regret
                             state (whether or not it's a real failure) if no
                             restart found one at all
    """
    rng = np.random.default_rng(seed)

    adversarial_catalog = []
    attack_rate = 0
    best_fooling_S, best_fooling_label = None, None
    best_fooling_regret = -np.inf
    best_S, best_label = None, None          # fallback if nothing ever fools MWPM
    best_regret = -np.inf
    n_detectors = dem.num_detectors # N_ancillas X N_rounds

    # extracted once per attack() call (not once per restart/step): it only
    # depends on dem, so re-deriving it from dem.flattened() on every
    # Metropolis step was pure wasted work, and the dominant cost of the
    # whole attack.
    e_instructions = extract_error_instructions_from_dem(dem)
    L = len(e_instructions)
    edge_weight = edge_weights_of(e_instructions)

    def regret_and_pred(E, S):
        pred, weight_found = matcher.decode(S.astype(np.uint8), return_weight=True)
        regret = edge_weight[E == 1].sum() - weight_found
        return regret, int(pred[0])

    for restart in range(n_restarts):

        # -- initialise: random pattern of exact weight W
        E = np.zeros(L, dtype=np.int8)
        init_idx = rng.choice(L, size=W, replace=False)
        E[init_idx] = 1

        S, label = E_to_synd_and_label(dem, E, e_instructions)
        curr_regret, curr_pred = regret_and_pred(E, S)

        local_best_regret = curr_regret
        local_best_S      = S.copy()
        local_best_label  = label

        # best CONFIRMED failure seen so far in THIS restart (checked at
        # every visited state, not just the regret-maximizer)
        restart_fooled_S      = None
        restart_fooled_label  = None
        restart_fooled_regret = -np.inf

        def record_if_fooled(S_, label_, pred_, regret_):
            nonlocal restart_fooled_S, restart_fooled_label, restart_fooled_regret
            if pred_ != int(label_) and regret_ > restart_fooled_regret:
                restart_fooled_S      = S_.copy()
                restart_fooled_label  = label_
                restart_fooled_regret = regret_

        record_if_fooled(S, label, curr_pred, curr_regret)

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

                S_new, label_new = E_to_synd_and_label(dem, E_new, e_instructions)
                new_regret, new_pred = regret_and_pred(E_new, S_new)

                # metropolis–Hastings acceptance
                delta = new_regret - curr_regret
                if delta > 0 or rng.random() < np.exp(delta / T):
                    E, curr_regret, curr_pred = E_new, new_regret, new_pred
                    S, label = S_new, label_new

                    record_if_fooled(S, label, curr_pred, curr_regret)

                    if curr_regret > local_best_regret:
                        local_best_regret = curr_regret
                        local_best_S      = S.copy()
                        local_best_label  = label

            T *= cooling   # geometric cooling

        # update the plain regret-maximizer fallback across restarts
        if local_best_regret > best_regret:
            best_S      = local_best_S.copy()
            best_label  = local_best_label
            best_regret = local_best_regret

        # update adversarial catalog + attack success rate across restarts,
        # using the best CONFIRMED failure found during this restart (if any)
        if restart_fooled_S is not None:
            adversarial_catalog.append((restart_fooled_S, restart_fooled_label))
            attack_rate += 1

            if restart_fooled_regret > best_fooling_regret:
                best_fooling_regret = restart_fooled_regret
                best_fooling_S      = restart_fooled_S.copy()
                best_fooling_label  = restart_fooled_label

        if (restart + 1) % 1 == 0:
            print(f"  restart {restart+1}/{n_restarts} | "
                  f"best regret so far: {best_regret:.4f} | "
                  f"confirmed failures so far: {len(adversarial_catalog)}")

    attack_rate /= n_restarts

    if best_fooling_S is not None:
        best_S, best_label = best_fooling_S, best_fooling_label

    return adversarial_catalog, attack_rate, best_S, best_label


def build_dem(distance, rounds, noise_type, noise_default):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        **nm(noise_type=noise_type, default=noise_default),
    )
    return circuit.detector_error_model(decompose_errors=True)


def run_full_attack_for_model(matcher, dem, distance, output_path, rounds=9, noise_type=None, noise_default=None, n_restarts=50, seed=0):
    """
    Attack MWPM for every weight budget in [1, distance//2 + 1], writing
    one JSON line to output_path as soon as each W finishes. output_path is
    overwritten from scratch (not resumed/appended to) -- re-running this
    always starts the file clean.
    """
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
                matcher, dem, w,
                n_restarts=n_restarts, seed=seed,
            )
            record = {
                "type": "result",
                "weight": w,
                "attack_rate": attack_rate,
                "catalog": [s.tolist() for s, _ in catalog],
                "catalog_labels": [int(l) for _, l in catalog],
                "best_S": best_S.tolist(),
                "best_label": int(best_label),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SA attack against the MWPM reference decoder, sweeping every weight budget.")
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

    dem = build_dem(args.distance, args.rounds, args.noise_type, args.noise_default)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    run_full_attack_for_model(
        matcher, dem, args.distance, args.output,
        rounds=args.rounds, noise_type=args.noise_type, noise_default=args.noise_default,
        n_restarts=args.n_restarts, seed=args.seed,
    )


if __name__ == "__main__":
    main()
