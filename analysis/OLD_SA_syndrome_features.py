import json
import h5py
import matplotlib.pyplot as plt
import argparse
from typing import List
from typing import Dict
import itertools
from tqdm import tqdm
import numpy as np
import sys, os
import stim
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

from generate_datasets import create_circuit

"""
This code produces the same statistics as syndrome_features.py (boxplots for
syndrome weight, spatial spread, temporal extent, boundary proximity) but
for the SA adversarial attack catalogs (results/attacks/*.jsonl) written by
attacks/simulated_annealing.py, instead of the disagreement map's
naturally-occurring blind spots.

For each config, the adversarial catalog C = union_W C_W (union of every
weight budget's successful adversarial syndromes) is compared against the
natural MLP blind spots P_blind, read from the same disagreement map .h5
used by syndrome_features.py -- so we can see whether deliberately crafted,
minimal-weight adversarial failures look like naturally-occurring ones.
"""

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


def read_sa_catalog(jsonl_path: str) -> Dict:
    """
    @Input :
        jsonl_path = path to one <NoiseModel>_d<D>_<param>.jsonl file
                     written by attacks/simulated_annealing.py

    @Output :
        A dictionary of the form :
        {
            "distance"     : int,
            "rounds"       : int,
            "noise_type"   : "depolarizing" or "circuit-level",
            "noise_default": float,
            "catalog"      : numpy array, shape (#adversarial syndromes, num_detectors)
                              = union of every weight budget's catalog
        }
    """
    header = None
    catalog = []

    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record["type"] == "header":
                header = record
            elif record["type"] == "result":
                catalog.extend(record["catalog"])

    if header is None:
        raise ValueError(f"{jsonl_path} has no header record")

    return {
        "distance": header["distance"],
        "rounds": header["rounds"],
        "noise_type": header["noise_type"],
        "noise_default": header["noise_default"],
        "catalog": np.array(catalog),
    }


def read_h5_blindspots(file_path: str, group_path: str) -> np.ndarray:
    """
    @Input :
        file_path = path of the h5 file containing the disagreement map for every model
        group_path = the path of the group corresponding to the table we want to read.
                    Its form is :     <NoiseModel>/d<distance>/<param>/

    @Output :
        numpy array, shape (#samples, num_detectors) : natural MLP blind spots (P_blind)
    """
    with h5py.File(file_path, "r") as f:
        group = f[group_path]
        mlp_blind_spot = group["mlp_blind_spot"][:]

    return mlp_blind_spot


def compute_boxplot_Weight(syndromes_adversarial:List, syndromes_blindspots:List, output_files_path:str, group_path:str)->None:
    weights_adversarial = syndromes_adversarial.sum(axis=1)/len(syndromes_adversarial[0])
    weights_blindspots = syndromes_blindspots.sum(axis=1)/len(syndromes_blindspots[0])
    fig, ax = plt.subplots()
    ax.boxplot(
        [weights_adversarial, weights_blindspots],
        label=["Adversarial (SA)", "Natural Blind Spots"]
    )
    ax.legend()
    ax.set_title(
        r"Boxplot: Normalized weight of $\mathcal{C}$ and $\mathcal{P}_{blind}$"
        f"\n{group_path}"
    )
    ax.set_ylabel("Fraction of detectors fired")
    os.makedirs(f"{output_files_path}/Weight", exist_ok=True)
    fig.savefig(f"{output_files_path}/Weight/SA_{group_path.replace('/', '_')}_WeightBoxplot.png")
    plt.close(fig)


def compute_spartial_spread_heatmap(syndromes_adversarial:List, syndromes_blindspots:List, output_files_path:str, group_path:str, coord:Dict)->None:
    #only showing the ones that fired :
    space_coord_adversarial_fired = []
    space_coord_blindspots_fired = []

    for i in range(len(syndromes_adversarial)):
        space_coord_adversarial_fired.append([coord[j][0:2] for j in range(len(syndromes_adversarial[i])) if syndromes_adversarial[i][j] == 1])
    for i in range(len(syndromes_blindspots)):
        space_coord_blindspots_fired.append([coord[j][0:2] for j in range(len(syndromes_blindspots[i])) if syndromes_blindspots[i][j] == 1])

    #+1 cuz 0 indexed so we need idx+1 elements in the grid
    width = max(xy[0] for xy in coord.values()) + 1
    lenght = max(xy[1] for xy in coord.values()) + 1

    def build_grid(space_coord_fired, n_samples):
        grid = np.zeros((int(lenght), int(width)))
        for fired in space_coord_fired:
            for x, y in fired:
                grid[int(y), int(x)] += 1
        return grid / n_samples

    grid_adversarial = build_grid(space_coord_adversarial_fired, len(syndromes_adversarial))
    grid_blindspots = build_grid(space_coord_blindspots_fired, len(syndromes_blindspots))

    # do the heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(grid_adversarial, vmin=0, cmap="RdBu_r", origin="lower")
    axes[0].set_title(r"$\mathcal{C}$ (Adversarial)")
    fig.colorbar(im0, ax=axes[0], label="Firing frequency")

    im1 = axes[1].imshow(grid_blindspots, vmin=0, cmap="RdBu_r", origin="lower")
    axes[1].set_title(r"$\mathcal{P}_{blind}$ (Natural)")
    fig.colorbar(im1, ax=axes[1], label="Firing frequency")
    fig.suptitle(
        r"Heatmap: Space Coordinates of $\mathcal{C}$ and $\mathcal{P}_{blind}$"
        f"\n{group_path}"
    )
    os.makedirs(f"{output_files_path}/HeatMap", exist_ok=True)
    fig.savefig(f"{output_files_path}/HeatMap/SA_{group_path.replace('/', '_')}_SpaceHeatmap.png")
    plt.close(fig)

    return None


def compute_temporal_histogram(syndromes_adversarial:List, syndromes_blindspots:List, output_files_path:str, group_path:str, coord:Dict)->None:
    #only showing the ones that fired, keeping the round (t) coordinate :
    n_rounds = int(max(xyt[2] for xyt in coord.values())) + 1

    # how many detectors actually exist at each round
    detectors_per_round = np.zeros(n_rounds)
    for xyt in coord.values():
        detectors_per_round[int(xyt[2])] += 1

    def firing_by_round(syndromes):
        counts = np.zeros(n_rounds)
        for sample in syndromes:
            for j in range(len(sample)):
                if sample[j] == 1:
                    counts[int(coord[j][2])] += 1
        #basically divide by the number of syndromes and the number of detectors per round (which the height per x axis) to get the fraction of that round's detectors that fired
        return counts / len(syndromes) / detectors_per_round

    freq_adversarial = firing_by_round(syndromes_adversarial)
    freq_blindspots = firing_by_round(syndromes_blindspots)

    rounds = np.arange(n_rounds)
    bar_width = 0.4

    fig, ax = plt.subplots()
    ax.bar(rounds - bar_width / 2, freq_adversarial, width=bar_width, label=r"$\mathcal{C}$ (Adversarial)", color="tab:red")
    ax.bar(rounds + bar_width / 2, freq_blindspots, width=bar_width, label=r"$\mathcal{P}_{blind}$ (Natural)", color="tab:blue")
    ax.legend()
    ax.set_xticks(rounds)
    ax.set_xlabel("Round")
    ax.set_ylabel("Fraction of that round's detectors fired")
    ax.set_title(
        r"Temporal extent: detectors fired per round for $\mathcal{C}$ and $\mathcal{P}_{blind}$"
        f"\n{group_path}"
    )

    os.makedirs(f"{output_files_path}/Temporal", exist_ok=True)
    fig.savefig(f"{output_files_path}/Temporal/SA_{group_path.replace('/', '_')}_TemporalHistogram.png")
    plt.close(fig)

    return None


def compute_boundary_proximity_boxplot(syndromes_adversarial:List, syndromes_blindspots:List, output_files_path:str, group_path:str, coord:Dict)->None:
    # distance (in the 2D spatial lattice) of every detector to the nearest edge of the grid
    #+1 cuz 0 indexed so we need idx+1 elements in the grid
    width = max(xy[0] for xy in coord.values()) + 1
    lenght = max(xy[1] for xy in coord.values()) + 1
    boundary_dist = {
        #minimum distance from one of the border (either distance from left/right/up/down and get smallest)
        j: min(xy[0], width - 1 - xy[0], xy[1], lenght - 1 - xy[1])
        for j, xy in coord.items()
    }

    def boundary_fraction_per_sample(syndromes):
        # fraction of THIS sample's fired detectors that sit right on the boundary (dist == 0).
        # normalized by how many detectors fired
        fracs = []
        for sample in syndromes:
            fired = [boundary_dist[j] for j in range(len(sample)) if sample[j] == 1]
            fracs.append(sum(d == 0 for d in fired) / len(fired) if fired else np.nan)
        return np.array(fracs)

    adversarial_frac = boundary_fraction_per_sample(syndromes_adversarial)
    blindspots_frac = boundary_fraction_per_sample(syndromes_blindspots)

    # samples with zero fired detectors have no defined fraction -> drop them
    adversarial_frac = adversarial_frac[~np.isnan(adversarial_frac)]
    blindspots_frac = blindspots_frac[~np.isnan(blindspots_frac)]

    fig, ax = plt.subplots()
    ax.boxplot(
        [adversarial_frac, blindspots_frac],
        tick_labels=["Adversarial (SA)", "Natural Blind Spots"]
    )
    ax.set_title(
        r"Boundary proximity: fraction of fired detectors on the edge of $\mathcal{C}$ and $\mathcal{P}_{blind}$"
        f"\n{group_path}"
    )
    ax.set_ylabel("Fraction of fired detectors on grid boundary")

    os.makedirs(f"{output_files_path}/Boundary", exist_ok=True)
    fig.savefig(f"{output_files_path}/Boundary/SA_{group_path.replace('/', '_')}_BoundaryBoxplot.png")
    plt.close(fig)

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Characterize SA adversarial syndromes vs natural MLP blind spots.")
    parser.add_argument("--attacks_dir", type=str, required=True, help="directory containing <NoiseModel>_d<D>_<param>.jsonl files")
    parser.add_argument("--h5_input", type=str, required=True, help="path to the disagreement map .h5 (same as syndrome_features.py --input)")
    parser.add_argument("--output_folder", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_configs = list(itertools.product(NOISE_MODELS, DISTANCES, PARAMS))
    coord_cache = {}

    final_output = args.output_folder
    if final_output[-1] == "/":
        final_output = final_output[:-1]

    for noise_folder, distance, param_folder in tqdm(all_configs):
        group_path = f"{noise_folder}/d{distance}/{param_folder}"
        jsonl_path = f"{args.attacks_dir}/{noise_folder}_d{distance}_{param_folder}.jsonl"

        if not os.path.exists(jsonl_path):
            tqdm.write(f"Skipping {group_path}: no attack file at {jsonl_path}")
            continue

        sa = read_sa_catalog(jsonl_path)
        syndromes_adversarial = sa["catalog"]

        if len(syndromes_adversarial) == 0:
            tqdm.write(f"Skipping {group_path}: adversarial catalog is empty")
            continue

        tqdm.write(f"Doing config for {group_path} ({len(syndromes_adversarial)} adversarial syndromes)")

        syndromes_blindspots = read_h5_blindspots(args.h5_input, group_path)

        compute_boxplot_Weight(syndromes_adversarial, syndromes_blindspots, final_output, group_path)

        # detector coordinates only depend on (noise_folder, distance, rounds), not on the error rate,
        # so we cache them across the two PARAMS values to avoid re-decomposing the error model.
        cache_key = (noise_folder, distance)
        if cache_key in coord_cache:
            coord = coord_cache[cache_key]
        else:
            circuit = create_circuit(
                distance=sa["distance"],
                rounds=sa["rounds"],
                noise_type=sa["noise_type"],
                noise_default=sa["noise_default"],
            )
            #recall that dem = detetor error model contains logic for the actual error to syndroms (and thus inverse)
            dem = circuit.detector_error_model(decompose_errors=True)

            coord = dem.get_detector_coordinates() # dictonary key = every detector (there are (r^2-1)*d detectors,  value = [x,y,t])
            coord_cache[cache_key] = coord

        compute_spartial_spread_heatmap(syndromes_adversarial, syndromes_blindspots, final_output, group_path, coord)
        compute_temporal_histogram(syndromes_adversarial, syndromes_blindspots, final_output, group_path, coord)
        compute_boundary_proximity_boxplot(syndromes_adversarial, syndromes_blindspots, final_output, group_path, coord)
