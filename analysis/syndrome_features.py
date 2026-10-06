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
This code will produce statistics from a .json file written by disagreement_map.py.
More specifically, it will for every disagreement map found in the json file 
it will compute the statistics (boxplots) for every of the following features : 
- Syndrome weight — sum(S), how many detectors fired. 
- Spatial spread — how SPATIALLY dispersed the fired detectors are (TODO)

- Temporal extent — how spread out in rounds the fired detectors are(TODO)

- Boundary proximity — closeness to the physical border of the decoder(TODO)


It will compare all the previous statistics of the syndromes to the "both correct" categories that will allow to see 
what are the differences between the blind syndromes of the model and what the model can correctly solve.
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


def read_h5_table(file_path:str, group_path:str) ->Dict :
    """
    @Input : 
        file_path = path of the h5 file containing the disagreement map for every model 
        group_path = the path of the group corresponding to the table we want to read. 
                    Its form is :     <NoiseModel>/d<distance>/<param>/
                    
    
    @Output : 
        A dictionary of the form : 
        {
            "noise_model"     : <NoiseModel>,
            "distance"        : <distance>,
            "param"           : <param>,
            both_correct     : array of syndromes where both decoders are correct
            both_wrong       : array of syndromes where both decoders are wrong
            mlp_blind_spot   : array of syndromes where MWPM is correct but MLP is wrong
            mwpm_blind_spot  : array of syndromes where MLP is correct but MWPM is wrong
        }

    
    """
    group_path_split = group_path.split("/")
    noise_model = group_path_split[0]
    distance = group_path_split[1]
    param = group_path_split[2]

    with h5py.File(file_path, "r") as f:
        group = f[group_path]
        #attention : on [:] pour load into memory sinon ca s'efface...
        both_correct = group["both_correct"][:]      # numpy array, shape (#samples, num_detectors)
        mwpm_blind_spot = group["mwpm_blind_spot"][:]# numpy array
        both_wrong = group["both_wrong"][:]         # plain int
        mlp_blind_spot = group["mlp_blind_spot"][:] # plain int
    
    return {
                "noise_model"     : noise_model,
                "distance"        : distance,
                "param"           : param,
                "both_correct"    : both_correct,
                "both_wrong"      : both_wrong,
                "mlp_blind_spot"  : mlp_blind_spot,
                "mwpm_blind_spot" : mwpm_blind_spot
            }



def compute_boxplot_Weight(syndromes_mlp_blindpsots:List, syndromes_both_correct:List, output_files_path:str, group_path:str)->None:
    weights_blindpsot = syndromes_mlp_blindpsots.sum(axis=1)/len(syndromes_mlp_blindpsots[0])
    weights_both_correct = syndromes_both_correct.sum(axis=1)/len(syndromes_both_correct[0])
    fig, ax = plt.subplots()
    ax.boxplot(
        [weights_blindpsot, weights_both_correct],
        label=["MLP blind spots", "Both Correct"]
    )
    ax.legend()
    ax.set_title(
        r"Boxplot: Normalized weight of $\mathcal{P}_{blind}$ spots and $\mathcal{P}_{both}$"
        f"\n{group_path}"
    )    
    ax.set_title(f"Normalized weight of MLP blind spots\n{group_path}")
    ax.set_ylabel("Fraction of detectors fired")
    os.makedirs(f"{output_files_path}/Weight", exist_ok=True)
    fig.savefig(f"{output_files_path}/Weight/{group_path.replace('/', '_')}_WeightBoxplot.png")
    plt.close(fig)


def compute_spartial_spread_heatmap(syndromes_mlp_blindpsots:List, syndromes_both_correct:List, output_files_path:str, group_path:str, coord:Dict)->None:
    #only showing the ones that fired :
    space_coord_mlp_blindspot_fired = []
    space_coord_both_correct_fired = []

    for i in range(len(syndromes_mlp_blindpsots)):
        space_coord_mlp_blindspot_fired.append([coord[j][0:2] for j in range(len(syndromes_mlp_blindpsots[i])) if syndromes_mlp_blindpsots[i][j] == 1])
        space_coord_both_correct_fired.append([coord[j][0:2] for j in range(len(syndromes_both_correct[i])) if syndromes_both_correct[i][j] == 1])

    ## will keep indx if needed and build info for heatmap sepparatly
    # for now we have [[x1,y1],...]
    #+1 cuz 0 indexed so we need idx+1 elements in the grid
    width = max(xy[0] for xy in coord.values()) + 1
    lenght = max(xy[1] for xy in coord.values()) + 1

    def build_grid(space_coord_fired, n_samples):
        grid = np.zeros((int(lenght), int(width)))
        for fired in space_coord_fired:
            for x, y in fired:
                grid[int(y), int(x)] += 1
        return grid / n_samples

    grid_blindspot = build_grid(space_coord_mlp_blindspot_fired, len(syndromes_mlp_blindpsots))
    grid_both_correct = build_grid(space_coord_both_correct_fired, len(syndromes_both_correct))

    # do the heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(grid_blindspot, vmin=0, cmap="RdBu_r", origin="lower")
    axes[0].set_title(r"$\mathcal{P}_{blind}$ spots")
    fig.colorbar(im0, ax=axes[0], label="Firing frequency")

    im1 = axes[1].imshow(grid_both_correct, vmin=0, cmap="RdBu_r", origin="lower")
    axes[1].set_title(r"$\mathcal{P}_{both}$")
    fig.colorbar(im1, ax=axes[1], label="Firing frequency")
    fig.suptitle(
        r"Heatmap: Space Coordinates of $\mathcal{P}_{blind}$ spots and $\mathcal{P}_{both}$"
        f"\n{group_path}"
    )
    os.makedirs(f"{output_files_path}/HeatMap", exist_ok=True)
    fig.savefig(f"{output_files_path}/HeatMap/{group_path.replace('/', '_')}_SpaceHeatmap.png")
    plt.close(fig)



    return None


def compute_temporal_histogram(syndromes_mlp_blindpsots:List, syndromes_both_correct:List, output_files_path:str, group_path:str, coord:Dict)->None:
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
        return counts / len(syndromes) / detectors_per_round

    freq_blindspot = firing_by_round(syndromes_mlp_blindpsots)
    freq_both_correct = firing_by_round(syndromes_both_correct)

    rounds = np.arange(n_rounds)
    bar_width = 0.4

    fig, ax = plt.subplots()
    ax.bar(rounds - bar_width / 2, freq_blindspot, width=bar_width, label=r"$\mathcal{P}_{blind}$ spots", color="tab:red")
    ax.bar(rounds + bar_width / 2, freq_both_correct, width=bar_width, label=r"$\mathcal{P}_{both}$", color="tab:blue")
    ax.legend()
    ax.set_xticks(rounds)
    ax.set_xlabel("Round")
    ax.set_ylabel("Fraction of that round's detectors fired")
    ax.set_title(
        r"Temporal extent: detectors fired per round for $\mathcal{P}_{blind}$ spots and $\mathcal{P}_{both}$"
        f"\n{group_path}"
    )

    os.makedirs(f"{output_files_path}/Temporal", exist_ok=True)
    fig.savefig(f"{output_files_path}/Temporal/{group_path.replace('/', '_')}_TemporalHistogram.png")
    plt.close(fig)

    return None


def compute_boundary_proximity_boxplot(syndromes_mlp_blindpsots:List, syndromes_both_correct:List, output_files_path:str, group_path:str, coord:Dict)->None:
    # distance (in the 2D spatial lattice) of every detector to the nearest edge of the grid
    #+1 cuz 0 indexed so we need idx+1 elements in the grid
    width = max(xy[0] for xy in coord.values()) + 1
    lenght = max(xy[1] for xy in coord.values()) + 1
    boundary_dist = {
        #minimum distance from one of the border (either distance from left/right/up/down and get smallest)
        j: min(xy[0], width - 1 - xy[0], xy[1], lenght - 1 - xy[1])
        for j, xy in coord.items()
    }

    def min_boundary_dist_per_sample(syndromes):
        dists = []
        for sample in syndromes:
            fired = [boundary_dist[j] for j in range(len(sample)) if sample[j] == 1]
            # get the min of all minimum distances from all samples
            dists.append(min(fired) if fired else np.nan)
        return np.array(dists)

    blindspot_dist = min_boundary_dist_per_sample(syndromes_mlp_blindpsots)
    both_correct_dist = min_boundary_dist_per_sample(syndromes_both_correct)

    # samples with zero fired detectors have no "closest fired detector" -> drop them
    blindspot_dist = blindspot_dist[~np.isnan(blindspot_dist)]
    both_correct_dist = both_correct_dist[~np.isnan(both_correct_dist)]

    fig, ax = plt.subplots()
    ax.boxplot(
        [blindspot_dist, both_correct_dist],
        tick_labels=["MLP blind spots", "Both Correct"]
    )
    ax.set_title(
        r"Boundary proximity: distance to nearest edge of $\mathcal{P}_{blind}$ spots and $\mathcal{P}_{both}$"
        f"\n{group_path}"
    )
    ax.set_ylabel("Distance of closest fired detector to grid boundary")

    os.makedirs(f"{output_files_path}/Boundary", exist_ok=True)
    fig.savefig(f"{output_files_path}/Boundary/{group_path.replace('/', '_')}_BoundaryBoxplot.png")
    plt.close(fig)

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Build an MWPM vs MLP disagreement map.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--rounds", type=str, default=9)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_configs = list(itertools.product(NOISE_MODELS, DISTANCES, PARAMS))
    coord_cache = {}
    for noise_folder, distance, param_folder in tqdm(all_configs):
        group_path = f"{noise_folder}/d{distance}/{param_folder}"
        tqdm.write(f"Doing config for {group_path}")


        dictionary = read_h5_table(args.input, group_path)
        syndromes_mlp_blindpsots = dictionary["mlp_blind_spot"]
        syndromes_both_correct = dictionary["both_correct"]

        final_output = args.output_folder
        if final_output[-1] == "/":
            final_output = final_output[:-1]
        compute_boxplot_Weight(syndromes_mlp_blindpsots, syndromes_both_correct, final_output, group_path)

        #build circuit
        noise_type = NOISE_FOLDER_TO_TYPE[noise_folder]
        noise_param = PARAM_FOLDER_TO_VALUE[param_folder]

        # detector coordinates only depend on (noise_folder, distance, rounds), not on the error rate,
        # so we cache them across the two PARAMS values to avoid re-decomposing the error model.
        cache_key = (noise_folder, distance)
        if cache_key in coord_cache:
            coord = coord_cache[cache_key]
        else:
            circuit = create_circuit(distance=distance, rounds=args.rounds, noise_type=noise_type, noise_default=noise_param)
            #recall that dem = detetor error model contains logic for the actual error to syndroms (and thus inverse)
            dem = circuit.detector_error_model(decompose_errors=True)

            coord = dem.get_detector_coordinates() # dictonary key = every detector (there are (r^2-1)*d detectors,  value = [x,y,t])
            coord_cache[cache_key] = coord

        compute_spartial_spread_heatmap(syndromes_mlp_blindpsots, syndromes_both_correct, final_output, group_path, coord)
        compute_temporal_histogram(syndromes_mlp_blindpsots, syndromes_both_correct, final_output, group_path, coord)
        compute_boundary_proximity_boxplot(syndromes_mlp_blindpsots, syndromes_both_correct, final_output, group_path, coord)
