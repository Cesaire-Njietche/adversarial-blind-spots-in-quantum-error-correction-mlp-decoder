import h5py
import matplotlib.pyplot as plt
import argparse
from typing import List
from typing import Dict
import itertools
from tqdm import tqdm 
import numpy as np
"""
This code will produce statistics from a .json file written by disagreement_map.py.
More specifically, it will for every disagreement map found in the json file 
it will compute the statistics (boxplots) for every of the following features : 
- Syndrome weight — sum(S), how many detectors fired. 
- Spatial spread — how SPATIALLY dispersed the fired detectors are (TODO)

- Temporal extent — how spread out in rounds the fired detectors are(TODO)

- Boundary proximity — closeness to the physical border of the decoder(TODO)

"""

NOISE_MODELS = ["Depolarizing", "FT"]
DISTANCES = [3, 5, 7]
PARAMS = ["005", "01"]


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



def compute_boxplot_Weight(syndromes:List, output_files_path:str, group_path:str)->None:
    weights = syndromes.sum(axis=1)/len(syndromes[0])
    fig, ax = plt.subplots()
    ax.boxplot(weights)
    ax.set_title(f"Normalized weight of MLP blind spots\n{group_path}")
    ax.set_ylabel("Fraction of detectors fired")
    fig.savefig(f"{output_files_path}/{group_path.replace('/', '_')}_WeightBoxplot.png")
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description="Build an MWPM vs MLP disagreement map.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_configs = list(itertools.product(NOISE_MODELS, DISTANCES, PARAMS))
    for noise_folder, distance, param_folder in tqdm(all_configs):
        group_path = f"{noise_folder}/d{distance}/{param_folder}"
        tqdm.write(f"Doing config for {group_path}")

        dictionary = read_h5_table(args.input, group_path)
        compute_boxplot_Weight(dictionary["mlp_blind_spot"], args.output_folder, group_path)
