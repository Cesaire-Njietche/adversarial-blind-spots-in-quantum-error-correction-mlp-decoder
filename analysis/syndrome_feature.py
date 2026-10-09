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
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

from generate_datasets import create_circuit

"""
This code will produce statistics from our syndromes. More specifically it will ingest both the h5 file of the disagreement map and the .jsoonl files of the 
SA and produce statistics for : 
1. Pblind vs Pboth 
2. Pblind vs C (the catalog of SA)

The statistics will be that we will try to fit a logistic regressions where the params are :
- spatial spread
- temporal spread
- weight
- isolation
- boundary proximity

and then produce statistics on the coefficients of the logistic regression to see if the blind spots are correlated with any of these features.
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

FEATURE_NAMES = ["weight", "spread_space", "spread_time", "isolated", "boundary_frac"]

# fixed layout shared by both plots, so cell (i, j) always means the same config
# in the AUC heatmap and in the coefficient grid -- matches the notebook's Figure 1
# convention: rows = distance (3), columns = noise model / physical error rate (4)
ROW_KEYS = [3, 5, 7]
ROW_LABELS = ["d=3", "d=5", "d=7"]
COL_KEYS = [("Depolarizing", "005"), ("Depolarizing", "01"), ("FT", "005"), ("FT", "01")]
COL_LABELS = ["DP p=0.005", "DP p=0.010", "FT p=0.005", "FT p=0.010"]


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

def synds_features(S_set, dem):
    """
            Parameters
                ---------
                synd_set: a flat detector vectors set
                dem: the circuit detector error model
        
                Returns
                -------
                features: a list of the geometric descriptors vector (weight, spatial and temporal spread, isolated and boundary)
    """
    coords = dem.get_detector_coordinates()
    N_det, N = len(coords), len(S_set)
    xy  = np.array([[coords[i][0], coords[i][1]] for i in range(N_det)],
                    dtype=np.float32)          # (N_det, 2) spatial position
    t   = np.array([coords[i][2] for i in range(N_det)],
                    dtype=np.float32)
    features = np.zeros((N, 5)).astype(np.float32)

    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
    on_boundary  = (
        (xy[:, 0] == x_min) | (xy[:, 0] == x_max) |
        (xy[:, 1] == y_min) | (xy[:, 1] == y_max)
    )

    grid_step = 2.0
    neighbour_map = {i: set() for i in range(N_det)}
    for i in range(N_det):
        for j in range(i+1, N_det):
            if t[i] == t[j]:
                dist = np.linalg.norm(xy[i] - xy[j])
                if abs(dist - grid_step) < 0.5:   # tolerance for float coords
                    neighbour_map[i].add(j)
                    neighbour_map[j].add(i)

    for i, s in enumerate(S_set):
        fired = np.flatnonzero(s)
        if len(fired) == 0:
            continue

        w = len(fired)
        features[i, 0] = w # weight

        pos   = xy[fired] 
        dists = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)

        features[i, 1] = dists.max() # spread space

        t_fired = t[fired]
        features[i, 2] = t_fired.max() - t_fired.min()  # spread time

        iso = sum(
            1 for f in fired
            if not any(s[nb] for nb in neighbour_map[f])
        )

        features[i, 3] = iso # isolated

        features[i, 4] = on_boundary[fired].mean() # boundary

    return features


def fit_logistic_regression(group_pos, group_neg, dem, sample_cap=20_000):
    """
    Fit logistic regression: blind spot (1) vs. correctly decoded (0).

    Weight-matched sampling: for each weight w present in P_blind,
    sample the same number of syndromes from P_both at weight w.
    
    Returns clf, scaler, auc, feature_names
    """
    

    phi_first = synds_features(group_pos, dem)
    phi_second  = synds_features(group_neg,  dem)

    n = min(len(phi_first), len(phi_second), sample_cap)
    X_blind = phi_first[np.random.choice(len(phi_first), n, replace=False)]
    X_both  = phi_second[ np.random.choice(len(phi_second),  n, replace=False)]

    #X_blind = np.vstack(phi_first)
    #X_both  = np.vstack(phi_second[:sample_cap])
    X = np.vstack([X_blind, X_both])
    y = np.array([1] * len(X_blind) + [0] * len(X_both))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=500, class_weight='balanced')
    clf.fit(X_scaled, y)
   
    auc = roc_auc_score(y, clf.predict_proba(X_scaled)[:, 1])

    return clf, scaler, auc


def plot_auc_heatmap(auc_matrix: np.ndarray, title: str, output_path: str) -> None:
    """
    @Input :
        auc_matrix  = (3, 4) array, rows = ROW_KEYS (code distance),
                      cols = COL_KEYS (noise model / physical error rate). NaN for configs with no data.
        output_path = full path (including filename) to save the PNG to
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(auc_matrix, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    fig.colorbar(im, ax=ax, label="AUC")

    ax.set_xticks(range(len(COL_LABELS)))
    ax.set_xticklabels(COL_LABELS)
    ax.set_yticks(range(len(ROW_LABELS)))
    ax.set_yticklabels(ROW_LABELS)
    ax.set_xlabel("Noise model, physical error rate")
    ax.set_ylabel("Code distance")
    ax.set_title(title)

    for i in range(auc_matrix.shape[0]):
        for j in range(auc_matrix.shape[1]):
            val = auc_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=11, color="black")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_coefficients(coef_lookup: Dict, title: str, output_path: str) -> None:
    """
    @Input :
        coef_lookup = dict {(noise_folder, distance, param_folder): coef array of length 5},
                      one entry per config that was successfully fit. Laid out as
                      rows = ROW_KEYS (code distance), cols = COL_KEYS (noise model /
                      physical error rate), matching plot_auc_heatmap. A missing config
                      is simply left out and its subplot is hidden.
        title       = suptitle (which comparison this is)
        output_path = full path (including filename) to save the PNG to
    """
    fig, axes = plt.subplots(len(ROW_KEYS), len(COL_KEYS), figsize=(16, 9), sharey=False)

    for i, distance in enumerate(ROW_KEYS):
        for j, (noise_folder, param_folder) in enumerate(COL_KEYS):
            ax = axes[i, j]
            coefs = coef_lookup.get((noise_folder, distance, param_folder))

            if coefs is None:
                ax.set_visible(False)
                continue

            bar_colors = ["#1D9E75" if wi >= 0 else "#D85A30" for wi in coefs]
            ax.barh(FEATURE_NAMES, coefs, color=bar_colors, edgecolor="black", linewidth=0.5)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_title(f"{ROW_LABELS[i]}, {COL_LABELS[j]}", fontsize=9)
            ax.tick_params(labelsize=8)

            if j == 0:
                ax.set_ylabel("Feature")
            if i == len(ROW_KEYS) - 1:
                ax.set_xlabel("Standardized coefficient")

    fig.suptitle(title, fontsize=12)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Characterize SA adversarial syndromes vs natural MLP blind spots.")
    parser.add_argument("--attacks_dir", type=str, required=True, help="directory containing <NoiseModel>_d<D>_<param>.jsonl files")
    parser.add_argument("--h5_input", type=str, required=True, help="path to the disagreement map .h5 (same as syndrome_features.py --input)")
    parser.add_argument("--rounds", type=int, default=9, help="number of rounds in the circuit (default: 9)")
    parser.add_argument("--output_folder", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_configs = list(itertools.product(NOISE_MODELS, DISTANCES, PARAMS))

    final_output = args.output_folder
    if final_output[-1] == "/":
        final_output = final_output[:-1]

    # arrays / lookups that collect what each plot needs, filled in as we go
    auc_matrix_DM_vs_both = np.full((len(ROW_KEYS), len(COL_KEYS)), np.nan)
    auc_matrix_DM_vs_sa = np.full((len(ROW_KEYS), len(COL_KEYS)), np.nan)
    auc_matrix_both_vs_sa = np.full((len(ROW_KEYS), len(COL_KEYS)), np.nan)
    coef_lookup_DM_vs_both = {}
    coef_lookup_DM_vs_sa = {}
    coef_lookup_both_vs_sa = {}

    for noise_folder, distance, param_folder in tqdm(all_configs):
        group_path = f"{noise_folder}/d{distance}/{param_folder}"
        row = ROW_KEYS.index(distance)
        col = COL_KEYS.index((noise_folder, param_folder))
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
        DM = read_h5_table(args.h5_input, group_path)

        # create the circuit and dem for this config
        noise_type = NOISE_FOLDER_TO_TYPE[noise_folder]
        noise_default = PARAM_FOLDER_TO_VALUE[param_folder]

        circuit = create_circuit(distance=distance, rounds=args.rounds, noise_type=noise_type, noise_default=noise_default)
        dem = circuit.detector_error_model(decompose_errors=True)
        ## Pblind vs Pboth
        # clf_blind, scaler_blind, auc_blind = fit_logistic_regression(DM["mlp_blind_spot"], DM["both_correct"], dem)
        # auc_matrix_DM_vs_both[row, col] = auc_blind
        # coef_lookup_DM_vs_both[(noise_folder, distance, param_folder)] = clf_blind.coef_[0]#first line cuz only 1 usable label class

        # ### Pblind vs C
        # clf_sa, scaler_sa, auc_sa = fit_logistic_regression(DM["mlp_blind_spot"], syndromes_adversarial, dem)
        # auc_matrix_DM_vs_sa[row, col] = auc_sa
        # coef_lookup_DM_vs_sa[(noise_folder, distance, param_folder)] = clf_sa.coef_[0]

        ### Pboth vs C
        clf_both_vs_sa, scaler_both_vs_sa, auc_both_vs_sa = fit_logistic_regression(DM["both_correct"], syndromes_adversarial, dem)
        auc_matrix_both_vs_sa[row, col] = auc_both_vs_sa
        coef_lookup_both_vs_sa[(noise_folder, distance, param_folder)] = clf_both_vs_sa.coef_[0]

    plot_auc_heatmap(
        auc_matrix_DM_vs_both,
        title=r"Logistic Regression AUC" "\n" r"($\mathcal{P}_{blind}$ vs $\mathcal{P}_{both}$)",
        output_path=f"{final_output}/Regression/AUC_blind_vs_both.png",
    )
    plot_auc_heatmap(
        auc_matrix_DM_vs_sa,
        title=r"Logistic Regression AUC" "\n" r"($\mathcal{P}_{blind}$ vs $\mathcal{C}$)",
        output_path=f"{final_output}/Regression/AUC_blind_vs_SA.png",
    )
    plot_feature_coefficients(
        coef_lookup_DM_vs_both,
        title=r"Logistic Regression Feature Coefficients ($\mathcal{P}_{blind}$ vs $\mathcal{P}_{both}$)"
              "\nGreen = higher in blind spots, Orange = higher in both correct",
        output_path=f"{final_output}/Regression/Coefficients_blind_vs_both.png",
    )
    plot_feature_coefficients(
        coef_lookup_DM_vs_sa,
        title=r"Logistic Regression Feature Coefficients ($\mathcal{P}_{blind}$ vs $\mathcal{C}$)"
              "\nGreen = higher in blind spots, Orange = higher in adversarial (SA)",
        output_path=f"{final_output}/Regression/Coefficients_blind_vs_SA.png",
    )
    plot_auc_heatmap(
        auc_matrix_both_vs_sa,
        title=r"Logistic Regression AUC" "\n" r"($\mathcal{P}_{both}$ vs $\mathcal{C}$)",
        output_path=f"{final_output}/Regression/AUC_both_vs_SA.png",
    )
    plot_feature_coefficients(
        coef_lookup_both_vs_sa,
        title=r"Logistic Regression Feature Coefficients ($\mathcal{P}_{both}$ vs $\mathcal{C}$)"
              "\nGreen = higher in both correct, Orange = higher in adversarial (SA)",
        output_path=f"{final_output}/Regression/Coefficients_both_vs_SA.png",
    )

