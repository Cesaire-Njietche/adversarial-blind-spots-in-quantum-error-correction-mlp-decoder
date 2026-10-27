# Adversarial Blind Spots in Neural QEC Decoders

Code repository for the research project:

> **"Failure Modes: Adversarial Blind Spots in Neural Quantum Error Correction Decoders"**

This work characterises where and why MLP-based surface code decoders fail under adversarially chosen error patterns, and builds towards a certifiably robust decoder via adversarial training.

---

## Overview

Neural decoders are fast and adaptive, but opaque. This project asks: are there structured, physically realisable error patterns that reliably fool a trained MLP decoder, even at low physical error rates? We answer this with two complementary tools:

1. **Simulated annealing attack** - searches over error pattern space for worst-case syndromes within a physical error budget
2. **Decision boundary analysis** - maps syndromes where the MLP fails but the optimal decoder (MWPM) succeeds

Experiments run on the **rotated surface code** at distances d=3, d=5 and d=7 under both depolarizing and circuit-level fault-tolerant noise models, using [Stim](https://github.com/quantumlib/Stim) for simulation and [PyMatching](https://github.com/oscarhiggott/PyMatching) as the MWPM reference decoder.

---

## Repository Structure

```
.
├── data/
│   ├── generate_datasets.py        # Stim-based dataset generation, saves a .npz + a matching .json of circuit params
│   └── noise_models.py             # Simple depolarizing + circuit-level FT configs
│
├── decoder/
│   ├── model.py                    # MLP architecture (build_model / build_model_from_checkpoint_as_eval)
│   ├── train.py                    # Trains one config on one dataset, saves config/checkpoint/metrics
│   ├── tune.py                     # Random hyperparameter search, calls train.train() per trial
│   ├── mwpm_reference.py           # PyMatching MWPM baseline, rebuilds the circuit from a dataset's .json
│   └── config_example.json         # Example --config file for train.py
│
├── attacks/
│   ├── simulated_annealing.py      # SA attack over error pattern space, sweeping every weight budget W, for ONE model
│   ├── run_all_attacks.py          # Runs simulated_annealing.py against every checkpoint under final_models/
│   └── results/                    # One <NoiseModel>_d<D>_<param>.jsonl per attacked model (SA catalogs)
│
├── analysis/
│   ├── disagreement_map.py         # Builds the MLP vs MWPM four-way disagreement map (both_correct/both_wrong/mlp_blind_spot/mwpm_blind_spot) as an HDF5 file
│   ├── syndrome_feature.py         # Geometric feature extraction (weight, spatial/temporal spread, isolation, boundary proximity) + logistic-regression characterization of blind spots vs SA catalogs
│   ├── disagreement.h5             # Disagreement map produced by disagreement_map.py
│   └── results/
│       └── Regression/             # AUC heatmaps + feature-coefficient plots produced by syndrome_feature.py
│
├── final_models/
│   └── <NoiseModel>/d<D>/<param>/  # config.json + model.pth for each trained decoder (Depolarizing|FT, d3|d5|d7, 005|01)
│
├── pyproject.toml / uv.lock / requirements.txt
└── README.md
```

---

## Getting Started

### Requirements

Install requirements from `requirements.txt` for example using pip :

```bash
pip install -r requirements.txt
```
Python 3.10+ recommended.

### 1 — Generate training data

```bash
python data/generate_datasets.py --distance 3 --rounds 9 --samples 1000000 --noise depolarizing --noise-default 0.005 --output data/train_depolarizing.npz --verbose
```

Only `--output` is mandatory.

Each run writes two files:
- the `.npz` itself, with `"labels"` (shape `(samples, 1)`) and `"features"` (shape `(samples, (d²-1)*rounds)`, detector/syndrome bits)
- a `.json` (same path, `.json` extension) recording `distance`, `rounds`, `noise_type`, `noise_default` — the parameters needed to rebuild the exact same circuit later (used by `mwpm_reference.py` below)

### 2 — Train the MLP decoder

```bash
python decoder/train.py --dataset data/train_depolarizing.npz --config decoder/config_example.json --output-dir runs/baseline --epochs 50 --seed 42 --verbose
```

From the flags shown in the command only `--dataset` and `--output-dir` are required, the rest (`--config`, `--epochs`, `--seed`, `--verbose`) are optional — see `--help`. Note that `--config` defaults to `model.DEFAULT_CONFIG` when omitted. 

For better understanding, here is a `--config` file, matching `model.DEFAULT_CONFIG`:

```json
{
  "hidden_sizes": [256, 128, 64],
  "dropout": [0.1, 0.1, 0.1],
  "lr": 0.003
}
```

`hidden_sizes` and `dropout` must have the same length — one dropout value per hidden layer.

Trains the MLP from `decoder/model.py` on the GPU when one is available (falls back to CPU otherwise), using a stratified train/val split on the logical-flip label. Writes three files into `--output-dir`:
- `model.pth` — the checkpoint with the best validation accuracy seen so far
- `config.json` — the hyperparameters actually used, plus provenance of dataset (`dataset_path`, `seed`, `epochs`) — you can pass the path of this file back in `--config` to retrain the *same* configuration on new data
- `metrics.json` — per-epoch train/val loss curves and the best validation accuracy/epoch. Can be used for analytics

### 2a — Hyperparameter search

```bash
python decoder/tune.py --dataset data/train_depolarizing.npz --output-dir runs/tuning --trials 20 --epochs 30 --seed 42 --verbose
```

Random search over the `tune.SEARCH_SPACE`. Every trial calls `train.train()` directly — `tune.py` and for maximum `trails` configs in `tune.SEARCH_SPACE` runs `epochs` training:
- `<output-dir>/trial_XXX/` — `config.json`/`model.pth`/`metrics.json` for each trial, written by `train()` itself
- `<output-dir>/leaderboard.json` — every trial's config + `best_val_accuracy`, ranked best first

To retrain the winning config (e.g. on more data, or more epochs): `python decoder/train.py --dataset <new_data>.npz --config runs/tuning/trial_003/config.json --output-dir runs/retrain --epoch 100 --verbose`.

### 2b — Run the MWPM reference decoder

```bash
python decoder/mwpm_reference.py --metadata data/train_depolarizing.json --dataset data/train_depolarizing.npz --samples 100000 --prediction-path data/mwpm_predictions.npz --verbose
```

`--dataset`/`--samples` are mutually exclusive in practice. Rebuilds the circuit from `--metadata`, decodes the syndromes in `--dataset` with PyMatching, and reports the logical error rate; `--prediction-path` stores predictions, ground-truth labels, and the logical error rate as a `.npz`. If `--dataset` is omitted, `--samples` fresh shots are sampled directly from the rebuilt circuit instead. This is NOT to do when the goal is to compare to an MPL model as for fairness they should be trained on the same dataset.

### 2c — Full worked example (data → MWPM baseline → tune → retrain)

```bash
# 1. Generate data
uv run data/generate_datasets.py --distance 3 --rounds 9 --samples 1000000 --noise depolarizing --output example_run/dataset.npz

# 2b. MWPM baseline on that data
uv run decoder/mwpm_reference.py --dataset example_run/dataset.npz --metadata example_run/dataset.json --prediction-path example_run/pred_mwpm.npz --verbose
#   MWPM baseline results: Shots=1000000, Logical error rate=0.001720

# 2a. Short hyperparameter search (3 epochs/trial, just to pick a config)
uv run decoder/tune.py --dataset example_run/dataset.npz --trials 2 --epochs 3 --output-dir example_run/ --verbose
#   Best trial: example_run/trial_000 (val_accuracy=0.9981)

# 2. Retrain the winning config for real (30 epochs)
uv run decoder/train.py --dataset example_run/dataset.npz --config example_run/trial_000/config.json --output-dir example_run/ --epochs 30 --verbose
#   Best val accuracy: 0.9981 (epoch 27) -> example_run/model.pth
```


### 3 — Run the SA attack against one trained decoder

```bash
python attacks/simulated_annealing.py --config final_models/Depolarizing/d3/005/config.json --checkpoint final_models/Depolarizing/d3/005/model.pth --distance 3 --noise-type depolarizing --noise-default 0.005 --n-restarts 50 --output attacks/results/Depolarizing_d3_005.jsonl
```

Sweeps every weight budget `W` from 1 to `distance//2 + 1`, running `--n-restarts` independent simulated-annealing chains per `W` to find the syndrome that maximises the decoder's BCE loss. `--output` is a `.jsonl` file: a `header` record (model/noise config) followed by one `result` record per `W` (attack success rate, the catalog of syndromes that fooled the decoder, and the single worst syndrome found).

### 3a — Run the SA attack against every final model

```bash
python attacks/run_all_attacks.py --final-models-dir final_models --output-dir attacks/results --rounds 9
```

Loops over every `<NoiseModel>/d<D>/<param>` combination under `--final-models-dir` (skipping any with no checkpoint) and calls the same attack as step 3 for each, writing one `<NoiseModel>_d<D>_<param>.jsonl` per model into `--output-dir`.

### 4 — Build the MWPM vs MLP disagreement map

```bash
python analysis/disagreement_map.py --final-models-dir final_models --samples 100000 --rounds 9 --output analysis/disagreement.h5
```

For every `<NoiseModel>/d<D>/<param>` model, samples `--samples` syndromes from the noise model, decodes them with both MWPM (PyMatching) and the MLP checkpoint, and sorts every syndrome into one of four categories: `both_correct`, `both_wrong`, `mlp_blind_spot` (MWPM right, MLP wrong) and `mwpm_blind_spot` (MLP right, MWPM wrong). Writes one HDF5 group per model (`<NoiseModel>/d<D>/<param>/<category>`), each a plain array of syndromes.

### 5 — Characterize blind spots vs the SA catalogs

```bash
python analysis/syndrome_feature.py --attacks_dir attacks/results --h5_input analysis/disagreement.h5 --rounds 9 --output_folder analysis/results
```

For every model, extracts 5 geometric features per syndrome (Hamming weight, spatial spread, temporal spread, isolation, boundary proximity) and fits a weight-matched logistic regression to see whether these features separate:
- `mlp_blind_spot` vs `both_correct` (are natural MLP failures geometrically distinct from syndromes it gets right?)
- `mlp_blind_spot` vs the SA catalog (do the adversarially-found syndromes look like natural blind spots?)
- `both_correct` vs the SA catalog

Writes AUC heatmaps and standardized-coefficient bar charts for each comparison to `<output_folder>/Regression/`.


---

## Notes

> **`config.json`'s `dataset_path` is just a recorded file path, not a hash of the file's contents.** If the `.npz` at that path is later regenerated or overwritten (e.g. re-running `generate_datasets.py` with different `--rounds`/`--noise-default`), `config.json` still points at the path but the data behind it has changed.

> **Always pass `--dataset` (not `--samples`) to `mwpm_reference.py` when comparing MWPM against the MLP.** `--samples` draws an independent random syndrome set, which is not fine for a fair, same-syndrome comparison against another decoder. This is exactly why `config.json` keeps a `dataset_path` reference — it's what tells you which `.npz` to point `--dataset` at.
---

## Key Concepts

| Term | Meaning |
|---|---|
| **Surface code** | A 2D grid of qubits that encodes one logical qubit redundantly across d² data qubits |
| **Syndrome** | Binary vector of ancilla measurement outcomes — what the decoder actually sees |
| **Error pattern E** | Assignment of Pauli faults to every circuit location across all syndrome rounds |
| **Adversary budget W** | Maximum Hamming weight of E — how many faults the adversary can inject |
| **Blind spot** | A syndrome the MLP decoder miscorrects but MWPM handles correctly |
| **Robustness radius** | Minimum bit-flips needed to move a correctly decoded syndrome to an MLP failure |

---

## Noise Models

Defined in `data/noise_models.py` via `generate_noise_constants(noise_type, default=0.005, ...)`, which returns a kwargs dict fed straight into `stim.Circuit.generated(...)`. Any parameter left unspecified falls back to `default` — controllable from the CLI via `--noise-default` (see step 1 below).

**Depolarizing (simple, `--noise depolarizing`):** only `before_round_data_depolarization` is set — uniform Pauli noise on data qubits between rounds, otherwise noiseless. Good for baselines and threshold analysis.

**Circuit-level FT (`--noise circuit-level`):** `after_clifford_depolarization`, `after_reset_flip_probability`, `before_measure_flip_probability`, and `before_round_data_depolarization` are all set — per-gate depolarizing noise, noisy measurements, and noisy ancilla resets. Matches real hardware noise structure more closely.

---

## Results

12 configurations were trained and evaluated: distances `d ∈ {3, 5, 7}` × noise models {Depolarizing (DP), circuit-level fault-tolerant (FT)} × physical error rates `p ∈ {0.005, 0.01}`. All models trained for `T=9` rounds on `10^6` Stim-sampled syndromes (800k/200k train/val split).

### MLP vs. MWPM validation accuracy

| Noise | d | p | MLP val. acc. | MWPM val. acc. |
|---|---|---|---|---|
| DP | 3 | 0.005 | 99.85% | 99.83% |
| DP | 3 | 0.01  | 99.38% | 99.32% |
| DP | 5 | 0.005 | 99.95% | 99.99% |
| DP | 5 | 0.01  | 99.82% | 99.92% |
| DP | 7 | 0.005 | 99.88% | 100%   |
| DP | 7 | 0.01  | 99.66% | 99.99% |
| FT | 3 | 0.005 | 95.34% | 95.35% |
| FT | 3 | 0.01  | 84.86% | 84.88% |
| FT | 5 | 0.005 | 89.58% | 97.35% |
| FT | 5 | 0.01  | 66.37% | 85.55% |
| FT | 7 | 0.005 | 78.08% | 98.71% |
| FT | 7 | 0.01  | 52.73% | 86.76% |

Under depolarizing noise the MLP tracks MWPM closely at every distance. Under circuit-level FT noise the gap widens sharply with `d` and `p`, with the MLP falling as much as 34 points behind MWPM at `d=7, p=0.01`.

### Simulated-annealing attack: success rate Φ(W) by weight budget

Minimum budget `W` at which the attack reaches `Φ(W)=1.0` scales with distance under depolarizing noise (`W=2` for d=3, `W=3` for d=5, `W=4` for d=7, consistent with `⌊d/2⌋`), and is systematically lower under FT noise — several FT configurations already have `Φ(W)>0` at `W=1`, i.e. a single physical fault is enough to fool the decoder on some runs.

| Noise | d | p | W=1 | W=2 | W=3 | W=4 |
|---|---|---|---|---|---|---|
| DP | 3 | 0.005 | 0.0 | 1.0 | — | — |
| DP | 3 | 0.01  | 0.0 | 1.0 | — | — |
| DP | 5 | 0.005 | 0.0 | 0.2 | 1.0 | — |
| DP | 5 | 0.01  | 0.0 | 0.14| 1.0 | — |
| DP | 7 | 0.005 | 0.0 | 0.0 | 0.98| 1.0 |
| DP | 7 | 0.01  | 0.0 | 0.0 | 0.98| 1.0 |
| FT | 3 | 0.005 | 0.0 | 1.0 | — | — |
| FT | 3 | 0.01  | 0.0 | 1.0 | — | — |
| FT | 5 | 0.005 | 0.0 | 1.0 | 1.0 | — |
| FT | 5 | 0.01  | 1.0 | 1.0 | 1.0 | — |
| FT | 7 | 0.005 | 1.0 | 1.0 | 1.0 | 1.0 |
| FT | 7 | 0.01  | 1.0 | 1.0 | 1.0 | 1.0 |

### Disagreement map: blind-spot fraction

`𝒫_blind` (MWPM correct, MLP wrong) as a fraction of 500,000 test syndromes per configuration — ranges from 0.036% (DP, d=3, p=0.005) up to 40.8% (FT, d=7, p=0.01). It grows with both `d` and `p`, and is consistently larger under FT noise than DP at matched `(d, p)`.

| Noise | d | p | 𝒫_both | 𝒫_blind | 𝒫_hard | 𝒫_MLP⁺ |
|---|---|---|---|---|---|---|
| DP | 3 | 0.005 | 0.997 | 3.58e-4 | 1.34e-3 | 4.78e-4 |
| DP | 3 | 0.01  | 0.991 | 1.62e-3 | 4.88e-3 | 2.28e-3 |
| DP | 5 | 0.005 | 0.999 | 4.88e-4 | 7e-5    | 1.6e-5  |
| DP | 5 | 0.01  | 0.997 | 1.23e-3 | 6.7e-4  | 1.62e-4 |
| DP | 7 | 0.005 | 0.998 | 1.19e-3 | 2e-6    | 6e-6    |
| DP | 7 | 0.01  | 0.988 | 1.13e-2 | 3.4e-5  | 3e-5    |
| FT | 3 | 0.005 | 0.936 | 1.69e-2 | 3e-2    | 1.6e-2  |
| FT | 3 | 0.01  | 0.801 | 4.82e-2 | 0.102   | 4.8e-2  |
| FT | 5 | 0.005 | 0.884 | 8.98e-2 | 1.44e-2 | 1.13e-2 |
| FT | 5 | 0.01  | 0.594 | 0.26    | 7.59e-2 | 6.93e-2 |
| FT | 7 | 0.005 | 0.772 | 0.213   | 6.86e-3 | 6.95e-3 |
| FT | 7 | 0.01  | 0.458 | 0.408   | 6.57e-2 | 6.73e-2 |

### Geometric characterization of blind spots

A logistic regression on 5 geometric features (weight, spatial spread, temporal spread, isolation, boundary proximity) separates `𝒫_blind` from `𝒫_both` with AUC ranging from **0.974** (DP, d=3, p=0.005) down to **0.524** (FT, d=7, p=0.01) — blind spots are well-structured at small `d` `p` and degrade towards random as the code and noise rate grow, with weight as the dominant predictor. Separating the SA attack catalog from `𝒫_both` gives consistently high AUC (0.72–1.0), but the SA catalog and `𝒫_blind` themselves barely overlap : SA finds the sparse, minimum-weight failure boundary, while the disagreement map captures the physically realistic failure distribution.
---

## Papers

**Paper 1** (target: PRX Quantum / QIP 2027)
> *Failure Modes: Adversarial Blind Spots in Neural Quantum Error Correction Decoders*
> Attack characterisation + decision boundary analysis

**Paper 2** (target: NeurIPS QML workshop / Nature Communications)
> Certified robustness bounds + adversarial training framework

---

## Citation

```bibtex
@article{adversarial_qec_2026,
  title   = {Failure Modes: Adversarial Blind Spots in Neural Quantum Error Correction Decoders},
  author  = {[Césaire Fangang Njietche, Martine Bellaïche, Victor Fairon]},
  journal = {PRX Quantum},
  year    = {2027}
}
```

---

## References

- Gidney, C. (2021). Stim: a fast stabilizer circuit simulator. *Quantum*, 5, 497.
- Higgott, O. et al. (2023). Sparse blossom: correcting a million errors per core per second with minimum-weight matching. *PRX Quantum*.
- Lenssen, J. & Paler, A. (2025). Fooling the decoder: an adversarial attack on quantum error correction. *arXiv:2504.19651*.
- Madry, A. et al. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR*.
- Fowler, A. et al. (2012). Surface codes: towards practical large-scale quantum computation. *PRA*, 86, 032324.
