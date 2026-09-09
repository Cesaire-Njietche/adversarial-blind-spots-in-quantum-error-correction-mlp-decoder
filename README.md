# Adversarial Blind Spots in Neural QEC Decoders

Code repository for the research project:

> **"Failure Modes: Adversarial Blind Spots in Neural Quantum Error Correction Decoders"**

This work characterises where and why MLP-based surface code decoders fail under adversarially chosen error patterns, and builds towards a certifiably robust decoder via adversarial training.

---

## Overview

Neural decoders are fast and adaptive — but opaque. This project asks: are there structured, physically realisable error patterns that reliably fool a trained MLP decoder, even at low physical error rates? We answer this with two complementary tools:

1. **Simulated annealing attack** — searches over error pattern space for worst-case syndromes within a physical error budget
2. **Decision boundary analysis** — maps syndromes where the MLP fails but the optimal decoder (MWPM) succeeds

Experiments run on the **rotated surface code** at distances d=3 and d=5, under both depolarizing and circuit-level fault-tolerant noise models, using [Stim](https://github.com/quantumlib/Stim) for simulation and [PyMatching](https://github.com/oscarhiggott/PyMatching) as the MWPM reference decoder.

---

## Repository Structure

```
.
├── data/
│   ├── generate_datasets.py        # Stim-based dataset generation, saves a .npz + a matching .json of circuit params
│   └── noise_models.py             # Simple depolarizing + circuit-level FT configs
│
├── decoder/
│   ├── model.py                    # MLP architecture (build_model from a config dict)
│   ├── train.py                    # Trains one config on one dataset, saves config/checkpoint/metrics
│   ├── tune.py                     # Random hyperparameter search, calls train.train() per trial
│   └── mwpm_reference.py           # PyMatching MWPM baseline, rebuilds the circuit from a dataset's .json
│
├── attacks/
│   ├── adversary_model.py          # Formal adversary model and budget constraint
│   └── simulated_annealing.py      # SA attack over error pattern space
│
├── analysis/
│   ├── disagreement_map.py         # MLP vs MWPM four-way partition
│   ├── syndrome_features.py        # Geometric feature extraction for blind spots
│   └── boundary_distance.py        # Minimum-flip distance to MLP failure
│
├── experiments/
│   ├── run_attack.py               # End-to-end SA attack pipeline
│   ├── run_boundary_analysis.py    # End-to-end decision boundary pipeline
│   └── run_evaluation.py           # Logical error rate curves across distances
│
├── notebook/
│   ├── model_analysis.ipynb        # Loads a trained checkpoint (config.json + model.pth), plots loss curves & performance
│   └── results_visualization.ipynb # Figures for the paper (not yet implemented)
│
├── requirements.txt
├── stim.ipynb
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
python data/generate_datasets.py --distance 3 --rounds 6 --samples 1000000 --noise depolarizing --output data/train_depolarizing.npz
python data/generate_datasets.py --distance 3 --rounds 6 --samples 1000000 --noise circuit-level --output data/train_circuit_level.npz --verbose
```

Arguments: `--distance` (default 3), `--rounds` (default 9), `--samples` (default 100,000), `--noise` (`depolarizing` or `circuit-level`) (default `depolarizing`), `--noise-default` (physical error rate applied to any noise parameter not explicitly overridden, default 0.005 — see `noise_models.py`), `--output` (required), `--verbose`.

Each run writes two files :
- the `.npz` itself, with `"labels"` (shape `(samples, 1)`, logical observable flips) and `"features"` (shape `(samples, (d²-1)*rounds)`, detector/syndrome bits)
- a `.json` (same path, but with `.json` extension) recording `distance`, `rounds`, `noise_type`, `noise_default` — the parameters needed to rebuild the exact same circuit later (used by `mwpm_reference.py` below)

With `--verbose`, a short analytics summary is printed before saving.

### 2 — Train the MLP decoder

```bash
python decoder/train.py --dataset data/train_depolarizing.npz --output-dir runs/baseline --epochs 50 --verbose
```

Arguments: `--dataset` (required, a `.npz` from data generation), `--config` (path to a hyperparameter `.json`; defaults to `model.DEFAULT_CONFIG`. json specifies `hidden_sizes`, `dropout` (one value per hidden layer), `lr`), `--output-dir` (required), `--epochs` (default 50), `--seed` (for replicability — if omitted, reuses the seed recorded in `--config` when it has one, e.g. retraining a `tune.py` trial reproduces its exact seed; otherwise defaults to 42), `--verbose`.

Example `--config` file, matching `model.DEFAULT_CONFIG` :

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
- `config.json` — the hyperparameters actually used, plus provenance (`dataset_path`, `seed`, `epochs`) — pass this straight back in as `--config` to retrain the *same* configuration on new data
- `metrics.json` — per-epoch train/val loss curves and the best validation accuracy/epoch

> [!NOTE]
> `dataset_path` is just a recorded file path, not a hash of the file's contents. If the `.npz` at that path is later regenerated or overwritten (e.g. re-running `generate_datasets.py` with different `--rounds`/`--noise-default`), `config.json` still points at the path but the data behind it has changed.

### 2a — Hyperparameter search

```bash
python decoder/tune.py --dataset data/train_depolarizing.npz --output-dir runs/tuning --trials 20 --epochs 30 --verbose
```

Arguments: `--dataset` (required), `--output-dir` (required), `--trials` (default 20), `--epochs` per trial (default 30 — usually shorter than a final training run), `--seed` (default 42, controls both which configs get tried and each trial's training seed), `--verbose`.

Random search over `hidden_sizes`/per-layer `dropout`/`lr` (see `tune.SEARCH_SPACE`). Every trial calls `train.train()` directly — `tune.py` never re-implements the training loop, so there is exactly one training implementation to trust. Writes:
- `<output-dir>/trial_XXX/` — `config.json`/`model.pth`/`metrics.json` for each trial, written by `train()` itself
- `<output-dir>/leaderboard.json` — every trial's config + `best_val_accuracy`, ranked best first

To retrain the winning config (e.g. on more data, or more epochs): `python decoder/train.py --dataset <new_data>.npz --config runs/tuning/trial_003/config.json --output-dir runs/retrain --verbose`.

### 2b — Run the MWPM reference decoder

```bash
python decoder/mwpm_reference.py --metadata data/train_depolarizing.json --dataset data/train_depolarizing.npz --prediction-path data/mwpm_predictions.npz --verbose
```

Rebuilds the circuit from the dataset's `.json` file, decodes the syndromes in `--dataset` with PyMatching, and reports the logical error rate. `--prediction-path` is required and stores predictions, ground-truth labels, and the logical error rate as a `.npz`.

If `--dataset` is omitted, `--samples` (default 100,000) fresh shots are sampled directly from the rebuilt circuit instead — useful for a quick standalone MWPM sanity check, but **not** for a fair comparison against another decoder, since that draws an independent random syndrome set.
> [!WARNING]
> **Attention**
> Always pass `--dataset` pointing at a shared `.npz` when comparing MWPM against the MLP.

### 2c — Analyze a trained model

Loads a run's `config.json`/`model.pth` (from step 2), rebuilds the same model + validation split, and plots the training/validation loss curves and validation performance (confusion matrix, prediction-probability distribution).

> [!NOTE]
> **Seeds used across the pipeline** — several distinct seeds show up in steps 1–2c, each controlling a different thing:
> - **Dataset generation (step 1)** has no `--seed` at all — `generate_datasets.py` samples the circuit unseeded, so re-running it produces a *different* dataset every time, even with identical arguments.
> - **`train.py --seed`** seeds `torch`/`numpy` for weight initialization *and* the stratified train/val split (via `sklearn`'s `random_state`). If omitted it reuses the seed recorded in `--config` when there is one (so retraining a saved `config.json` reproduces that exact run), otherwise it defaults to 42 — this is what makes a training run reproducible given the same dataset and config.
> - **`tune.py --seed`** (default 42) seeds one master RNG that both picks which configs the search tries *and* derives each trial's own training seed — reproduces the whole search, not just one trial.
> - **The notebook's `EVAL_SEEDS`** (`1000`–`1004`) seed the independently-generated evaluation sets in `model_analysis.ipynb`, deliberately distinct from the training seed so they can't accidentally overlap with data the model trained on.
>
> None of these seed the dataset itself, so "reproducible training" only holds as long as you keep the original `.npz` around (see the note above) rather than regenerating it.

### 3 — Run the SA attack

```bash
python experiments/run_attack.py --distance 3 --budget 3 --restarts 50
```

### 4 — Run the decision boundary analysis

```bash
python experiments/run_boundary_analysis.py --distance 3 --shots 2000000
```

### Example of a full run for parameters optimisation : 
```bash 

adversarial-blind-spots-in-quantum-error-correction-mlp-decoder on  Victor [✘!+?] is 📦 v0.1.0 via 🐍 v3.13.5 
❯ uv run data/generate_datasets.py --distance 3 --rounds 9 --samples 1000000 --noise depolarizing --output example_run/dataset.npz
Generating noise constants for noise model: depolarizing with default value: 0.005
########################################################

adversarial-blind-spots-in-quantum-error-correction-mlp-decoder on  Victor [✘!+?] is 📦 v0.1.0 via 🐍 v3.13.5 
❯ uv run decoder/mwpm_reference.py --dataset example_run/dataset.npz --metadata example_run/dataset.json --prediction-path example_run/pred_mwpm.npz --verbose
########################################################
Loaded circuit metadata from example_run/dataset.json: {'distance': 3, 'rounds': 9, 'noise_type': 'depolarizing', 'noise_default': 0.005}
########################################################
Generating noise constants for noise model: depolarizing with default value: 0.005
########################################################
Generated circuit with distance=3, rounds=9, noise=depolarizing and default noise value=0.005
########################################################
Loading syndromes to decode from example_run/dataset.npz
########################################################
MWPM baseline results
  Shots              : 1000000
  Logical error rate : 0.001720
########################################################
Saved predictions to example_run/pred_mwpm.npz


adversarial-blind-spots-in-quantum-error-correction-mlp-decoder on  Victor [✘!+?] is 📦 v0.1.0 via 🐍 v3.13.5 
❯ uv run decoder/tune.py --dataset example_run/dataset.npz --trials 2 --epochs 3 --output-dir example_run/ --verbose
[1/2] config={'hidden_sizes': (128, 64), 'dropout': (0.2, 0.1), 'lr': 0.003} seed=929893137
Using device: cuda
Epoch 1/3: train_loss=0.0425 val_loss=0.0104 val_acc=0.9979
Epoch 2/3: train_loss=0.0113 val_loss=0.0094 val_acc=0.9980
Epoch 3/3: train_loss=0.0106 val_loss=0.0090 val_acc=0.9981
Best val accuracy: 0.9981 (epoch 3) -> example_run/trial_000/model.pth
[2/2] config={'hidden_sizes': (256, 128, 64, 32), 'dropout': (0.0, 0.2, 0.0, 0.0), 'lr': 0.003} seed=2095133045
Using device: cuda
Epoch 1/3: train_loss=0.0338 val_loss=0.0097 val_acc=0.9979
Epoch 2/3: train_loss=0.0092 val_loss=0.0089 val_acc=0.9980
Epoch 3/3: train_loss=0.0082 val_loss=0.0085 val_acc=0.9981
Best val accuracy: 0.9981 (epoch 3) -> example_run/trial_001/model.pth
Best trial: example_run/trial_000 (val_accuracy=0.9981)
[ble: elapsed 27.437s (CPU 109.9%)] uv run decoder/tune.py --dataset example_run/dataset.npz --trials 2 --epochs 3 --output-dir example_run/ --verbose


adversarial-blind-spots-in-quantum-error-correction-mlp-decoder on  Victor [✘!+?] is 📦 v0.1.0 via 🐍 v3.13.5 took 1m59s 
❯ uv run decoder/train.py --dataset example_run/dataset.npz --config example_run/trial_000/config.json --output-dir example_run/ --epochs 30 --verbose
Using device: cuda
Epoch 1/30: train_loss=0.0425 val_loss=0.0104 val_acc=0.9979
Epoch 2/30: train_loss=0.0113 val_loss=0.0094 val_acc=0.9980
Epoch 3/30: train_loss=0.0106 val_loss=0.0090 val_acc=0.9981
Epoch 4/30: train_loss=0.0101 val_loss=0.0091 val_acc=0.9981
Epoch 5/30: train_loss=0.0096 val_loss=0.0090 val_acc=0.9981
Epoch 6/30: train_loss=0.0094 val_loss=0.0090 val_acc=0.9980
Epoch 7/30: train_loss=0.0091 val_loss=0.0087 val_acc=0.9981
Epoch 8/30: train_loss=0.0091 val_loss=0.0085 val_acc=0.9981
Epoch 9/30: train_loss=0.0088 val_loss=0.0085 val_acc=0.9981
Epoch 10/30: train_loss=0.0086 val_loss=0.0084 val_acc=0.9981
Epoch 11/30: train_loss=0.0085 val_loss=0.0088 val_acc=0.9980
Epoch 12/30: train_loss=0.0083 val_loss=0.0088 val_acc=0.9981
Epoch 13/30: train_loss=0.0084 val_loss=0.0085 val_acc=0.9981
Epoch 14/30: train_loss=0.0082 val_loss=0.0083 val_acc=0.9981
Epoch 15/30: train_loss=0.0081 val_loss=0.0089 val_acc=0.9980
Epoch 16/30: train_loss=0.0081 val_loss=0.0089 val_acc=0.9981
Epoch 17/30: train_loss=0.0079 val_loss=0.0084 val_acc=0.9981
Epoch 18/30: train_loss=0.0079 val_loss=0.0083 val_acc=0.9981
Epoch 19/30: train_loss=0.0079 val_loss=0.0084 val_acc=0.9981
Epoch 20/30: train_loss=0.0078 val_loss=0.0085 val_acc=0.9981
Epoch 21/30: train_loss=0.0076 val_loss=0.0084 val_acc=0.9980
Epoch 22/30: train_loss=0.0077 val_loss=0.0085 val_acc=0.9981
Epoch 23/30: train_loss=0.0076 val_loss=0.0084 val_acc=0.9981
Epoch 24/30: train_loss=0.0075 val_loss=0.0084 val_acc=0.9980
Epoch 25/30: train_loss=0.0074 val_loss=0.0087 val_acc=0.9980
Epoch 26/30: train_loss=0.0075 val_loss=0.0085 val_acc=0.9980
Epoch 27/30: train_loss=0.0075 val_loss=0.0085 val_acc=0.9981
Epoch 28/30: train_loss=0.0074 val_loss=0.0084 val_acc=0.9981
Epoch 29/30: train_loss=0.0072 val_loss=0.0086 val_acc=0.9979
Epoch 30/30: train_loss=0.0073 val_loss=0.0086 val_acc=0.9980
Best val accuracy: 0.9981 (epoch 27) -> example_run/model.pth
[ble: elapsed 111.179s (CPU 104.7%)] uv run decoder/train.py --dataset example_run/dataset.npz --config example_run/trial_000/config.json --output-dir example_run/ --epochs 30 --verbose


```

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

## Results (to be updated as experiments complete)

| Experiment | d=3 depol | d=3 FT | d=5 depol | d=5 FT |
|---|---|---|---|---|
| MLP logical error rate | — | — | — | — |
| MWPM logical error rate | — | — | — | — |
| MLP blind-spot fraction | — | — | — | — |
| SA success rate (W=3) | — | — | — | — |
| Median robustness radius | — | — | — | — |

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
