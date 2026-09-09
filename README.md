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
│   ├── mlp_decoder.py              # MLP architecture, training loop, evaluation (WIP)
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
├── notebooks/
│   └── results_visualization.ipynb # Figures for the paper
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

Arguments: `--distance` (default 3), `--rounds` (default 9), `--samples` (default 100,000), `--noise` (`depolarizing` or `circuit-level`) (default `depolarizing`), `--output` (required), `--verbose`.

Each run writes two files :
- the `.npz` itself, with `"labels"` (shape `(samples, 1)`, logical observable flips) and `"features"` (shape `(samples, (d²-1)*rounds)`, detector/syndrome bits)
- a `.json` (same path, but with `.json` extension) recording `distance`, `rounds`, `noise_type`, `noise_default` — the parameters needed to rebuild the exact same circuit later (used by `mwpm_reference.py` below)

With `--verbose`, a short analytics summary is printed before saving.

### 2 — Train the MLP decoder

```bash
python decoder/mlp_decoder.py --distance 3 --noise depolarizing
```

*(work in progress)*

### 2b — Run the MWPM reference decoder

```bash
python decoder/mwpm_reference.py --metadata data/train_depolarizing.json --dataset data/train_depolarizing.npz --prediction-path data/mwpm_predictions.npz --verbose
```

Rebuilds the circuit from the dataset's `.json` file, decodes the syndromes in `--dataset` with PyMatching, and reports the logical error rate. `--prediction-path` is required and stores predictions, ground-truth labels, and the logical error rate as a `.npz`.

If `--dataset` is omitted, `--samples` (default 100,000) fresh shots are sampled directly from the rebuilt circuit instead — useful for a quick standalone MWPM sanity check, but **not** for a fair comparison against another decoder, since that draws an independent random syndromes. 
> [!WARNING]
> **Attention**
> Always pass `--dataset` pointing at a shared `.npz` when comparing MWPM against the MLP.

### 3 — Run the SA attack

```bash
python experiments/run_attack.py --distance 3 --budget 3 --restarts 50
```

### 4 — Run the decision boundary analysis

```bash
python experiments/run_boundary_analysis.py --distance 3 --shots 2000000
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

Defined in `data/noise_models.py` via `generate_noise_constants(noise_type, default=0.05, ...)`, which returns a kwargs dict fed straight into `stim.Circuit.generated(...)`. Any parameter left unspecified falls back to `default`.

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
