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
│   ├── generate_datasets.py        # Stim-based dataset generation (Z and X basis)
│   └── noise_models.py             # Simple depolarizing + circuit-level FT configs
│
├── decoder/
│   ├── mlp_decoder.py              # MLP architecture, training loop, evaluation
│   └── mwpm_reference.py           # PyMatching wrapper for MWPM baseline
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

```bash
pip install stim pymatching torch numpy scikit-learn matplotlib
```

Python 3.10+ recommended.

### 1 — Generate training data

```bash
python data/generate_datasets.py --distance 3 --rounds 6 --shots 1000000 --noise depolarizing
python data/generate_datasets.py --distance 3 --rounds 6 --shots 1000000 --noise circuit_level
```

### 2 — Train the MLP decoder

```bash
python decoder/mlp_decoder.py --distance 3 --noise depolarizing
```

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

**Depolarizing (simple):** uniform independent Pauli noise on data qubits only, perfect measurements. Good for baselines and threshold analysis.

**Circuit-level FT:** per-gate depolarizing noise, noisy measurements, noisy ancilla resets, and idle data qubit errors — all configurable per-qubit. Matches real hardware noise structure. Generated via `stim.Circuit.generated(..., after_clifford_depolarization=..., before_measure_flip_probability=...)`.

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
