# CLD-Trans: Project Overview

> A standalone introduction to the project for readers who are not familiar
> with the prior `causalbioFM` codebase or the team's earlier work.
>
> Status: this document describes the proposed research direction and target
> claims for CLD-Trans. Theorems, experimental results, and benchmark wins are
> goals to validate, not results already established.

---

## TL;DR

**CLD-Trans** (*Causal-Lagged Dynamic Transformer*) is a proposed foundation
model for multivariate physiological signals — EEG, ECG, polysomnography —
designed to learn **how signals propagate through the body in continuous
time**. The intended method combines three ideas that, to our knowledge, no
prior biosignal model has unified in one system:

1. **Tokenizes** raw waveforms into a learned library of discrete
   *physiological motifs* (e.g., spike-and-wave discharges, QRS complexes,
   K-complexes) via a Vector-Quantized VAE.
2. **Infers a per-channel-pair propagation lag** $\tau_{ij}$ — a continuous,
   sub-sample delay — using a differentiable Fourier phase-shift operator.
3. **Evolves latent state with a graph-conditioned Neural ODE**, so that the
   model's notion of "what happens next" is governed by the same dynamical
   system used in physics and physiology, not by discrete attention layers.

Crucially, CLD-Trans is built around a **theoretical identifiability goal**:
under stated, testable assumptions, we aim to show that the lag matrix $\tau$
and the underlying causal graph are recoverable from observational data
alone. If that theorem goes through, the model becomes a causal-discovery
tool, not just a predictor.

---

## The Clinical Problem

Standard biosignal classifiers ignore three facts that any clinician knows:

1. **Signals propagate.** A seizure starts at a focal lead (e.g., F7) and
   spreads to its neighbors over hundreds of milliseconds. The ECG R-wave
   reaches lead V6 a few milliseconds after V1.
2. **Dependencies cycle.** Cortical regions feed back on each other; cardiac
   conduction loops through the AV node. Acyclic graph models (the standard
   in causal discovery) are biologically wrong.
3. **Dynamics are continuous, not discrete.** Sampling a recording at 256 Hz
   is an artifact of the ADC, not a feature of the underlying physiology.

Existing biosignal foundation models (BIOT, BENDR, NeuroLM, ECG-FM) ignore
all three. Existing causal-discovery methods (NOTEARS, DYNOTEARS, Rhino)
assume integer lags and acyclicity. CLD-Trans is built to address the gap.

---

## What's New

| Aspect | Prior biosignal FMs | Prior causal discovery | **CLD-Trans (target)** |
|--------|---------------------|------------------------|---------------|
| Channel interactions | static / none | lagged DAG | **continuous-lag, possibly cyclic graph** |
| Lag granularity | n/a | integer samples | **sub-sample, differentiable** |
| Temporal evolution | discrete attention | linear VAR | **graph-conditioned Neural ODE** |
| Identifiability guarantee | none | classical (DAG, integer lag) | **non-Gaussian LD-SEM theorem** |
| Pretraining scale | 1.5k–800k recordings | small | **EEGMMIDB + MIMIC-IV-ECG** |
| Headline downstream claim | supervised AUROC | synthetic recovery | **zero-shot focal-lead localization on CHB-MIT** |

---

## How It Works (High Level)

```
 raw multichannel signal
        │
        ▼
 ┌──────────────────────┐
 │  Motif VQ-VAE         │  → discrete motif tokens per channel/patch
 └──────────────────────┘
        │
        ▼
 ┌──────────────────────┐
 │  Differentiable       │  → per-pair lag τ_ij ∈ [0, τ_max]
 │  Lag Inferencer       │  → time-varying adjacency A(t)
 └──────────────────────┘
        │
        ▼
 ┌──────────────────────┐
 │  Graph Neural ODE     │  dh/dt = f(h(t), A(t), t)
 └──────────────────────┘
        │
        ▼
 task head (only at fine-tune)
```

The planned training objective during pretraining is the **negative
log-likelihood of a Lagged-Delay Structural Equation Model (LD-SEM)** with
non-Gaussian innovations — matching the objective we would use in the
identifiability argument. Task heads would be attached only for downstream
fine-tuning. The headline CHB-MIT hypothesis is that no task head should be
necessary: the focal lead could be read off directly from the learned $\tau$
as $\arg\min_i \sum_j \tau_{ij}$.

---

## Datasets

### Pretraining (no labels)
- **EEG Motor Movement/Imagery Dataset (EEGMMIDB)** — over 1,500 public
  64-channel EEG recordings from 109 volunteers, sampled at 160 Hz.
- **MIMIC-IV-ECG** — ~800,000 open-access 12-lead ECGs from BIDMC patients.

Data lives on the training server's 40 TB scratch SSD; no streaming or
subsetting is required.

### Downstream Benchmarks
- **CHB-MIT** (23-channel pediatric scalp EEG, seizure annotations) — the
  zero-shot focal-lead localization benchmark and the headline result.
- **PTB-XL** (12-lead clinical ECG, 5 superclasses) — few-shot arrhythmia
  classification with a focus on conduction-delay-sensitive classes (RBBB,
  LBBB).
- **Sleep-EDF** (PSG with 2–7 EEG/EOG channels, AASM stage labels) —
  few-shot 5-stage sleep scoring.

The downstream datasets are expected to live on the training server scratch
volume alongside the pretraining corpora; lightweight metadata, splits, and
loader code can live in this repository.

---

## Why This Belongs at NeurIPS

NeurIPS rewards **methodological novelty backed by theory and a sharp,
reproducible empirical claim**. If executed well, CLD-Trans could offer:

- **Theory** — a plausible identifiability theorem for continuous-lag,
  possibly cyclic structural equation models in a learned latent space,
  extending the LiNGAM / Rhino line of work.
- **Method** — a closed-form differentiable fractional-lag operator, a
  graph-ODE that uses the inferred lag-graph as its vector field, and a
  motif quantizer that may make the latent space discrete enough for the
  identifiability argument to apply.
- **Empirical claim** — a testable *zero-shot* focal-lead localization
  hypothesis on CHB-MIT, without training on seizure labels. If supported,
  that would be a sharp and surprising result.

The three downstream datasets serve to validate the method, not to win
leaderboards.

---

## Repository Layout

This repo contains:

- `New_Project.md` — the original brainstorming doc that seeded the project.
- [CODING_PLAN.md](CODING_PLAN.md) — the engineering blueprint
  (modules, losses, training protocol, ablations, tests, risks).
- [PAPER_PLAN.md](PAPER_PLAN.md) — the NeurIPS paper plan
  (theorem statement, experimental matrix, planned figures, narrative risks).

Datasets are *not* checked into this repo — they live on the training
server's scratch volume at `/scratch/cld-trans/datasets/`.

## Compute

Training target: an 8× AMD **MI300X** node (1.5 TB aggregate VRAM, 160 vCPU,
1920 GB RAM, 40 TB scratch NVMe). Code runs in a plain Python `venv` with
the PyTorch ROCm wheel — no container required.

---

## Glossary

- **Motif**: a short, recurring waveform pattern (spike-wave, QRS, K-complex)
  represented as a discrete token in the VQ-VAE codebook.
- **Lag $\tau_{ij}$**: the time delay between channel $j$'s influence and
  channel $i$'s response. Continuous (sub-sample) and learnable.
- **LD-SEM**: Lagged-Delay Structural Equation Model — the generative model
  underlying CLD-Trans's identifiability theorem.
- **Graph Neural ODE**: an ordinary differential equation
  $dh/dt = f(h(t), A(t), t)$ whose vector field depends on the inferred
  graph $A(t)$.
- **Focal lead**: the EEG electrode at which a seizure begins; the gold
  standard is determined by clinician annotation.
- **Zero-shot**: evaluated without using any labels from the downstream
  task during training.
