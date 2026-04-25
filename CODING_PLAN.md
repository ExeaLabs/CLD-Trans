# CLD-Trans: In-Depth Coding Plan

> Engineering blueprint for the **Causal-Lagged Dynamic Transformer (CLD-Trans)**.
> Goal: deliver a reproducible, modular, NeurIPS-grade codebase that swaps the
> static-graph backbone of `causalbioFM/` for a continuous-time, lag-aware,
> motif-quantized model — while reusing the existing data pipelines for
> CHB-MIT, PTB-XL, and Sleep-EDF.

---

## 0. Guiding Principles

- **Reuse, don't rewrite**: keep the validated dataset loaders, splits, and
  evaluation harness from `causalbioFM/`. Only `models/`, `modules/`, and
  `losses/` are new.
- **Theory drives code**: the LD-SEM identifiability result (paper §3) is the
  methodological contribution. Every module must support a clean test of
  that theorem on synthetic data **before** it is run on real biosignals.
- **Two-stage training**: Stage 1 jointly pretrains the motif VQ-VAE *and*
  the LD-SEM lag inferencer on **TUH-EEG + MIMIC-IV-ECG** with no labels;
  Stage 2 attaches task heads for few-/zero-shot downstream evaluation.
- **Determinism & reproducibility**: 5-seed protocol (`42, 123, 7, 0, 256`).
- **Environment**: plain Python `venv` + `pip` (or `uv`). The previous
  Docker container is no longer used — the MI300X server's ROCm stack is
  preconfigured.
- **Target hardware**: 8× AMD MI300X (1.5 TB aggregate VRAM), 160 vCPU,
  1920 GB RAM, 2 TB boot NVMe, **40 TB scratch NVMe** (datasets + caches
  live on scratch).

---

## 1. Repository Layout

```
CLD-Trans/
├── data/                         # loaders
│   ├── chbmit_loader.py           # downstream
│   ├── ptbxl_loader.py            # downstream
│   ├── sleepedf_loader.py         # downstream
│   ├── tuh_eeg_loader.py          # pretraining (~26k recs)
│   ├── mimic_ecg_loader.py        # pretraining (~800k ECGs)
│   ├── synthetic_ldsem.py         # ground-truth τ/W generators for theory
│   └── transforms.py             # patching, normalization, augmentations
├── modules/
│   ├── vq_tokenizer.py           # PhysiologicalMotifVAE + VectorQuantizer
│   ├── lag_inferencer.py         # Differentiable lag τ_ij + dynamic adjacency
│   ├── fractional_delay.py        # FFT phase-shift operator (closed-form gradients)
│   ├── flow_layers.py             # GraphODEFunc + adjoint odeint wrapper
│   └── positional.py             # Continuous-time / Fourier-time embeddings
├── models/
│   └── cld_transformer.py        # End-to-end CLD-Trans backbone
├── losses/
│   ├── task_loss.py               # CE / focal / multi-label BCE (fine-tune only)
│   ├── ldsem_loss.py              # Negative log-likelihood of LD-SEM (§3 of paper)
│   ├── vq_loss.py                # Codebook + commitment losses
│   └── regularizers.py           # Sparsity, smoothness on τ, ODE energy
├── engine/
│   ├── trainer_stage1.py         # VQ-VAE pretraining loop
│   ├── trainer_stage2.py         # Joint task + flow training loop
│   ├── evaluator.py              # Metrics, bootstrap CIs, per-class reports
│   └── callbacks.py              # EMA, ckpt, early stop, grad clip
├── analysis/
│   ├── propagation_maps.py        # CHB-MIT zero-shot focal-lead + spread visualizer
│   ├── conduction_dynamics.py     # PTB-XL lead-vector trajectories
│   ├── motif_atlas.py             # Codebook usage, NN motif retrieval
│   ├── identifiability.py         # Synthetic LD-SEM τ/W recovery curves
│   └── interpretability.py       # τ heatmaps, ODE phase portraits
├── configs/
│   ├── base.yaml
│   ├── chbmit.yaml
│   ├── ptbxl.yaml
│   └── sleepedf.yaml
├── scripts/
│   ├── train_stage1.sh           # Pretrain motif VQ-VAE
│   ├── train_stage2.sh           # Joint Task + Causal Flow training
│   ├── eval_all.sh               # 5-seed evaluation across datasets
│   └── make_figures.sh           # Reproduce every paper figure
├── tests/
│   ├── test_vq_tokenizer.py
│   ├── test_fractional_delay.py   # gradient correctness vs. autograd
│   ├── test_lag_inferencer.py
│   ├── test_flow_layers.py
│   ├── test_ldsem_recovery.py     # synthetic identifiability sanity check
│   └── test_end_to_end.py
├── results/                      # JSON metrics per seed/dataset
├── logs/                         # training logs (mirrors causalbioFM/logs)
└── main.py                       # Hydra entry point
```

---

## 2. Module-Level Specifications

### 2.1 `modules/vq_tokenizer.py`

**Purpose**: convert raw multivariate biosignal patches into discrete
"physiological motif" tokens.

- `class MotifEncoder`: 1-D depthwise-separable CNN, 4 blocks, GELU + GroupNorm,
  stride-based downsampling to a patch length of ~64 ms (CHB-MIT/Sleep-EDF) or
  ~40 ms (PTB-XL).
- `class VectorQuantizer`: EMA-updated codebook (`codebook_size=512`,
  `embed_dim=128`), straight-through estimator, dead-code revival every
  `revive_every=2k` steps.
- `class MotifDecoder`: mirror of encoder for reconstruction loss in Stage 1.
- `class PhysiologicalMotifVAE(nn.Module)`: orchestrates encoder + quantizer
  + decoder; exposes `encode_indices`, `decode_indices`, `forward`.
- Outputs: `z_q` (quantized embeddings), `indices` (token IDs), `commit_loss`,
  `codebook_loss`, `perplexity`.

### 2.2 `modules/lag_inferencer.py`

**Purpose**: learn a per-pair lag τ_ij and produce a time-varying adjacency.

- `class LearnableLagMatrix`: parameter `tau_raw ∈ R^{C×C}` mapped via
  `softplus` to `[0, τ_max]`; symmetric or asymmetric (configurable).
- `class FractionalDelay`: differentiable delay using FFT phase shift
  `X̂(f) * exp(-j2πfτ)`; supports sub-sample lags. Fallback: sinc interpolation
  for short windows.
- `class LaggedEdgeScorer`: shifts each channel by τ_ij, then scores
  `s_ij = MLP([h_i, h_j(t-τ_ij), |h_i - h_j(t-τ_ij)|])`; returns sigmoid edge
  probability and a Gumbel-soft top-k mask.
- Sparsity hooks for `flow_loss.py` (L1 on edge probs, entropy on τ).

### 2.3 `modules/flow_layers.py`

**Purpose**: continuous-time latent evolution conditioned on the dynamic graph.

- `class GraphODEFunc(nn.Module)`: `f(t, h) = MessagePassing(h, A(t)) + ϕ(t)`,
  where `ϕ` is a sinusoidal time embedding.
- `class CLDOdeBlock`: wraps `torchdiffeq.odeint_adjoint` with configurable
  solver (`dopri5`, `rk4`), `rtol=1e-4`, `atol=1e-5`, `max_steps=64`.
- Integrates over a per-sample time grid produced by the data loader.
- Returns the trajectory `H ∈ R^{T×C×D}` plus an estimated NFE for logging.

### 2.4 `modules/fractional_delay.py`

**Purpose**: realize sub-sample lags with closed-form gradients (referenced
by the identifiability theorem in paper §3).

- `class FractionalDelay`: $\hat x(f)\,e^{-j 2\pi f \tau}$ via `torch.fft`;
  exposes a custom `torch.autograd.Function` with analytic gradient w.r.t.
  $\tau$, validated against `torch.autograd.gradcheck` in tests.
- Supports batched per-pair $\tau$ tensors of shape `[C, C]`.

### 2.5 `models/cld_transformer.py`

Pipeline:

1. Patchify input → motif indices via `PhysiologicalMotifVAE`
   (trainable in Stage 1, frozen in Stage 2 fine-tune).
2. Embed indices + continuous-time positional encoding.
3. `LaggedEdgeScorer` (with `FractionalDelay`) produces `A(t_k)` at K
   anchor times.
4. `CLDOdeBlock` integrates latent state through `A(t)`.
5. Pooling head per task: temporal mean + class-token MLP for
   CHB-MIT/Sleep-EDF; multi-label sigmoid head for PTB-XL.
   The **headline zero-shot CHB-MIT result uses no head** — prediction is
   read directly off `τ`.

Configurable forward modes: `["pretrain_ldsem", "linear_probe", "fine_tune"]`.

---

## 3. Loss Functions

| Loss | File | Used In | Notes |
|------|------|---------|-------|
| Reconstruction (MSE + spectral) | `vq_loss.py` | Stage 1 | Adds STFT-magnitude term |
| Codebook + commitment | `vq_loss.py` | Stage 1 | β=0.25 |
| LD-SEM negative log-likelihood | `ldsem_loss.py` | Stage 1 | The training-time analogue of the identifiability objective in paper §3 |
| Cross-entropy / focal | `task_loss.py` | Stage 2 | CHB-MIT, Sleep-EDF (fine-tune only) |
| Multi-label BCE | `task_loss.py` | Stage 2 | PTB-XL (fine-tune only) |
| Edge sparsity (L1) | `regularizers.py` | Stage 1 | λ_sparse tuned on a held-out pretrain shard |
| τ smoothness | `regularizers.py` | Stage 1 | TV penalty on adjacent pairs |
| ODE energy | `regularizers.py` | Stage 1 | Penalizes NFE blow-up |

`ldsem_loss.py` realizes the score implied by the LD-SEM in paper §3:
maximize $\log p(z_i(t) \mid z_{-i}(t-\tau_{i,\cdot}))$ under a non-Gaussian
(e.g., Laplace or Student-t) innovation likelihood, providing gradients to
both $A$ and $\tau$ and matching the identifiability assumptions used in
the proof.

---

## 4. Data Pipeline

- Reuse existing `causalbioFM` loaders; expose them as `data/*_loader.py`
  thin wrappers returning `(x, y, t_grid, channel_meta)`.
- Add `data/transforms.py`:
  - Per-channel z-score with cached statistics.
  - Random temporal crop (Stage 1) and class-balanced window sampling
    (Stage 2 for CHB-MIT seizure scarcity).
  - Optional 50/60 Hz notch + 0.5–70 Hz band-pass for EEG/ECG.
- All loaders emit a `t_grid` aligned to the original sampling rate so the ODE
  integrator sees physical seconds.

### 4.1 Storage Layout (40 TB scratch NVMe)

All datasets and caches live on the server's scratch SSD:

```
/scratch/cld-trans/
├── datasets/
│   ├── chb-mit/             # ~121 GB — downstream + headline result
│   ├── ptb-xl/              # ~3 GB  — records100 + records500
│   ├── sleep-edf/           # ~8 GB  — sleep-cassette + telemetry
│   ├── tuh-eeg/             # ~600 GB — full TUH-EEG corpus
│   └── mimic-iv-ecg/        # ~200 GB — full MIMIC-IV-ECG
├── cache/
│   ├── motif_indices/       # int16 token shards from Stage 1
│   └── stats/               # per-channel normalization stats
└── checkpoints/
    ├── stage1/              # VQ + LD-SEM checkpoints
    └── stage2/              # downstream fine-tunes
```

With ~1 TB raw data + a few hundred GB of caches/checkpoints, the 40 TB
scratch is comfortably sized; **no streaming or subsetting is required**.
The boot NVMe (2 TB) hosts only the venv and code.

---

## 5. Training Protocol

### Stage 1 — Foundation-Scale Pretraining (no labels)
- **Corpora**: full **TUH-EEG (~26k recordings)** for the EEG branch and
  full **MIMIC-IV-ECG (~800k ECGs)** for the ECG branch; per-modality
  codebooks plus an ablation with a shared cross-modality codebook.
- **Objective**: VQ recon + codebook + LD-SEM NLL + sparsity/smoothness regs.
- **Optimizer**: AdamW, lr `3e-4`, cosine schedule, 100 epochs.
- **Hardware**: 8× MI300X with FSDP + bf16 + activation checkpointing;
  per-GPU micro-batch 32, gradient accumulation 1, global batch 256.
- **Logging**: codebook perplexity, dead-code count, recon MSE/STFT,
  per-pair τ distribution, edge density.

### Stage 2 — Downstream Evaluation
- **Zero-shot (CHB-MIT focal-lead)**: Stage-1 model used directly; predict
  focal lead as $\arg\min_i \sum_j \tau_{ij}$ over seizure-onset windows.
- **Linear probe**: freeze backbone, train a logistic head per task.
- **Few-shot fine-tune**: 1% / 10% / 100% label budgets; AdamW lr `1e-4`,
  warmup 1k steps, 60 epochs, EMA on backbone.
- **Per-dataset configs** (`configs/*.yaml`):
  - CHB-MIT: subject-wise leave-one-out, focal loss γ=2, window 30 s.
  - PTB-XL: 10-fold per official split, multi-label BCE, window 10 s.
  - Sleep-EDF: subject-wise 5-fold, balanced sampling, window 30 s.
- **Seeds**: `42, 123, 7, 0, 256`. With 8 GPUs, the 5-seed sweep per
  dataset can be parallelized one-seed-per-GPU on a single node.

### Scripts
- `scripts/setup_env.sh`: `python -m venv .venv && pip install -r requirements.txt`
  (PyTorch ROCm wheel for MI300X).
- `scripts/train_stage1.sh`: pretrain on full TUH-EEG + MIMIC-IV-ECG with
  `torchrun --nproc-per-node=8`.
- `scripts/eval_zero_shot.sh`: CHB-MIT focal-lead headline result.
- `scripts/train_stage2.sh`: linear probe + few-shot fine-tune sweeps.
- `scripts/eval_synthetic.sh`: identifiability validation on synthetic LD-SEM.
- `scripts/eval_all.sh`: aggregates `results/*.json` into a single CSV with
  bootstrap 95% CIs.

---

## 6. Evaluation & Metrics

| Dataset   | Primary | Secondary | Interpretability |
|-----------|---------|-----------|------------------|
| CHB-MIT   | AUROC, AUPRC, Sensitivity@FPR=0.1/h | Detection latency (s) | Focal-lead accuracy vs. clinician annotations |
| PTB-XL    | Macro AUROC (5 superclasses) | F1, balanced accuracy | Lead-conduction lag plots |
| Sleep-EDF | Macro F1, Cohen's κ | Per-stage F1 | Motif transition matrices |

All metrics are reported as `mean ± 95% CI` over the 5 seeds (BCa bootstrap,
1000 resamples) — same convention used by the current
`causalbioFM/results/*.json`.

---

## 7. Ablation Matrix

| Variant | VQ Motifs | Lag τ | Neural ODE | Pretrain Scale |
|---------|:---------:|:-----:|:----------:|:--------------:|
| A0 — Baseline (current paper) | ✗ | ✗ | ✗ | downstream-only |
| A1 — +Motifs only | ✓ | ✗ | ✗ | downstream-only |
| A2 — +Integer lag (DYNOTEARS-style) | ✓ | int | ✗ | downstream-only |
| A3 — +Continuous τ, no ODE | ✓ | ✓ | ✗ | downstream-only |
| A4 — Full CLD-Trans, downstream-only | ✓ | ✓ | ✓ | downstream-only |
| A5 — Full CLD-Trans, foundation pretrain | ✓ | ✓ | ✓ | TUH+MIMIC |

A5 is the headline configuration; A0–A4 are ablations. Each variant runs the
full 5-seed protocol on each downstream dataset.

---

## 8. Testing & CI

- `pytest` smoke tests for every module (forward shape, gradient flow,
  determinism with fixed seed).
- `tests/test_end_to_end.py`: 1-batch overfit on a synthetic 4-channel toy
  signal must reach > 0.99 train accuracy.
- GitHub Actions workflow (or local `make ci`) running lint (`ruff`),
  type check (`mypy --strict modules/ models/ losses/`), and unit tests.

---

## 9. Risk Register & Mitigations

| Risk | Likelihood | Mitigation |
|------|:----------:|------------|
| Identifiability assumptions violated by volume conduction | High | Re-state theorem on Laplacian-rereferenced signals; report τ recovery on referenced vs. raw data |
| Synthetic→real gap on focal-lead localization | Med | Pre-registered fall-back: report linear-probe focal-lead accuracy if zero-shot fails |
| ODE solver instability on long EEG windows | Med | Use adjoint + step-size caps; fall back to `rk4` fixed-step |
| Codebook collapse | Med | EMA updates + dead-code revival + entropy bonus |
| Pretraining compute cost (TUH+MIMIC) | Low | 8× MI300X (1.5 TB VRAM) handles full corpora; FSDP + bf16 keeps memory headroom |
| ROCm wheel / kernel quirks | Med | Pin PyTorch ROCm version; smoke-test FFT, scatter, and `torchdiffeq` adjoint on the MI300X early |
| Class imbalance (CHB-MIT seizures) | High | Focal loss + balanced sampler + per-subject thresholds |

---

## 10. Milestone Timeline (engineering only — no calendar dates)

1. **M1** — Repo scaffold, data loaders (incl. TUH-EEG, MIMIC-IV-ECG,
   synthetic LD-SEM), smoke tests green.
2. **M2** — Synthetic LD-SEM identifiability validated: τ / edge-support
   recovered to within published tolerance, matching the theorem.
3. **M3** — Stage-1 pretraining on TUH-EEG + MIMIC-IV-ECG completed; motif
   atlas figure ready.
4. **M4** — Zero-shot CHB-MIT focal-lead localization beats baselines.
5. **M5** — Few-shot transfer matrix on PTB-XL / Sleep-EDF complete.
6. **M6** — Reproducibility pass: clean container build, `make_figures.sh`
   reproduces every paper figure end-to-end.
