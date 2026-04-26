# CLD-Trans: NeurIPS Research Paper Plan

> **Working title**: *CLD-Trans: Identifiable Lagged Causal Discovery for
> Multivariate Biosignals via Motif-Quantized Latent Flows*
> **Target venue**: NeurIPS Main Track (Causality / Dynamics).
> **Status**: planning document. Statements below describe intended
> contributions, hypotheses, and experiments rather than completed results.

---

## 1. Headline Contributions

We propose **CLD-Trans**, a foundation-scale model for multivariate
biosignals built around a single sharp claim we want to test:
**per-pair propagation lags may be identifiable from observational
physiology under stated assumptions, and those lags may be clinically
actionable**. Concretely:

1. **Theory (§3).** We formulate biosignal generation as a continuous-time,
   non-Gaussian, additive-noise, lagged structural equation model (LD-SEM)
   in a learned discrete motif space. The target theoretical result is to
   show that the lag matrix τ and edge support are identifiable up to a
   label-permutation/sign equivalence class under (i) non-Gaussian
   innovations, (ii) bounded τ ≤ τ_max, and (iii) faithfulness — a delayed
   analogue of LiNGAM/Rhino-style results.
2. **Method (§4).** We instantiate LD-SEM with three components: a
   **PhysiologicalMotifVAE** discretizing raw waveforms into a 512-entry
   codebook, a **differentiable fractional-lag operator** (FFT phase shift)
   driving a **lagged edge scorer**, and a **graph-conditioned Neural ODE**
   that integrates latent state through the inferred A(t).
3. **Foundation-scale pretraining (§5).** The motif VQ-VAE would be pretrained on
  **EEGMMIDB (>1.5k public EEG recordings)** and **MIMIC-IV-ECG (~800k ECGs)**. The
   lean submission centers downstream evaluation on CHB-MIT zero-shot seizure
   propagation, with PTB-XL/Sleep-EDF retained as optional appendix transfer checks.
4. **Headline empirical result (§6).** **Zero-shot focal-lead localization
  on CHB-MIT** is the headline hypothesis: without training on seizure
  labels, CLD-Trans should recover clinician-annotated onset zones from the
  learned τ and ideally outperform static-graph and discrete-Transformer
  baselines.

---

## 2. Positioning vs. Prior Work

- **Biosignal foundation models** (BIOT, BENDR, NeuroLM, ECG-FM): tokenize
  biosignals but treat channels as either independent or via a static
  attention pattern; provide *no* causal/lag structure and *no* identifiability
  guarantees.
- **Static-graph biosignal models** (GraphSleepNet, EEG-GCNN): assume
  time-invariant, acyclic dependencies — violated by seizure spread,
  cardiac conduction, and sleep micro-architecture.
- **Neural ODEs on graphs** (LG-ODE, CG-ODE, MTGODE): integrate latent state
  on a *given* graph; do not infer lags or causal structure.
- **Time-delayed causal discovery** (DYNOTEARS, Rhino, CUTS+, NTS-NOTEARS):
  recover lagged DAGs but assume *integer* lags, *acyclicity*, and operate on
  raw observed channels — untenable for biosignals dominated by volume
  conduction and feedback loops.

**Our delta.** If successful, CLD-Trans would aim to be the first method to
(a) prove identifiability of **continuous-valued** lags in a *cyclic*,
*latent-motif* SEM, and (b) operate at foundation scale with provable lag
recovery as the optimization target.

---

## 3. Theory (paper §3)

### 3.1 Lagged-Delay Structural Equation Model (LD-SEM)
Let $z_i(t) \in \mathbb{R}^d$ denote the latent motif embedding of channel $i$
at time $t$. We posit

$$z_i(t) = \sum_{j \ne i} W_{ij}\, g\!\left(z_j(t - \tau_{ij})\right) + \varepsilon_i(t),$$

with $\tau_{ij} \in [0, \tau_{\max}]$, $W_{ij} \in \mathbb{R}^{d \times d}$,
and $\varepsilon_i$ mutually independent, **non-Gaussian**, with finite
variance.

### 3.2 Identifiability Theorem (informal)
*Under (A1) non-Gaussian $\varepsilon$, (A2) $\tau_{ij}$ bounded and pairwise
distinct on every directed cycle, (A3) faithfulness w.r.t. the lagged graph,
the pair $(\tau, \mathrm{supp}(W))$ is identifiable from the observational
distribution of $z(\cdot)$ up to a permutation of motif labels.*

Proof sketch follows the LiNGAM / Rhino strategy: cumulant-based
over-determination of the delayed covariance tensor pins down $\tau$ first,
then $W$ is recovered via the standard ICA argument applied per lag slice.
If the theorem is established, the full proof would appear in Appendix A.

## 4. Method (paper §4)

1. **Motif Quantizer** — `PhysiologicalMotifVAE` learns a 512-entry codebook
   of waveform motifs from raw patches; provides the latent $z$ in which
   LD-SEM is posed.
2. **Differentiable Fractional-Lag Operator** — FFT phase shift
   $\hat z(f)\,e^{-j 2\pi f\tau}$ realizes sub-sample lags with closed-form,
   provably correct gradients w.r.t. $\tau$.
3. **Lagged Edge Scorer** — MLP on $(z_i, z_j(t-\tau_{ij}))$ produces a
   continuous-time adjacency $A(t)$; trained to maximize the LD-SEM
   likelihood (which is the optimization target justified by §3.2).
4. **Graph-ODE Backbone** — $dh/dt = f(h(t), A(t), t)$ integrated with the
   adjoint method; provides the inference-time latent trajectory.
5. **Task Heads** — attached only at fine-tuning; the headline zero-shot
   result uses *no* head.

---

## 5. Datasets and Protocol (paper §5)

### 5.1 Pretraining (no labels used)

Pretraining uses the full open corpora; data lives on the training server's
40 TB scratch SSD.

| Corpus | Modality | Scale | Use |
|--------|----------|------:|-----|
| EEG Motor Movement/Imagery Dataset (EEGMMIDB) | EEG | >1.5k 64-channel recordings | Stage-1 motif VQ + LD-SEM lag pretraining |
| MIMIC-IV-ECG | ECG | ~800k open-access ECGs | Stage-1 motif VQ + LD-SEM lag pretraining |

### 5.2 Downstream Benchmarks
| Dataset   | Channels | Task | Regime |
|-----------|---------:|------|--------|
| CHB-MIT   | 23 EEG   | Seizure detection + **focal-lead localization** | **primary** zero-shot + 10%/100% few-shot |
| PTB-XL    | 12 ECG   | 5-superclass arrhythmia | compact 10%/100% ECG transfer |
| Sleep-EDF | 2–7 EEG/EOG | 5-stage sleep scoring | optional sleep transfer appendix |

Default post-Stage-1 runs use three seeds (`42, 123, 7`) on CHB-MIT and PTB-XL,
an uncapped CHB-MIT zero-shot evaluation, a tiny validation-only learning-rate
sanity sweep, and `42, 123, 7, 0, 256` only for the camera-ready appendix.

---

## 6. Experimental Plan

### 6.1 Headline Result — Zero-Shot Focal-Lead Localization (CHB-MIT)
- **Table 1**: top-1 / top-3 / top-5 focal-lead accuracy vs. clinician
  annotations, **with no seizure labels used in training**. Baselines:
  BIOT, BENDR, EEG-GCNN, DYNOTEARS, Rhino. CLD-Trans's prediction comes
  directly from the learned $\tau$ matrix at seizure-onset windows.

### 6.2 Few-Shot Transfer
- **Table 2**: AUROC/AUPRC under 10% / 100% label budgets on CHB-MIT and
  PTB-XL, both with three-seed uncertainty.

### 6.3 Identifiability Validation (Synthetic)
- **Table 3**: on synthetic LD-SEM data with known ground-truth $\tau$ and
  $W$, report (a) $\tau$ recovery error, (b) edge-support F1, (c) effect of
  violating non-Gaussianity (A1) and faithfulness (A3). Confirms the theorem
  empirically and exposes failure modes.

### 6.4 Ablations
- **Table 4**: A0–A4 grid (no-VQ, no-lag, integer-lag, no-ODE, full).

### 6.5 Robustness
- Channel dropout (0–40%), resampling factor, and leave-one-subject-out
  generalization on CHB-MIT.

### 6.6 Compute & Efficiency
- Pretraining wall-clock, NFE distribution, fine-tuning cost vs. baselines.

---

## 7. Planned Figures

> Each figure has a clear takeaway and a draft caption. All figures are
> reproducible via `scripts/make_figures.sh`.

### Figure 1 — Motivation: Why Static Graphs Fail
- **Panels**: (a) seizure spreading across CHB-MIT scalp montage at 0/2/5 s;
  (b) ECG QRS propagation across PTB-XL leads; (c) static-graph model failure
  case (predicted edges ≠ ground-truth propagation).
- **Takeaway**: physiological dependencies are time-varying and lagged.
- **Source**: `analysis/propagation_maps.py`.

### Figure 2 — CLD-Trans Architecture
- **Panels**: schematic of (1) Motif VQ-VAE → (2) Lagged Edge Scorer →
  (3) Graph-ODE → (4) Lag-Biased Attention → (5) Task Heads.
- **Takeaway**: unified continuous-time pipeline.
- **Source**: hand-drawn / TikZ; data not required.

### Figure 3 — Motif Atlas (Stage 1)
- **Panels**: (a) t-SNE of codebook vectors colored by dominant frequency;
  (b) top-12 motif waveforms with their nearest real exemplars from each
  dataset; (c) per-dataset codebook usage histograms.
- **Takeaway**: codebook captures cross-domain physiological primitives.
- **Source**: `analysis/motif_atlas.py`.

### Figure 4 — Learned Lag Heatmaps
- **Panels**: τ matrices for (a) CHB-MIT seizure vs. interictal,
  (b) PTB-XL normal vs. RBBB, (c) Sleep-EDF N2 vs. N3.
- **Takeaway**: lags shift in physiologically plausible directions per
  pathology / state.
- **Source**: `analysis/interpretability.py`.

### Figure 5 — **Headline**: Zero-Shot Seizure Propagation Maps (CHB-MIT)
- **Panels**: (a) ground-truth focal lead from clinician annotation;
  (b) CLD-Trans top-1 focal lead **with no seizure labels seen in training**;
  (c) propagation arrows over the 10–20 montage at t = 0, 2, 5 s;
  (d) bar chart of zero-shot focal-lead top-k accuracy vs. BIOT, BENDR,
  EEG-GCNN, DYNOTEARS, Rhino.
- **Takeaway**: a foundation model with identifiable lags localizes seizure
  onset zones without supervision — baselines do not.
- **Source**: `analysis/propagation_maps.py`.

### Figure 6 — Conduction Dynamics (PTB-XL)
- **Panels**: (a) 12-lead vector trajectory in 2-D PCA of ODE latent space
  for normal sinus, RBBB, LBBB; (b) per-pair lag distributions across
  superclasses; (c) AUROC vs. lag-perturbation magnitude.
- **Takeaway**: latent ODE trajectory separates conduction abnormalities;
  lags are diagnostic.
- **Source**: `analysis/conduction_dynamics.py`.

### Figure 7 — Sleep Motif Transitions (Sleep-EDF)
- **Panels**: (a) motif-transition matrix per sleep stage;
  (b) emergence-rate of "slow-wave" motifs around N2→N3 transitions;
  (c) confusion matrix vs. AASM scorers.
- **Takeaway**: discrete motifs explain stage transitions better than
  band-power features.
- **Source**: `analysis/motif_atlas.py`.

### Figure 8 — Ablation Grid
- **Panels**: bar chart of A0–A5 on the primary metric of each dataset
  with 95% CI error bars.
- **Takeaway**: each component (motifs, lag, ODE, lag-biased attention)
  contributes additively.
- **Source**: `engine/evaluator.py` aggregation.

### Figure 9 — Robustness Curves
- **Panels**: (a) performance vs. % channels dropped;
  (b) performance vs. resampling factor;
  (c) leave-one-subject-out per-subject scatter.
- **Takeaway**: CLD-Trans degrades gracefully where baselines collapse.
- **Source**: `engine/evaluator.py`.

### Figure 10 — Compute & Efficiency Frontier
- **Panels**: (a) AUROC vs. parameter count;
  (b) AUROC vs. training wall-clock (h);
  (c) NFE distribution per epoch (violin).
- **Takeaway**: CLD-Trans sits on the Pareto frontier despite ODE overhead.
- **Source**: training logs in `logs/`.

---

## 7. Tables Plan

| # | Title | Content |
|---|-------|---------|
| 1 | Main results | CLD-Trans vs. 6 baselines × 3 datasets |
| 2 | Ablations | A0–A5 grid |
| 3 | Interpretability | Focal-lead top-k acc., conduction-lag effect sizes, motif-transition KL |
| 4 | Robustness | AUROC under channel dropout / resample / LOSO |
| 5 | Hyperparameters | Per-dataset configs from `configs/*.yaml` |
| 6 | Compute | Params, FLOPs, NFE, wall-clock |

---

## 9. Paper Outline

1. **Introduction** — clinical motivation + the 4 headline contributions.
2. **Related Work** — biosignal foundation models, static-graph biosignal
   models, graph Neural ODEs, time-delayed causal discovery.
3. **Theory** — LD-SEM, identifiability theorem, proof sketch.
4. **Method** — motif quantizer, fractional-lag operator, lagged edge scorer,
   graph-ODE, training objectives.
5. **Experimental Setup** — pretraining corpora, downstream benchmarks,
   baselines, metrics, seeds.
6. **Results** — zero-shot focal-lead localization, few-shot transfer,
   synthetic identifiability validation, ablations, robustness, compute.
7. **Discussion** — biological realism, identifiability limitations
   (volume conduction, faithfulness), broader impact.
8. **Conclusion**.
9. **Appendix** — full proof, hyperparameters, additional figures.

---

## 10. Reproducibility Statement

- All code released under MIT in `CLD-Trans/`.
- Pure Python `venv` setup; PyTorch ROCm wheel for AMD MI300X (CUDA wheel
  for users on NVIDIA hardware).
- Pretrained motif VQ-VAE checkpoints released for EEGMMIDB and
  MIMIC-IV-ECG corpora.
- Per-seed JSON metrics in `results/`; logs in `logs/`.
- `scripts/make_figures.sh` reproduces every figure end-to-end.

---

## 11. Risks to the Narrative

| Risk | Mitigation |
|------|------------|
| Identifiability assumptions violated by volume conduction | Re-state theorem on Laplacian-rereferenced signals; report empirical τ recovery on referenced vs. unreferenced data |
| Zero-shot focal-lead result fails | Fall back to *self-supervised* (label-free pretrain + lag-only readout) headline; pre-registered |
| Synthetic identifiability gap to real EEG | Report a graded synthetic→semi-synthetic→real evaluation in Appendix |
| ODE adds compute without accuracy gain | Pre-register no-ODE ablation; if equal, frame ODE as buying interpretability — still consistent with the theory contribution |
| Reviewer flags “combination of known parts” | Identifiability theorem (§3) is the methodological delta; foreground it |
