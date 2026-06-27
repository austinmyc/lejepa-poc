## **Project:** LeJEPA-Text (Joint Embedding Predictive Architecture for language pretraining)


## Implementation

Built and validated the full proof-of-concept pipeline: bidirectional chunk encoder, projection MLP, causal predictor, SIGReg loss, and collapse diagnostics (embedding rank + mean cosine similarity). Ran small-scale trials on tiny-shakespeare and synthetic data to confirm the loop runs end-to-end.

**Literature read:**

- LeJEPA (original) — SIGReg formulation, projection MLP design, stop-gradient rationale
- I-JEPA — EMA teacher approach for vision JEPA
- VICReg / Barlow Twins — variance-invariance-covariance regularization as an alternative to SIGReg
- DINO — centering and normalization tricks for preventing mode collapse in self-distillation
- Ethayarajh (2019) — anisotropy in contextual embeddings; motivation for why projection into isotropic space is needed

---

## Main problem: representation collapse

Without contrastive negatives or masked reconstruction, the encoder can degenerate toward producing the same embedding for every input — giving the predictor a trivially easy target. EMA (as in I-JEPA/BYOL) was tried first but insufficient on its own.

Current approach uses two mechanisms:

1. **Projection MLP** — the encoder should remain free to be anisotropic (semantic geometry); the predictor needs an isotropic input to avoid exploiting low-rank shortcuts. The projection bridges these: it maps encoder output into isotropic space before the predictor, and SIGReg is applied there, not on the encoder directly.

2. **Stop-gradient on the SIGReg path** — blocks isotropy pressure from reaching the encoder. The encoder trains on MSE only; the raw encoder output is what we use at eval time.

---

## Plan for next week

- **Longer Shakespeare run** (~5k steps) — first real collapse stress test, confirm embedding rank stays high.
- **DINO centering trick** — subtract an EMA of the target mean from targets before MSE; try this as an additional anti-collapse measure.
- **Ablation: stop-grad vs. full-grad vs. no projection** — verify the stop-gradient is load-bearing.
- **Ablation: bidirectional vs. causal encoder** — compare rank and loss curves; decide before the L20 run.

