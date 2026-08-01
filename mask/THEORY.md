# Theory notes — chance floors (proved), and why prediction ≠ contrast (working)

Status key: **PROVED** = deductive, assumptions verified on our actual runs.
**WORKING** = derivation sketch, to be checked before entering the paper.
**CONJECTURE / REGISTERED PREDICTION** = written down BEFORE the deciding runs.

Scope discipline: the propositions bind the real system because their
assumptions are *measured properties of our runs*, not modeling idealizations.
The linear analysis is a toy — it explains and constrains claims; it does not
prove anything about 12-layer transformers. The paper must always say "across
all tested configurations", never "cannot".

---

## Proposition 1 — the 2/P per-token floor. **PROVED**

**Statement.** Let u, v be unit vectors in R^P (normalized prediction and
target) and ℓ = (1/P)‖u − v‖² the per-coordinate MSE. Then

    E[ℓ] = (2 − 2·E⟨u, v⟩) / P .

In particular E[ℓ] = 2/P iff E⟨u, v⟩ = 0, and E[ℓ] < 2/P iff the mean cosine
alignment is strictly positive.

**Proof.** ‖u − v‖² = ‖u‖² + ‖v‖² − 2⟨u,v⟩ = 2 − 2⟨u,v⟩; take expectations. ∎

**Sufficient conditions for E⟨u,v⟩ = 0** (either suffices):
(i) u independent of v and E[v] = 0 (isotropized targets);
(ii) u ≡ c constant (collapsed predictor) and E[v] = 0.

**Assumption verification (RQ1, 120k ckpt, diagnose.py):**
predictor ≈ constant — cos between predictions from unrelated contexts = 0.90,
Var(pred)/Var(target) = 0.011, eff_rank(pred) = 5.7/768; targets isotropized by
SIGReg (batch mean ≈ 0, target eff_rank 340.9). **Predicted floor 2/768 =
0.0026; measured training MSE 0.0025.**

**Diagnostic corollary.** Training loss pinned at 2/P for the whole run ⟺ zero
mean directional alignment ⟺ the latent objective is contributing no usable
signal about targets. This is a pre-registered, compute-free test applicable to
any normalized latent-prediction objective.

---

## Proposition 2 — the pooled whitened floor 1/L + (1−1/L)ρ̄. **PROVED**

**Statement.** Let z₁…z_L ∈ R^P have per-coordinate unit variance and zero mean
under the data distribution (the SIGReg constraint), with average per-coordinate
cross-position correlation ρ̄. Let z̄ = (1/L)Σᵢ zᵢ. Then per coordinate

    Var(z̄) = 1/L + (1 − 1/L)·ρ̄ ,

and any predictor independent of the conditioning context has E-MSE ≥ Var(z̄),
with equality for the constant mean-predictor. Hence the **mean-only floor**

    F(L, ρ̄) = 1/L + (1 − 1/L)·ρ̄ .

**Proof.** Variance of a mean of correlated unit-variance variables; the
context-free optimum is the unconditional mean, whose MSE is the variance. ∎

**Assumption verification (wave 1 trajectories):**
- l_span flatlined at **0.125 = 1/8 exactly** (L=8) ⇒ ρ̄ ≈ 0 — the whitening
  also *decorrelated positions*, itself a finding the formula extracts.
- l_glob flatlined at **0.011** (L=128): F = 1/128 + (127/128)·ρ̄ = 0.011 ⇒
  ρ̄ ≈ 0.003. The formula explains the residual the ledger glossed as "≈1/128".

**Diagnostic corollary.** A pooled latent loss that reaches F within ~2k steps
and never descends below it has zero context-predictable variance — the
"stabilize → sterilize" effect, quantified. This happened in every SIGReg-
coupled pooled arm.

---

## Linear two-view analysis. **WORKING** — and it forces a sharper claim

**Setup.** Views (x_a, x_b) jointly Gaussian, cross-covariance Σ_ab. Linear
encoder W, linear predictor M, stop-grad target, whitening constraint on target
codes (SIGReg idealized as W Σ_bb Wᵀ = I):

    min_{W,M}  E‖M W x_a − sg(W x_b)‖²   s.t.   W Σ_bb Wᵀ = I .

**Reduction.** Given W, the optimal predictor is the cross-code regression
M*(W) = WΣ_ab Wᵀ (WΣ_aa Wᵀ)⁻¹. Substituting, the constrained objective becomes
maximizing the sum of squared canonical correlations captured by the subspace —
i.e. **linear JEPA with target whitening is CCA**: its optimum spans the top
canonical directions of the pair distribution, achieving loss Σᵢ(1 − ρᵢ²) over
the retained directions.

**Consequence (important, and against our first intuition):** the from-scratch
failure is **NOT information-theoretic**. A linear predictive learner with exact
optimization provably extracts the pairing. Therefore the observed failure must
live in the *dynamics*, and the paper's claim must be stated as a dynamics
claim, not an information claim:

1. At random deep init, code-space predictable variance ≈ 0 (measured: R² ≈ 0
   at the floor; the init-time alignment between the two views' random-feature
   kernels is the only seed signal).
2. The soft whitening gradient dominates the prediction gradient (measured 25×
   at λ=.05, calibrated ~3× at λ=.006) and shapes codes toward context-
   independent isotropy before weak correlations can amplify (the rich-get-
   richer regime of Tian et al. 2021).
3. The reachable fixed point is exactly the Prop-1/Prop-2 floors.

**Central mechanism conjecture (CONJECTURE — the paper's contrast).**
In predictive SSL, the anti-collapse term and the signal term are
**antagonistic**: whitening the target marginal destroys the conditional
structure prediction feeds on (cf. splitspace: the toxin travels through target
geometry, not the regularizer gradient). In contrastive SSL they are the
**same term**: the in-batch denominator supplies uniformity *and* the
alignment contrast (Wang & Isola 2020). Hence negatives matter from scratch
not because pairs lack predictive signal (CCA says they don't) but because
contrast is the one mechanism whose collapse-prevention does not fight its
signal source.

**Literature anchors.** Tian, Chen, Ganguli 2021 (non-contrastive dynamics);
Garrido et al. 2022 (contrastive/non-contrastive duality); Wang & Isola 2020
(alignment/uniformity); HaoChen et al. 2021 (provable guarantees exist for the
contrastive side — the asymmetry in the theory literature itself supports the
contrast); Ethayarajh 2019 + Gao et al. 2019 (anisotropy of text embedding
geometry — grounds the "isotropic prior is mismatched to language" premise).

---

## Registered predictions (written 2026-07-27, BEFORE the deciding runs)

| # | Prediction | Deciding run(s) |
|---|---|---|
| P1 | `llmjepa` (λ=0) ≈ `mlmonly`: without SIGReg the cross-view term is **inert, not harmful** (cf. Part I round-1: mlm_jepa .3726 ≈ mlm_only .3806) | llmjepa, mlmonly |
| P2 | `mlmonly` − `anchored` ≈ the SIGReg tax, ~.03–.08 (so anchored's deficit vs .4485 is pipeline + SIGReg, not the pairing) | mlmonly |
| P3 | `contrastive` ≫ `con_shuffled` on cross-view retrieval (r@1 by ≥10×) and ≫ `pure` (.086/.070) on MTEB — the pairing signal **is** extractable from scratch by contrast | contrastive, con_shuffled |
| P4 | P3 holding while anchored ≈ shuffled ⇒ the antagonism conjecture stands: same pairs, contrast extracts what prediction cannot | all |

**Falsification honesty:** if `contrastive` ≈ `con_shuffled`, the conjecture is
wrong and the claim reverts to the broader null "pairs are not exploitable from
scratch by any tested mechanism" — still publishable because the control
structure makes the null interpretable either way.

## OUTCOMES (2026-07-31) — all four predictions confirmed

| # | Prediction | Outcome |
|---|---|---|
| P1 | llmjepa ≈ mlmonly (λ=0 cross-view inert, not harmful) | ✅ code .3464 vs .3401 |
| P2 | mlmonly ≥ anchored; deficit vs .4485 is pipeline + SIGReg | ✅ both corpora: no-JEPA ≥ no-SIGReg > +SIGReg (summary .4298/.4184/.3868); SIGReg tax ≈ .03–.04 |
| P3 | contrastive ≫ con_shuffled while JEPA real ≈ shuffled | ✅ pairing effect **+.296 / +.269** (contrastive) vs **−.006 / +.005** (JEPA) |
| P4 | ⇒ antagonism conjecture stands | ✅ same pairs/encoder/batch/steps — only the objective differs |

**Interpretation.** The CCA reduction said the pair signal is extractable in
principle; the runs now show it IS extractable in practice from scratch — by
contrast, not by prediction. This rules out "the from-scratch regime is simply
too weak" (the alternative explanation for the null) and localises the failure
to the predictive objective's dynamics, as conjectured. The conjecture is
supported, not proved: it survives a falsification test it could have failed.

**Remaining theoretical work (optional, for the ICLR version):** the linear
init-dynamics lemma quantifying the gap — predictive gradient's pairing
alignment at init ~O(1/√d) vs InfoNCE's O(1) contrast signal. The empirical
~50× asymmetry is the target the lemma should explain.
