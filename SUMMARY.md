# Project summary (2026-08-04)

Complete account of what has been run, how, and what it establishes.
Detail: [mask/RESULTS.md](mask/RESULTS.md) (every run), [mask/THEORY.md](mask/THEORY.md)
(derivations), [ft/PLAN.md](ft/PLAN.md) (next phase).

---

## 1. Background: what is being tested

A language model is normally trained by **predicting tokens** — mask a word, guess
which word it was (BERT), or guess the next word (GPT). The model is scored
against the actual vocabulary item.

**JEPA** (Joint-Embedding Predictive Architecture, LeCun's proposal) replaces that
with prediction **in embedding space**. Take two related inputs A and B; encode
both; train a small "predictor" network so that

$$\text{Predictor}(\text{Enc}(A)) \approx \text{Enc}(B)$$

No tokens are predicted — the model predicts *the other view's vector*. The
appeal is that the model can ignore unpredictable surface detail and represent
only what matters semantically. This works very well in vision (I-JEPA, V-JEPA).

Two published papers apply it to language and report large gains:
**LLM-JEPA** (Sep 2025) uses text↔code pairs; **DLLM-JEPA** (May 2026) uses two
masking rates of the same text, reporting GSM8K 42.61 → 61.33 on LLaDA-8B.
**Both apply it as finetuning on an already-pretrained model.**

Three components recur and matter for reading the results below:

- **The predictor** — a small network mapping A's embedding toward B's.
- **Stop-gradient** — the target Enc(B) is detached, so the model can't cheat by
  moving the target instead of improving the prediction.
- **A collapse preventer** — without one, the encoder can output a constant
  vector for everything and score a perfect loss. Options: an **EMA teacher**
  (target from a slowly-updated copy), or a regularizer like **SIGReg** (LeJEPA's
  method: force embeddings toward an isotropic Gaussian).

Our loss throughout is the standard one, normalized MSE, which is *identical* to
the cosine loss these papers use: for unit vectors,
$\frac{1}{P}\lVert u-v\rVert^2 = \frac{2}{P}(1-\cos)$.

## 2. How everything is measured

After training we **freeze the encoder**, mean-pool its token vectors into one
vector per sentence, and score on a 4-task MTEB slice: STS-Benchmark and SICK-R
(correlation with human similarity judgements), Banking77 (77-way intent
classification), TwentyNewsgroups (clustering). The reported number is their mean.

Reference points on this scale:

| | score |
|---|---|
| BERT-base, mean-pooled (no finetuning) | **.4825** |
| Our plain-MLM model, same architecture and budget | **.4485** |
| An untrained / non-learning encoder | **≈ .05–.10** |

Seed noise is **σ ≈ .01**, so differences below roughly .02 are not meaningful.

---

## 3. Phase 1 — Can latent prediction train a text encoder from scratch?

~40 runs. 768-dim, 12-layer bidirectional transformer, OpenWebText, 30k steps
(120k for the scale check). The two views are the **same text**, one masked and
one clean.

**3a. Pure JEPA, no token supervision (13 runs).** Swept the collapse preventer
(SIGReg weight λ, EMA teacher, gradient shielding), target normalization, mask
ratio (15/30/50/75%) and mask strategy (random/span/block).

| Best | Worst | vs plain MLM |
|---|---|---|
| .164 | .052 | .4485 |

Nothing helped. Notably, "mask harder" — the trick that makes I-JEPA work in
vision — made results *monotonically worse*.

**3b. Why it failed — a derivation, not a guess.** A model that has learned
**nothing** still shows a small loss. We derived that "know-nothing" value:

- Per-token normalized targets in 768 dims → chance loss = 2/768 = **.0026**.
  Measured training loss: **.0025**, for the entire run.
- Pooled 8-token targets under SIGReg → chance loss = 1/8 = **.125**.
  Measured: exactly **.125**, reached within 2k steps and flat for 28k more.

Diagnostics confirmed the mechanism: the predictor collapsed to a near-constant
output (effective rank 5.7 of 768; R² over a constant predictor = 0.007).

The reason is a chicken-and-egg problem. At the start of training the target
embeddings are essentially random, so "move toward this vector" is an instruction
pointing at noise — and SIGReg, which prevents collapse by whitening the targets,
removes what little structure they had. It stabilizes and sterilizes at once.

**3c. Adding a token-prediction anchor (5 runs).** Adding an MLM loss makes
learning work (.30–.45 vs .164), and *where* it attaches matters more than
expected: on the encoder .4485, on the predictor .3806. Attaching SIGReg to the
space MLM must decode from costs a further .08.

**3d. Does the JEPA term then add anything? (6 loss families, ~15 runs.)**

| variant | result |
|---|---|
| per-token latent MSE | .3726 vs .3806 MLM-only — neutral |
| span-pooled prediction | .222 (SIGReg-coupled — harmful) |
| global pooled prediction | .306 (harmful) |
| encoder-space targets (no SIGReg) | .4307 vs .4311 ctrl — exact tie |
| diffusion head (distributional prediction) | .4254 — inside seed noise |
| manifold contraction | neutral |

Also tested: 4 different readout methods (few-shot probes, token-level NER,
learned attention pooling), convergence curves at 10k/20k/30k, and 4× the
training budget (120k steps). **Every JEPA number sat inside the control's
seed-noise band.** Once token supervision exists, the latent term adds nothing.

**3e. The one published from-scratch success, dissected (2 runs).** data2vec is
the only reported case of latent prediction working from scratch in text. Its
recipe uses layer-averaged, instance-normalized EMA targets. Reproduced:

| | score | reading |
|---|---|---|
| no anchor (pure JEPA) | .164 | dead |
| data2vec targets, no token loss | **.2245** | first latent-only objective to escape the floor (R² ≈ .9) |
| explicit MLM anchor | .4485 | much stronger |
| data2vec targets **+** MLM | .4471 | adds nothing on top |

This quantifies the whole picture: data2vec works because layer-averaging pulls
its targets toward the input-indexed embedding layer — an *implicit* anchor —
which is real but weaker than explicit token supervision, and redundant with it.

**3f. Post-hoc geometry.** Whitening and manifold-fitting the frozen embeddings
trade one task family against another; neither beats the raw embeddings on the
mean. No free lunch.

---

## 4. Phase 2 — Does genuinely paired data change the answer?

Phase 1's two views are the same text, so arguably there was nothing interesting
to predict. Here the two views are **different texts with the same meaning**:
docstring↔code (CodeSearchNet) and document↔summary (XSum). This is LLM-JEPA's
setting, run from scratch. 14 runs, both corpora, 7 arms each.

**The key methodological device — the shuffle control.** For each objective we
train a second, identical model in which the pairs are **scrambled within the
batch**: each docstring is matched with a *random other example's* code, re-drawn
every step. Everything is preserved — same texts, same compute, same loss scale —
except the correspondence. If an objective is learning the correspondence,
scrambling must destroy its benefit.

| Arm | what it contains | code | summary |
|---|---|---|---|
| pure | JEPA only | .086 | .070 |
| anchored | JEPA + MLM | .338 | .387 |
| shuffled | JEPA + MLM, **pairs scrambled** | .344 | .382 |
| llmjepa | JEPA + MLM, no SIGReg (faithful LLM-JEPA) | .346 | .418 |
| mlmonly | **MLM only, no JEPA** | .340 | .430 |
| contrastive | InfoNCE only, no MLM | **.369** | **.377** |
| con_shuffled | InfoNCE only, **pairs scrambled** | **.072** | **.107** |

Reading it by isolating each objective:

- **JEPA alone → .086.** Barely above an untrained encoder.
- **MLM alone → .340.**
- **JEPA + MLM → .338**, and with scrambled pairs **.344**. Adding the JEPA term
  changes nothing, and whether the pairs are correct changes nothing.
- **InfoNCE alone → .369**, with no token supervision at all — and scrambling the
  pairs collapses it to **.072**.

**InfoNCE** (the standard contrastive objective behind CLIP and SimCSE) asks a
*relative* question — "which of these 96 code snippets goes with this
docstring?" — using the rest of the batch as negatives. That question is
answerable even when embeddings are still random, because it needs only
comparisons. Prediction asks the model to *produce* a target vector, which
requires the target space to already be meaningful.

So the pairs carry a large, learnable signal. Contrastive learning extracts
**+.296 / +.269** of it; predictive JEPA extracts **−.006 / +.005**, i.e. none.
The contrastive arm is also what proves the test works: our setup *can* detect a
pairing effect ~50× larger when the objective can use one, so the JEPA null is a
measurement, not blindness.

Predictions for all of this were written down and committed **before** the
deciding runs (mask/THEORY.md); all four held, including one that could have
falsified the explanation.

**Second metric.** Cross-view retrieval (given a docstring, retrieve its code from
1000 candidates) agrees: pure JEPA sits at chance (.0005 vs .001 chance), while
real and scrambled pairs give the same score (.1175 vs .1155) — that retrieval
ability comes from MLM plus word overlap, not from learned correspondence.

---

## 5. Phase 3 — DLLM-JEPA reproduction (running now)

Phases 1–2 are all *from scratch*. The published gains are all on **pretrained**
models, so this phase moves there. Implemented DLLM-JEPA's method faithfully
(two mask rates .2/.7 of the same input, cosine loss to an EMA target, k-layer
predictor, on top of the diffusion SFT loss), with two compute adaptations: LoRA
instead of full finetuning, and an EMA teacher over the LoRA weights only —
equivalent to a full EMA copy since the base is frozen, but megabytes instead of
a second 16 GB model.

**Gate 1** (running): reproduce the published gain before spending compute on
anything else.

---

## 6. Why this matters — the critical questions

**"Nobody claimed from-scratch JEPA works. Why test it?"**
Fair, and on its own this phase is motivation rather than a headline. Its value
is that it built and validated the two instruments used everywhere else: the
**chance-level diagnostic** (which turns an uninterpretable loss number into a
yes/no on whether an objective is learning) and the **shuffle control**. Both are
cheap and general.

**"So what if it fails from scratch — the papers finetune."**
Correct, and that is exactly the argument for the next experiment rather than a
defence of this one. What Phase 1–2 establish is that a *reported gain is not
evidence that the pairing caused it*. Our own anchored arm looked perfectly
reasonable (.338) until the scrambled arm (.344) showed the pairing contributed
nothing whatsoever.

**"Isn't 'contrastive works on pairs' already known?"**
Yes — from pretrained initialization (SimCSE) or at CLIP scale. Its role here is
not discovery; it is the **positive control** that makes the JEPA null
interpretable. Without it, "we found no effect" is indistinguishable from "our
setup can't detect an effect."

**"Then what is the actual contribution?"**
The honest version, in order of strength:
1. A measured contrast between two objectives on identical data showing the
   pairing is usable, and that predictive latent objectives cannot use it.
2. Two reusable instruments: the chance-level test and the shuffle control.
3. A mechanism, with derivation and confirmed pre-registered predictions.
4. The dissection of data2vec's implicit anchor — the field's one from-scratch
   success — explained quantitatively (.16 → .22 → .45).

**The crucial gap, stated plainly.** LLM-JEPA and DLLM-JEPA report gains against a
no-JEPA baseline. Neither, as far as we have found, reports the *scrambled-pair*
comparison — so their gains are not yet attributed to the pairing rather than to
the auxiliary loss's regularizing effect or the predictor's extra capacity. We
have the control, built and validated. Applying it where the gains are claimed is
the decisive experiment, and it is informative either way: if the gain survives
scrambling, the published mechanism is wrong; if it vanishes, the mechanism is
confirmed and we have isolated what makes the pairing usable.

## 7. What's next

- **Now:** DLLM-JEPA Gate 1, then add the scrambled-view arm — turning a
  reproduction into an attribution test.
- **Then:** chain-of-thought as the predictor's **action** input. Both papers drop
  the action from LeCun's original formulation; for reasoning tasks the CoT is the
  natural action. If it works, reasoning is absorbed into the representations at
  training time and no CoT need be generated at test time.
- **Finishing Phase 1–2:** 3 seeds for error bars, one arm where prediction is the
  only loss term (so it differs from the contrastive arm in exactly one way), and
  retrieval reference points from BERT/MiniLM.
