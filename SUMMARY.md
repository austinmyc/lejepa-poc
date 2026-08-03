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

## 6. Where this sits in the literature

**JEPA's core claim is a from-scratch pretraining claim.** This matters, because
it means Phase 1 tests the paradigm on its own terms rather than a strawman:

| paper | setup | evaluation | headline |
|---|---|---|---|
| I-JEPA (CVPR 2023) | ViT-H/14, **random init**, ImageNet | linear probe, **frozen** features | 81.7% — beats MAE (pixel reconstruction) |
| LeJEPA (2025) | ImageNet-1k, 60+ architectures | linear eval, **frozen** backbone | ~79%; SIGReg *replaces* stop-grad/EMA/schedules |
| V-JEPA / V-JEPA 2 | from-scratch video pretraining | frozen features | — |

That is our exact protocol: pretrain from scratch, freeze, evaluate frozen
representations. And I-JEPA's headline is precisely a *comparison of pretraining
objectives* — latent prediction beats reconstruction. So Phase 1 asks whether
that transfers to language:

| | vision (published) | text (ours) |
|---|---|---|
| latent prediction | **81.7** (wins) | **.164** |
| reconstruction baseline | MAE (lower) | MLM **.4485** |

Same design, opposite outcome — a 3× gap in the other direction. LeJEPA is hit
more directly still: its claim is that SIGReg *replaces* the heuristics, validated
on 60+ models, **all vision**. In text we find SIGReg is not merely unnecessary
but actively harmful (−.14 to −.24), and we isolated the pathway — the damage
travels through the whitened target geometry, not the regularizer's gradient.

**The candidate explanation for the asymmetry.** In vision, pixels are not
abstractions: there is a large gap between raw input and semantic content, and
that gap is what latent prediction exploits. In text, tokens are *already*
abstractions — a word carries meaning — so token prediction already operates at
the semantic level and there is no comparable gap to exploit.

**What the text-JEPA papers actually did** (checked against the papers):

| | LLM-JEPA (2509.14252) | DLLM-JEPA (2606.00091) |
|---|---|---|
| finetuning gains | ✓ (+14.17pp NL-RX) | ✓ (+18.7pp GSM8K, LLaDA-8B) |
| **from-scratch pretraining** | **✓ — Llama-3.2-1B from random init on NL-RX-SYNTH: 54.38 → 60.59 (p = 2.9e-4)** | ✗ |
| shuffled / mismatched-pair control | **✗ none reported** | **✗ none reported** |
| ablations reported | LoRA rank | λ, predictor depth, mask rates |

So LLM-JEPA *does* report a from-scratch pretraining gain — which superficially
contradicts our null. Two things reconcile them, and both are the gap:

1. **Task-alignment confound.** Their from-scratch experiment pretrains on
   NL-RX-SYNTH and evaluates on NL-RX-SYNTH accuracy — where the task is
   literally "map natural language → regex" and the JEPA views are literally
   (natural language, regex). The JEPA loss is therefore a soft form of the
   downstream task itself: auxiliary task supervision, not general
   representation learning. Our setting (general corpora, frozen general-purpose
   embeddings) has no such alignment — and there the effect is zero.
2. **No attribution control.** Neither paper reports what happens when the pairs
   are scrambled. So none of these gains — finetuning or pretraining — has yet
   been attributed to the *semantic correspondence* rather than to the auxiliary
   loss's regularizing effect or the predictor's added capacity.

Our own data shows why point 2 is not pedantry: the anchored arm looked entirely
reasonable at **.338** until the scrambled arm came in at **.344**.

## 7. The gap we can fill

> Published JEPA-for-text gains are (a) never attributed via a pairing control,
> and (b) measured on tasks the pairs themselves define. Whether latent
> prediction improves **general-purpose** text representations — the claim vision
> JEPA makes and validates by linear probe — has not been tested. We tested it,
> and it does not.

Three contributions follow, in order of strength:

1. **The transfer failure, with a mechanism.** Vision's central JEPA result does
   not carry to text; the chance-floor derivation says exactly why, and the
   token-vs-pixel abstraction argument says why it should have been expected.
2. **An attribution result.** Contrastive extracts +.296/+.269 from the same
   pairs that predictive JEPA extracts −.006/+.005 from — so the signal is there
   and the objective is what fails. The shuffle control and chance-level test are
   reusable instruments the field currently lacks.
3. **data2vec's implicit anchor dissected** — the one published from-scratch text
   success, explained quantitatively (.16 → .22 → .45).

**The decisive next experiment** is now precisely defined: reproduce LLM-JEPA's
own from-scratch pretraining setting (Llama-3.2-1B, random init, NL-RX-SYNTH) and
add the scrambled-pair arm. It is cheap — 1B parameters, 10k examples — and
informative either way:

- gain **survives** scrambling → the published mechanism attribution is wrong;
  the benefit is regularization, not the pairing.
- gain **vanishes** → the pairing is real but only under task-aligned evaluation,
  which together with our general-representation null gives the complete picture:
  latent prediction on pairs is task-specific auxiliary supervision, not a
  general pretraining objective.

## 8. What's next

- **Immediate:** the LLM-JEPA from-scratch reproduction **plus its missing
  scrambled-pair control** — the experiment above.
- **Running:** DLLM-JEPA Gate 1, then the same scrambled-view arm, turning a
  reproduction into an attribution test.
- **Then:** chain-of-thought as the predictor's **action** input. Both papers drop
  the action from LeCun's original formulation; for reasoning tasks the CoT is the
  natural action. If it works, reasoning is absorbed into the representations at
  training time and no CoT need be generated at test time.
- **Finishing Phase 1–2:** 3 seeds for error bars, one arm where prediction is the
  only loss term (so it differs from the contrastive arm in exactly one way), and
  retrieval reference points from BERT/MiniLM.
