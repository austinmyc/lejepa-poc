# Paper Story — JEPA-first framing
# "When Does Latent Prediction Have Signal in Language?"

(Framing doc; numbers live in RESULTS.md, plan in EXPERIMENT_PLAN.md.)

## JEPA-CENTRIC ARC (2026-07-08 reframe — Austin: keep the story about JEPA)
Central question: under what conditions does latent prediction carry learning
signal in language? Every section is a JEPA component:
- VIEWS: masked-vs-clean same-text views lack information asymmetry; no view
  dial (ratio/strategy/span-len) creates it. Condition 1: JEPA needs a VIEW GAP
  (cf. LLM-JEPA's text<->code). Masking alone doesn't make one in text.
- TARGETS: hierarchy measured — random -> floor 2/P; SIGReg-whitened -> floor
  1/L (what stabilizes can sterilize); CE-grounded -> learnable but redundant.
- COLLAPSE MECHANISMS (the LeJEPA question in text): EMA fails (rank collapse,
  scale explosion); SIGReg prevents collapse via target-poisoning whitening;
  stop-grad+predictor+anchor stable. Vision hierarchy INVERTS in text.
- PREDICTOR: mean-collapse under point regression; distributional (diffusion)
  head fixes the loss pathology, still transfers nothing -> failure is
  informational, not architectural.
- THESIS — THE ABSTRACTION GAP: JEPA pays when representations are far more
  abstract than inputs (pixels->semantics). Tokens ARE abstractions; text has
  no gap from scratch. Predicts where literature gains live: pretrained
  backbones restore the abstraction gap; cross-modal views restore the view
  gap; small-data adaptation changes signal economics.
- PRESCRIPTIVE: two conditions (view gap + abstraction gap) + chance-floor
  diagnostics to test any latent objective for signal before burning compute.
BERT-parity = ONE setup sentence (competence certificate only, never a
contribution). Mean-pool bottleneck + geometry no-free-lunch -> appendix.

## Abstract-shaped pitch

First systematic study of JEPA for text FROM SCRATCH (~40 controlled runs).
Definitive mechanistic negative: latent prediction cannot bootstrap a text
encoder, and once a token-CE anchor exists it adds nothing — at any
granularity, target space, loss family, readout, or training stage tested.
Three quantified mechanisms: chance floor 2/P (normalized point regression),
chance floor ~1/L (pooled whitened targets), whitened-target toxicity
(pathway isolated via grad-gating: target geometry, not regularizer gradient).
Surviving thesis: text begins at the semantic level — token CE claims the
signal first; the residual is redundant or irreducible. Explains the
literature (data2vec's machinery, LLM-JEPA/DLLM-JEPA's pretrained backbones
and cross-modal views). Artifacts: chance-floor diagnostics, BERT-quality
recipe at 1/4 budget, mean-pool bottleneck (+5-6 pts attention pooler),
no-free-geometric-lunch table.

## Contributions
1. First systematic from-scratch text-JEPA study (reference paper timing:
   3 text-JEPA papers in last 9 months, none from scratch).
2. Two computable chance floors as diagnostics for dead latent objectives.
3. Whitened-target mechanism + alpha-gating pathway ablation.
4. Readout-independent redundancy: 6 loss families x 4 readout families x
   convergence curves — no escape hatch.
5. Literature reconciliation: published gains live exactly where the
   mechanism predicts (pretrained init, cross-modal views, small-data FT).
6. Artifacts: reference point (.4485 @ 30k ~= 93% of BERT-base frozen score at ~0.4% compute; seed sigma .01 — frames frozen-eval saturation, not a recipe claim);
   mean-pool bottleneck; geometry no-free-lunch.

## Section -> evidence map
- S3 Failure I (no anchor): RQ1 15-run block, diagnose.py forensics.
- S4 Anchor + attach point (+.07), collapse-mechanism table (round 1).
- S5 Failure II (whitened targets): waves 1-2 floors, splitspace alpha=0.
- S6 Redundancy: wave 3 (enc-space tie), wave 5 (diffusion tie, l_diff
  flatline), probe suite + glued 10k/20k/30k curves.
- S7 Scale: 120k pair (PENDING — ctrl_random_120k unlaunched).
- S8 Geometry no-free-lunch: whiten/manfit/raw post-hoc table.
- S9 Where JEPA's signal lives: reconciliation; optional constructive ending
  = small-data fine-tuning regularizer test (LLM-JEPA regime in miniature).

## Reviewer attacks -> answers
weak baseline -> BERT-quality ctrl + error bars everywhere.
wrong metric -> 4 readout families incl. I-JEPA's own (few-shot probes).
wrong scale -> BERT-budget-matched 120k pair.
didn't try X -> 6 loss families incl. LatentLM diffusion head; gap = data2vec
layer-averaged targets (one cheap run, wave-6 checkbox).
negative result -> anatomy-with-mechanisms + artifacts; timed to the wave.

## Venue & endings
ICLR / EMNLP main (TMLR alternative). Endings: (a) 120k tie -> scale-robust
inertness; (b) 120k gap -> scale-interaction positive; (c) FT-regularizer
positive -> "inert in pretraining, valuable at the adaptation margin".

## Blocking items
ctrl_random_120k (S7 does not exist without it); contraction arms (Austin's
manifold contribution); optional: layer-avg targets checkbox, FT-regularizer
experiment.
