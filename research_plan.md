# LeJEPA-Text: Joint Embedding Predictive Architecture for Language Pretraining

## Research Plan & Paper Outline

> **Scope:** Paper 1 — chunk embedding pretraining quality. Generative fine-tuning (LM head) is deferred to Paper 2.
> **Timeline:** 12 weeks (3 months).

---

## 1. Core Claim

JEPA-style pretraining (predict in embedding space) learns richer semantic chunk representations than next-token prediction (NTP) at matched compute, without requiring masked autoencoders, contrastive negatives, or EMA teacher networks.

**Falsifiable:** A LeJEPA-Text encoder at GPT-2 small scale (12 layers, 768 hidden dim, ~110M params), pretrained on OpenWebText for 10B tokens, should outperform a GPT-2 small baseline trained identically on the same data on semantic probing tasks and GLUE fine-tuning.

---

## 2. Method

### 2.1 Architecture

| Component | Design |
|-----------|--------|
| **Chunk encoder** | 12-layer transformer, 768 hidden, 12 heads, **bidirectional** → mean-pool over chunk tokens → 1 vector per chunk (~110M params) |
| **Projection MLP** | 2-layer MLP (768 → 2048 → 768) + LayerNorm; maps encoder output to isotropic space before predictor |
| **Autoregressive predictor** | 4-layer transformer, 768 hidden, causal attention over chunk embeddings (~25M params) |
| **Regularizer** | SIGReg on projected chunk embeddings (stop-gradient from encoder — see §2.2) |
| **Total params** | ~140M |
| **Baseline (NTP)** | NanoGPT, GPT-2 small spec: 12-layer, 768 hidden, 12 heads, causal, 124M params |

The encoder uses bidirectional attention within each chunk (full context before pooling) while the predictor uses causal attention across chunks. The comparison to GPT-2 is scale-matched, not architecture-identical — this is explicit in the paper.

### 2.2 Training Objective

```
Loss = (1 - λ) · MSE(pred_emb[t], target_proj[t])  +  λ · SIGReg(Z)
```

where `target_proj[t]` uses an **attached** encoder output for the MSE path, and a **stop-gradient** detached encoder output for the SIGReg path:

```python
proj_detached = projection(encoder_out.detach())   # SIGReg only trains projection
proj_attached  = projection(encoder_out)            # MSE trains encoder + projection + predictor
sigreg_loss = SIGReg(proj_detached)
mse_loss    = MSE(predictor(context_projs), proj_attached_target)
```

**Why stop-gradient:** Text embeddings are intrinsically anisotropic. Without a corrective step the predictor can exploit low-rank input structure to achieve low MSE without learning semantics. A projection MLP maps encoder outputs to isotropic space before the predictor. The stop-gradient ensures SIGReg only shapes the projection — not the encoder — so the encoder is free to develop semantic geometry driven purely by the prediction signal.

- No EMA, no contrastive negatives, no masked token prediction
- Chunk size: 64 tokens (ablated at 32, 128)
- λ: 0.1 (ablated at 0.01, 0.5)

### 2.3 GPT-2 Baseline: Train from Scratch

**Use nanochat (`--depth=12`) to train GPT-2 small from scratch on OpenWebText.** Do not use OpenAI's pretrained weights — that model was trained on proprietary WebText (~40B tokens) with unknown hyperparameters. Training from scratch takes ~8–12 hours on 4×L20 for 10B tokens, which is acceptable overhead for a clean controlled comparison.

Both models see identical data in identical order. Batch size, learning rate schedule, and optimizer (**Muon**) are matched. Using Muon for both ensures the comparison isolates the pretraining objective, not the optimizer.

### 2.4 Bidirectional vs. Causal Encoder

The encoder is bidirectional — this is a deliberate choice and a difference from GPT-2 that must be acknowledged. The paper frames this as: "we use the same compute budget and parameter count as GPT-2 small, with the architectural modification necessary for chunk-level encoding (bidirectionality within chunk)." The predictor remains causal, preserving the autoregressive structure across chunks.

A causal encoder variant (last-token pooling, GPT-2 identical architecture) is run as an ablation to isolate this choice.

### 2.5 What makes this novel vs. prior work

- **vs. BERT/MLM:** Predicts in embedding space, not token space — no reconstruction pressure, no masking
- **vs. I-JEPA (image):** No EMA teacher; SIGReg alone prevents collapse
- **vs. contrastive methods (SimCSE, etc.):** No negative pairs, no data augmentation policy
- **vs. autoregressive LMs (GPT-2):** Embedding space is free to organize semantically rather than being shaped by next-token distributions

---

## 3. Experimental Design

### 3.1 Primary Comparison: JEPA vs. GPT-2 at Matched Scale

**Setup:**
- JEPA encoder: 12L/768H bidirectional, ~110M + 25M predictor
- GPT-2 baseline: NanoGPT, 12L/768H causal, 124M, trained from scratch
- Pretraining corpus: OpenWebText (~40GB, ~10B tokens), identical data order
- Compute budget: 4×L20 (48GB), ~8–12h per run
- FLOPs matched as closely as possible; any gap reported explicitly

**Downstream evaluation — fine-tune both:**

| Task | Metric | Why |
|------|--------|-----|
| STS-B (semantic similarity) | Spearman ρ | Tests embedding geometry |
| SST-2 (sentiment) | Accuracy | Simple transfer |
| MNLI (NLI) | Accuracy | Compositional semantics |
| SQuAD v1.1 (QA) | F1/EM | Span-level understanding |
| GLUE aggregate | Average | Standard benchmark |

Evaluation protocol: (1) freeze encoder, linear probe; (2) full fine-tune. Report both. GPT-2 baseline uses mean-pooled last-layer hidden states for (1).

---

### 3.3 Representation Quality: Probing Studies

Do chunk embeddings encode semantically richer structure than GPT-2 hidden states?

**Probes:**
1. **Syntactic depth probe** — predict parse tree depth from chunk embedding
2. **Entity type probe** — predict NER label of most salient entity in chunk
3. **Coreference probe** — do embeddings of coreferent chunks cluster?
4. **Isotropy analysis** — measure effective rank (JEPA should be less anisotropic)

Compare JEPA chunk embeddings vs. GPT-2 last-layer hidden states (mean-pooled over same 64-token spans).

---

### 3.4 Ablations

| Variable | Values tested | Metric | Priority |
|----------|--------------|--------|----------|
| Encoder direction | **bidirectional**, causal (last-token) | GLUE + STS-B | **week 1** |
| Projection MLP | **with stop-grad**, with full grad, no projection | GLUE + collapse rate | **week 1** |
| Chunk size | 32, **64**, 128 tokens | GLUE + STS-B | week 4 |
| λ (SIGReg weight) | 0.01, **0.1**, 0.5 | collapse rate + GLUE | week 4 |
| Predictor depth | 2, **4**, 6 layers | GLUE + STS-B | week 5 |
| Regularizer | **SIGReg**, VICReg, none | collapse rate + GLUE | week 5 |

Bold = default configuration. The projection ablation (stop-grad vs full-grad vs none) is a core contribution and must run early.

---

### 3.5 Collapse Diagnostics

Track during training (log every 500 steps):
- **Embedding rank** (effective dimensionality via explained variance)
- **Dead dimensions** (% of embedding dims with near-zero variance)
- **Cosine similarity distribution** across batch
- **Prediction loss curve** — does it decrease meaningfully or plateau?

If rank collapses before step 5k, halt and adjust λ before wasting a full run.

---

## 4. Paper Structure

### Title (candidate)
*"LeJEPA-Text: Pretraining Language Models by Predicting in Embedding Space"*

### Abstract (key beats)
1. Problem: NTP shapes representations toward token distributions, not semantic structure
2. Method: predict chunk embeddings autoregressively at GPT-2 scale; SIGReg prevents collapse; no EMA, no teacher
3. Result: JEPA matches/beats GPT-2 on GLUE fine-tune; JEPA pretraining → NTP finetune reaches lower perplexity faster
4. Significance: embedding-space pretraining transfers to generative objectives, suggesting richer weight initialization

---

### Section 1 — Introduction
- NTP as the dominant pretraining objective: power and limitations
- The semantic gap: LMs predict token distributions; JEPA predicts meaning
- JEPA in vision: LeCun's motivation for embedding-space prediction
- Gap: no clean JEPA-style pretraining for text at LM scale without EMA complexity
- Contributions:
  1. LeJEPA-Text: JEPA pretraining for language at GPT-2 scale, no EMA, no teacher
  2. Controlled comparison against from-scratch GPT-2 on identical data
  3. Transfer result: JEPA pretrain → NTP finetune outperforms NTP throughout
  4. Probing analysis: richer representation structure in JEPA embeddings

### Section 2 — Background & Related Work
- Self-supervised learning for text: BERT, RoBERTa, GPT family
- Contrastive methods: SimCSE, CLIP-text
- JEPA family: I-JEPA, V-JEPA, A-JEPA
- Collapse prevention: VICReg, Barlow Twins, SIGReg
- Chunk-level representations: SpanBERT, Sentence-BERT, paragraph embeddings

### Section 3 — Method
- Chunking procedure and its limitations (fixed spans)
- Encoder architecture (bidirectionality choice justified)
- Autoregressive predictor
- SIGReg formulation
- Full loss, hyperparameters, training details
- GPT-2 baseline implementation (NanoGPT, from-scratch)

### Section 4 — Experiments
- 4.1: GLUE fine-tune comparison (§3.1 — primary result)
- 4.2: Probing study (§3.3)
- 4.3: Ablations (§3.4)

### Section 5 — Analysis
- When does JEPA beat GPT-2? (task type breakdown)
- Failure modes: tasks where GPT-2 wins and why
- Collapse behavior over training
- Effect of stop-gradient projection on encoder isotropy vs. downstream quality

### Section 6 — Discussion & Limitations
- Fixed chunk boundaries: a hard inductive bias (future: learned chunking)
- Bidirectional encoder vs. causal — tradeoffs for downstream generation
- Comparison to masked methods (BERT) is not apples-to-apples; left for future work
- Future: generative fine-tuning (LM head, Paper 2), hierarchical JEPA, scaling beyond 125M

### Section 7 — Conclusion

---

## 5. Repository & Compute Setup

### 5.1 Repository structure

Two separate repos — do not merge them:

```
lejepa-poc/          ← this repo — all JEPA code
  model.py           encoder, projection MLP, predictor
  train.py           JEPA training loop (written from scratch)
  data.py            data pipeline (adapted from nanochat dataloader.py)
  config.py          hyperparameters
  eval.py            GLUE / STS-B / probing evaluation
  optim.py           Muon optimizer (copied from nanochat)
  smoke_test.py      quick sanity check (already exists)

nanochat/            ← separate clone, run as-is for GPT-2 baseline
  --depth=12         auto-configures GPT-2 small spec
```

**From nanochat, copy into lejepa-poc:**
- `nanochat/optim.py` — Muon optimizer + parameter grouping (AdamW for embeddings/LN, Muon for linear layers)
- `nanochat/dataloader.py` — tokenized distributed data loader
- `nanochat/gpt.py` — transformer block; remove causal mask for the bidirectional encoder

Do not copy `base_train.py` — the JEPA training loop is too different to fork cleanly.

### 5.2 Optimizer strategy

| Stage | JEPA optimizer | GPT-2 optimizer | Data |
|-------|---------------|-----------------|------|
| Local validation (Mac) | AdamW | AdamW | tiny-shakespeare |
| L20 full run | Muon | Muon | OpenWebText |

Muon uses `torch.distributed.all_reduce` internally and cannot run on CPU/MPS without a distributed process group. AdamW is used locally purely to validate correctness — the optimizer choice does not affect what the local run proves. Both models switch to Muon on the L20 so the final comparison isolates the pretraining objective only.

### 5.3 Compute strategy

Staged approach — validate cheaply before requesting expensive compute:

**Stage 0 — Mac local validation (week 1, free)**
- Train tiny JEPA + tiny GPT-2 baseline on Shakespeare with AdamW
- Goal: confirm training loop, eval pipeline, and stop-gradient all work end-to-end before touching the server
- Both models use AdamW here — optimizer correctness is not what's being tested

**Stage 1 — L20 validation (weeks 1-2, free)**
- Train a 6L/512H JEPA model (~30M params) for 500M tokens on the lab L20
- Goal: confirm training loop converges, collapse doesn't happen, stop-grad projection works
- Run early ablations (encoder direction, projection variants) at this scale — they're cheap and de-risk the main run
- Validate GPT-2 baseline at the same small scale for a quick sanity comparison

**Stage 2 — Full run (weeks 3-4, request from supervisor)**
- Request 4×H100 or 4×A100 once Stage 1 confirms the architecture is stable
- Main JEPA run: 12L/768H, 10B tokens (~8-12h on 4×H100)
- GPT-2 baseline: nanochat `--depth=12`, same data, same compute budget
- Bring Stage 1 results + research plan to supervisor as justification for the compute request

**Stage 3 — Ablations (weeks 5-6, cloud)**
- Ablation runs are shorter (2-3B tokens each) — can run on spot instances to save cost
- Estimated total ablation compute: ~20-30 GPU-hours

## 6. Implementation Checklist (ordered by dependency)

- [ ] Data pipeline: OpenWebText → tokenize (GPT-2 BPE) → shuffle, shard, chunk into 64-token spans
- [ ] **GPT-2 baseline: clone nanochat, run with `--depth=12` on OpenWebText from scratch**
- [ ] JEPA encoder: 12L/768H bidirectional transformer (borrow `gpt.py` block from nanochat, remove causal mask), mean-pool → chunk embedding
- [ ] Projection MLP: 768→2048→768 + LayerNorm; wire up stop-gradient for SIGReg path
- [ ] JEPA predictor: 4L/768H causal transformer over projected chunk sequence
- [ ] SIGReg: install `lejepa` library, verify SIGReg input shape and API; confirm it receives detached projections
- [ ] Training loop: bf16 mixed precision, gradient checkpointing, DeepSpeed ZeRO-2, **Muon optimizer** (borrow from nanochat `optim.py`)
- [ ] Collapse monitor: log encoder embedding rank + projection embedding rank + cosine sim every 500 steps to W&B
- [ ] Evaluation: HuggingFace for GLUE/STS-B, SentEval for probing
- [ ] **Ablation (week 1): bidirectional vs. causal encoder**
- [ ] **Ablation (week 1): projection stop-grad vs. full-grad vs. no projection**
- [ ] Ablation grid (week 4-5): chunk_size × predictor_depth × lambda × regularizer

## 7. Timeline (12 weeks)

> Weeks 1-2 on lab L20. Weeks 3+ on requested cloud compute (bring Stage 1 results to supervisor).

| Week | Milestone | Compute |
|------|-----------|---------|
| 1 | Copy nanochat files; data pipeline; encoder + projection + predictor implementation | L20 |
| 2 | Small-scale validation (6L/512H, 500M tokens); early ablations: encoder direction + projection variants | L20 |
| 3 | Request cloud compute. Main JEPA run starts (12L/768H, 10B tokens) | H100/A100 |
| 4 | GPT-2 baseline run (nanochat `--depth=12`, 10B tokens); GLUE + STS-B eval on both | H100/A100 |
| 5 | Ablations: chunk size × λ × predictor depth | Cloud spot |
| 6 | Ablations: regularizer variants; finalize ablation grid | Cloud spot |
| 7 | Probing studies: syntactic depth, entity type, coreference, isotropy analysis | L20 |
| 8 | Analysis: task-type breakdown, failure modes, collapse curves, learning curves | L20 |
| 9 | Figures + paper skeleton | — |
| 10 | Full draft | — |
| 11 | Revision + internal review | — |
| 12 | Polish + submit | — |

## 8. Paper 2 (deferred)

Transfer to generative fine-tuning — deferred from Paper 1 to keep scope manageable.

**Core question:** does JEPA pretraining serve as a better initialization for token-level language modeling than NTP pretraining itself?

**Protocol:** pretrain JEPA encoder (Paper 1 checkpoint) → swap in causal LM head → fine-tune with NTP at 100M / 500M / 1B tokens → compare perplexity on OpenWebText + WikiText-103 against GPT-2 baseline (matched total compute).

This requires ~4 extra weeks of training runs and is cleanly separable from Paper 1's results. Paper 1's embedding quality findings motivate Paper 2's hypothesis.

---

## 8. Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Collapse despite SIGReg | Monitor encoder + projection rank every 500 steps; halt + increase λ if rank < 10 by step 5k |
| Stop-grad projection causes encoder to ignore isotropy entirely | Check encoder embedding rank directly; if severely collapsed, add a mild (low-weight) SIGReg on encoder output as a floor |
| JEPA underperforms GPT-2 on all tasks | Pivot to honest analysis paper: why does JEPA not transfer to text? That's publishable too |
| Bidirectionality confounds the comparison | Run causal encoder ablation week 1; report both; be transparent in paper |
| NanoGPT baseline diverges or underperforms reference GPT-2 | Validate against known OpenWebText perplexity numbers before proceeding |
| Ablation grid too large for timeline | Prioritize encoder direction + projection ablation (week 1-2); treat remaining ablations as supporting evidence, not blockers |
