# LeJEPA-Text: Methods Note

## Core design decisions and rationale

---

### 1. Why predict in embedding space?

Next-token prediction (NTP) forces the model's representation space to be organized around token co-occurrence statistics. The embedding geometry is shaped by what tokens tend to follow other tokens — not by what chunks of text *mean*. JEPA-style pretraining sidesteps this by having the model predict the *embedding* of the next chunk, not the tokens inside it. This frees the encoder to organize its representation space around semantic content.

---

### 2. Architecture overview

**Encoder** — bidirectional transformer, 12 layers, 768 hidden dim, 12 heads (~110M params). Each chunk (64 tokens) is encoded independently with full bidirectional attention, then mean-pooled to produce a single chunk embedding vector. Bidirectionality is a deliberate choice: it gives the encoder access to the full context within each chunk before producing a representation. The transformer block is borrowed from nanochat's `gpt.py` with the causal mask removed.

**Projection MLP** — 2-layer MLP (768 → 2048 → 768) with LayerNorm at the output. Maps encoder outputs into an approximately isotropic space before feeding the predictor. See §4 for the motivation.

**Predictor** — causal transformer, 4 layers, 768 hidden dim. Takes a sequence of projected chunk embeddings as input and autoregressively predicts the projected embedding of the next chunk. Causal attention across chunks preserves the autoregressive structure.

---

### 3. Training objective

```
Loss = (1 - λ) · MSE(pred_emb[t], target_proj[t])  +  λ · SIGReg(Z)
```

- `pred_emb[t]` — the predictor's output for position t
- `target_proj[t]` — the projection MLP applied to the target chunk's encoder output (attached gradient for MSE path; detached for SIGReg path — see §4)
- `Z` — the full batch of projected chunk embeddings
- `λ = 0.1` (default; ablated at 0.01 and 0.5)

No EMA teacher, no stop-gradient on the target for MSE, no masked token prediction. Optimizer: **Muon** for both JEPA and GPT-2 baseline (borrowed from nanochat `optim.py`), ensuring the comparison isolates the pretraining objective and not the optimizer.

---

### 4. Projection MLP and the anisotropy problem

Text embeddings produced by transformers are intrinsically anisotropic — they concentrate in a narrow cone of the embedding space rather than distributing uniformly. This is well-documented (Ethayarajh 2019) and creates a poor prior for the predictor. If the input distribution is highly non-uniform, the predictor can exploit that structure to achieve low MSE without learning anything about semantic content — it simply predicts toward the high-density region.

Adding a projection MLP between the encoder and predictor addresses this by mapping encoder outputs into a more isotropic space before the predictor sees them. SIGReg on the projected embeddings pushes the batch covariance toward a scaled identity, maintaining isotropy throughout training.

#### The disentanglement problem

A naive implementation would backpropagate the SIGReg loss through the projection MLP *and* into the encoder. This reintroduces the problem: the encoder gets shaped by the isotropy pressure, distorting whatever semantic geometry it is learning. There is no architectural guarantee that the encoder specializes in semantics and the projection specializes in geometry correction — the optimizer will find whatever combination minimizes the combined loss.

#### Solution: stop-gradient on the SIGReg path

We apply a stop-gradient at the encoder output for the SIGReg computation only:

```python
encoder_out = encoder(chunk)

# SIGReg path — projection sees a detached encoder output
proj_detached = projection(encoder_out.detach())
sigreg_loss = SIGReg(proj_detached)

# MSE path — full graph, gradients flow through projection into encoder
proj_attached = projection(encoder_out)
pred = predictor(proj_attached_context_sequence)
mse_loss = MSE(pred, proj_attached_target)

loss = (1 - λ) * mse_loss + λ * sigreg_loss
```

**What each loss trains:**

| Loss | Trains | Does not train |
|------|--------|----------------|
| MSE (attached) | encoder, projection, predictor | — |
| SIGReg (detached) | projection only | encoder |

This gives clean gradient responsibilities:
- The encoder is trained purely by the prediction signal. It learns whatever representation makes next-chunk prediction easy in the projected space.
- The projection is trained by both MSE (to be a useful input for the predictor) and SIGReg (to maintain isotropy). It absorbs the geometry correction without imposing that correction on the encoder.

The projection has two gradient signals that partially conflict — MSE wants it to preserve information useful for prediction, SIGReg wants it to expand toward a uniform distribution. This tension is analogous to the expander network in VICReg and appears to be beneficial: the projection learns a transformation that is both isotropic and semantically structured.

---

### 5. What the predictor does

The predictor operates at chunk granularity: given a sequence of projected context chunk embeddings, it autoregressively predicts the projected embedding of the next chunk. Its role is not eliminated after pretraining.

**At eval time, use depends on the task:**

- **Representation tasks** (GLUE, STS-B, probing) — use the raw encoder output. The projection and predictor are not involved. The encoder output is what we claim has rich semantic structure.
- **Chunk-level sequential prediction** — keep both projection and predictor. Given context chunks, predict the next chunk's embedding and evaluate against ground truth (e.g., nearest-neighbor ranking, cosine similarity to target).
- **Generative fine-tuning (§3.2)** — attach a token-level causal LM head to the encoder output and fine-tune with NTP. The predictor operates at chunk granularity and is incompatible with token-level generation without a bridging decoder; it is not used in this setting.

---

### 6. Relationship to prior work

| Method | Teacher/EMA | Negatives | Prediction space | Isotropy mechanism | Optimizer |
|--------|------------|-----------|------------------|--------------------|-----------|
| I-JEPA | EMA teacher | no | embedding | implicit (EMA stability) | AdamW |
| SimCLR | no | yes | embedding | contrastive uniformity | LARS |
| VICReg | no | no | embedding | explicit variance term | AdamW |
| BERT/MLM | no | no | token space | none | AdamW |
| GPT-2 baseline | no | no | token space | none | **Muon** |
| **LeJEPA-Text** | **no** | **no** | **embedding** | **SIGReg + stop-grad projection** | **Muon** |

The key novelty is combining JEPA-style embedding prediction with a stop-gradient projection that absorbs isotropy correction without distorting the encoder, and doing so without EMA or contrastive negatives.

---

### 7. Open questions

- Does the projection MLP's dual gradient signal (MSE + SIGReg) create instability at high λ? Monitor projection weight norms during training.
- Should the projection be shared between context and target paths, or have separate weights? Shared is simpler and standard (SimCLR); separate might allow the target projection to specialize differently.
- For downstream representation evaluation, is the raw encoder output always better than the projected output? This should be verified empirically — if projected outputs are better, it suggests the isotropy correction is semantically useful, not just a training convenience.
