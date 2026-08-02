# Where the project stands (2026-08-03)

## The question

JEPA-style training predicts one view's **embedding** from another view, instead
of predicting tokens. It works very well in vision. Does it work for text?

We tested this from scratch, at BERT scale, with controls — roughly 55 training
runs so far. Everything is measured the same way: freeze the encoder, mean-pool,
score on a 4-task MTEB slice (BERT-base scores .4825 there; a plain MLM model
trained in our setup scores .4485).

## What we found

**1. From scratch, latent prediction does not work.** Best score across a
13-run sweep was **.164**, versus .4485 for plain MLM. No setting helped: mask
ratio, mask strategy, span length, target space, EMA teacher, or SIGReg.

**2. We know exactly why, and can prove it.** A model that has learned *nothing*
still gets a small-looking loss. We derived that "know-nothing" level: for our
768-dim normalized targets it is 2/768 = **.0026**, and the training loss sat at
**.0025** for the entire run. For pooled 8-token targets the level is 1/8 =
**.125**, and the loss sat at exactly .125. The objective was never learning —
it was flat at chance the whole time. This also gives a cheap diagnostic:
compute the chance level first, and you know within minutes whether a latent
objective can possibly be learning.

**3. With a token-prediction (MLM) anchor added, learning works — but then the
JEPA part contributes nothing.** Tested across six different latent loss
families, four evaluation methods, and 4× more training. Every result sat inside
the noise band of the MLM-only baseline.

**4. Giving it genuinely paired data does not rescue it either.** We trained on
real semantic pairs — docstring↔code and document↔summary. Control: scramble the
pairs so each text is matched with the wrong partner. **Real pairs and scrambled
pairs gave the same result** (difference ±.006, and the sign flipped between the
two datasets). The model never used the correspondence.

**5. But the pairs are fine — the objective is the problem.** We trained a
contrastive (InfoNCE) model on the *same* pairs, same encoder, same budget.

| objective | effect of real vs scrambled pairs | |
|---|---|---|
| | code | summary |
| JEPA (prediction) | −.006 | +.005 |
| Contrastive | **+.296** | **+.269** |

The pairs carry a large learnable signal. Contrastive learning extracts it;
prediction extracts none of it.

## What this means

Prediction asks the model to *produce* a target embedding. At the start of
training those target embeddings are essentially random, so "move toward this
vector" is an instruction pointing at noise — and the regularizer that prevents
collapse actively destroys what little structure the targets have. Contrastive
learning asks a *relative* question instead ("which of these goes with which"),
which is well-defined even when the embeddings are still meaningless. It
bootstraps structure rather than assuming it.

This also explains the published results. LLM-JEPA and DLLM-JEPA report real
gains — but always on top of a **pretrained** model, where the embeddings are
already meaningful. Our work separates the two ingredients they always use
together, and shows the pairing alone is not what does the work.

We wrote the key predictions down *before* running the deciding experiments;
all four came out as predicted, including one that could have falsified the
explanation.

## What's next

- **Finishing the from-scratch study:** seeds for error bars, one more control
  arm, and a second evaluation metric.
- **Moving to the regime where JEPA does work** (pretrained model + finetuning):
  reproducing DLLM-JEPA, then testing our own idea — using the chain-of-thought
  as the predictor's **action** input. Both LLM-JEPA and DLLM-JEPA drop the
  action from LeCun's original JEPA formulation; for reasoning tasks the CoT is
  the natural action. If it works, reasoning gets absorbed into the
  representations during training, and no CoT needs to be generated at test time.
