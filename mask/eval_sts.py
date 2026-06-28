"""
STS-B evaluation for a trained mask/ checkpoint — a cheap, quantitative quality
metric for iteration (one of the MTEB tasks).

Encodes each sentence (encoder → mean-pool, the standard SSL backbone read-out),
takes cosine similarity of each pair, and reports Spearman correlation against
the human similarity ratings on STS-B validation (~1500 pairs).

    python mask/eval_sts.py <checkpoint.pt> [--readout encoder|proj]

A random-init model scores ~0; BERT-base mean-pool (no fine-tune) is ~0.45–0.6.
Use the delta vs a random-init checkpoint as the real signal.
"""

import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from config import Config          # noqa: F401 (needed to unpickle cfg in ckpt)
from model import LeJEPAText


def spearman(a, b):
    """Spearman rho with average-rank tie handling, no scipy dependency."""
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)
    except Exception:
        pass

    def avg_rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        ranks = [0.0] * len(x)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            r = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = r
            i = j + 1
        return ranks

    ra, rb = avg_rank(a), avg_rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    vb = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return cov / (va * vb + 1e-12)


@torch.no_grad()
def embed(model, tok, sentences, device, seq_len, readout):
    """Encoder mean-pool embedding per sentence (variable length, no padding)."""
    out = []
    for s in sentences:
        ids = tok.encode(s)[:seq_len] or [tok.eos_token_id]
        x = torch.tensor([ids], device=device)
        h = model.encoder(x)                       # (1, L, d_model)
        if readout == "proj":
            h = model.proj(h)                      # (1, L, d_proj)
        out.append(h.mean(dim=1).squeeze(0))       # mean-pool over tokens
    return torch.stack(out)


def evaluate(ckpt_path, readout="encoder"):
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = LeJEPAText(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    # MTEB-aligned STS-B (proper namespace/name repo). Cols: sentence1/2, score.
    ds = load_dataset("mteb/stsbenchmark-sts", split="test")

    e1 = embed(model, tok, ds["sentence1"], device, cfg.seq_len, readout)
    e2 = embed(model, tok, ds["sentence2"], device, cfg.seq_len, readout)
    cos = F.cosine_similarity(e1, e2).cpu().tolist()

    rho = spearman(cos, ds["score"])
    print(f"STS-B Spearman ({readout} readout, step {ckpt['step']}): "
          f"{rho:.4f}  on {len(ds['score'])} pairs")
    return rho


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--readout", choices=["encoder", "proj"], default="encoder")
    args = ap.parse_args()
    evaluate(args.checkpoint, args.readout)
