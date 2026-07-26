"""
Cross-view retrieval eval for frozen paired-JEPA checkpoints — with HF baselines.

Why this and not GSM8K / NL-Regex / Spider (LLM-JEPA's tasks): those are
GENERATIVE and need a pretrained decoder — impossible on our from-scratch 768-d
bidirectional ENCODER checkpoints. The encoder-appropriate analogue that ALSO
directly tests the paired hypothesis is bitext retrieval: freeze the encoder,
embed held-out (view_A, view_B) pairs, and ask whether A retrieves its true B.

The sharp reads:
    anchored > shuffled            → real pairs taught the A↔B mapping,
    anchored ≈ shuffled            → the pairing carried no retrievable signal.
    ours     vs base BERT (--hf-model bert-base-uncased)
                                   → does a PRETRAINED encoder (abstraction gap)
                                     beat our from-scratch ones? (cell-2 preview)
    ours     vs all-MiniLM-L6-v2   → a real (contrastively-trained) retriever =
                                     the ceiling, to calibrate how weak mean-pool is.

Both our checkpoints and the HF baselines embed the SAME streamed pairs, so the
comparison is apples-to-apples. Mean-pool over non-pad tokens for both (the MTEB
readout; matches eval_baselines.MeanPoolEncoder).

Usage:
    python mask/eval_retrieval.py --ckpt checkpoints_mask/paired_..._final.pt --pair-source code
    python mask/eval_retrieval.py --hf-model bert-base-uncased --pair-source summary
"""

import argparse
import os
import torch
import torch.nn.functional as F

from config import Config
from model  import LeJEPAText, _masked_mean
from data   import PAIR_PRESETS


# ── data ────────────────────────────────────────────────────────────────────

def pick_split(repo, config, requested):
    """Prefer a held-out split; fall back to train (with a warning) if that's all."""
    from datasets import get_dataset_split_names
    try:
        splits = get_dataset_split_names(repo, config) if config else get_dataset_split_names(repo)
    except Exception:
        return requested or "train"
    if requested and requested in splits:
        return requested
    for s in ("test", "validation", "valid", "dev"):
        if s in splits:
            return s
    print(f"  ! no held-out split in {splits}; using '{splits[0]}' (OVERLAPS training data)")
    return splits[0]


def stream_raw_pairs(pair_source, n, requested_split):
    """Stream up to n (text_a, text_b) string pairs from a PAIR_PRESETS corpus."""
    from datasets import load_dataset
    repo, col_a, col_b, _, config = PAIR_PRESETS[pair_source]
    split = pick_split(repo, config, requested_split)
    ds = (load_dataset(repo, config, split=split, streaming=True) if config
          else load_dataset(repo, split=split, streaming=True))
    A, B = [], []
    for ex in ds:
        a, b = ex.get(col_a), ex.get(col_b)
        if not a or not b:
            continue
        A.append(a); B.append(b)
        if len(A) >= n:
            break
    return A, B, split


# ── embedding backends (both mean-pool over non-pad, L2-normalized) ─────────

@torch.no_grad()
def embed_ours(ckpt_path, texts_ab, device, seq_len, bs=128):
    from transformers import GPT2TokenizerFast
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)  # pickles Config
    cfg = ckpt["config"]
    model = LeJEPAText(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.model_max_length = 1_000_000
    pad_id = cfg.mask_token_id

    def embed(texts):
        ids, pads = [], []
        for t in texts:
            x = tok.encode(t)[:seq_len]; m = len(x)
            ids.append(x + [pad_id] * (seq_len - m))
            pads.append([False] * m + [True] * (seq_len - m))
        ids = torch.tensor(ids); pad = torch.tensor(pads)
        outs = []
        for i in range(0, len(texts), bs):
            p = pad[i:i + bs].to(device)
            h = model.encoder(ids[i:i + bs].to(device), key_padding_mask=p)
            outs.append(_masked_mean(h, p).float().cpu())
        return F.normalize(torch.cat(outs), dim=-1)

    return embed(texts_ab[0]), embed(texts_ab[1]), cfg


@torch.no_grad()
def embed_hf(model_name, texts_ab, device, bs=64):
    from transformers import AutoTokenizer, AutoModel
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception:                                     # slow-only tokenizers (no fast/sentencepiece)
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def embed(texts):
        outs = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to(device)
            o = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (o * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            outs.append(emb.float().cpu())
        return F.normalize(torch.cat(outs), dim=-1)

    return embed(texts_ab[0]), embed(texts_ab[1]), None


# ── metric ──────────────────────────────────────────────────────────────────

def retrieval(EA, EB):
    """recall@1/@10 and MRR, averaged over both directions. EA,EB: (N,D) normed."""
    def one_way(Q, K):
        sims = Q @ K.T
        diag = sims.diag().unsqueeze(1)
        ranks = (sims > diag).sum(1) + 1                  # 1-indexed rank of true match
        return ((ranks <= 1).float().mean().item(),
                (ranks <= 10).float().mean().item(),
                (1.0 / ranks.float()).mean().item())
    ab = one_way(EA, EB)                                  # A→B
    ba = one_way(EB, EA)                                  # B→A
    return [(a + b) / 2 for a, b in zip(ab, ba)], ab, ba


def main():
    p = argparse.ArgumentParser(description="Cross-view retrieval (checkpoint or HF baseline).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="Our checkpoint (dict with model+config).")
    g.add_argument("--hf-model", help="HF model id for a baseline (e.g. bert-base-uncased).")
    p.add_argument("--pair-source", default="code", help="PAIR_PRESETS key (retrieval corpus).")
    p.add_argument("--split", default="", help="Held-out split; '' = auto (test/validation, else train).")
    p.add_argument("--n-pairs", type=int, default=1000, help="Pool size (chance recall@1 = 1/N).")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run-name", default="")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"

    A, B, split = stream_raw_pairs(a.pair_source, a.n_pairs, a.split or None)
    tag = os.path.basename(a.ckpt) if a.ckpt else f"HF:{a.hf_model}"
    print(f"{tag} | corpus={a.pair_source} (split={split}) | N={len(A)}")

    if a.ckpt:
        EA, EB, cfg = embed_ours(a.ckpt, (A, B), device, a.seq_len)
        default_name = "retr_" + os.path.splitext(os.path.basename(a.ckpt))[0]
    else:
        EA, EB, cfg = embed_hf(a.hf_model, (A, B), device)
        default_name = "retr_baseline_" + a.hf_model.replace("/", "-") + "_" + a.pair_source

    (r1, r10, mrr), ab, ba = retrieval(EA, EB)
    print(f"  recall@1={r1:.4f}  recall@10={r10:.4f}  MRR={mrr:.4f}  "
          f"(chance r@1={1.0/len(A):.4f})")
    print(f"  A→B r@1={ab[0]:.4f}  B→A r@1={ba[0]:.4f}")

    if a.wandb:
        import wandb
        from dotenv import load_dotenv
        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        cfg = cfg or Config()                             # HF baseline: use defaults for entity/project
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project,
                   name=a.run_name or default_name,
                   config={"src": tag, "pair_source": a.pair_source, "split": split, "n_pairs": len(A)})
        wandb.log({"retr/recall@1": r1, "retr/recall@10": r10, "retr/mrr": mrr,
                   "retr/AtoB_r@1": ab[0], "retr/BtoA_r@1": ba[0]})
        wandb.finish()


if __name__ == "__main__":
    main()
