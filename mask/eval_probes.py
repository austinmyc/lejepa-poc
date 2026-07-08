"""
Beyond-mean-pool probes — does JEPA change what MTEB cannot see?

Three readouts on FROZEN checkpoints, all cheap (no pretraining):

  1. FEW-SHOT linear probes (Banking77, k ∈ {4,16,64} labels/class, 5 seeds):
     sample-efficiency of the pooled representation — I-JEPA's headline eval.
  2. TOKEN-LEVEL probe (CoNLL-2003 NER, linear probe on frozen first-subtoken
     vectors): whether latent prediction enriched token representations that
     mean-pooling hides.
  3. LEARNED ATTENTION POOLER (Banking77 + STS-B): trains a small attention
     pooling head on frozen token matrices vs the same-budget mean-pool head —
     tests whether useful information exists in the token matrix that
     mean-pooling bottlenecks.

Usage (server):
    GPU=0 python mask/eval_probes.py \
        checkpoints_mask/design_20260705_033954_L8_ctrl_final.pt \
        checkpoints_mask/design3_L8_encspan_w01_final.pt \
        checkpoints_mask/diff_w1_final.pt

Interpretation: any JEPA-vs-ctrl gap here, given the MTEB tie, means the
latent term shapes structure pooled-embedding benchmarks cannot detect.
Results → stdout, JSON (mteb_results/probes/), W&B run per checkpoint.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from config import Config          # noqa: F401 (unpickle ckpt cfg)
from model import LeJEPAText


# ── frozen encoding ───────────────────────────────────────────────────────────

@torch.no_grad()
def encode_batch(model, tok, texts, device, seq_len, max_len=None):
    """Padding-masked encoder forward. Returns (h, key_pad): (B, L, D), (B, L)."""
    enc = model.encoder
    pad_id = tok.eos_token_id
    ids = [tok.encode(t)[: (max_len or seq_len)] or [pad_id] for t in texts]
    L = max(len(x) for x in ids)
    input_ids = torch.full((len(ids), L), pad_id, dtype=torch.long, device=device)
    key_pad = torch.ones((len(ids), L), dtype=torch.bool, device=device)
    for i, seq in enumerate(ids):
        input_ids[i, :len(seq)] = torch.tensor(seq, device=device)
        key_pad[i, :len(seq)] = False
    pos = torch.arange(L, device=device).unsqueeze(0)
    h = enc.drop(enc.tok_emb(input_ids) + enc.pos_emb(pos))
    h = enc.norm(enc.transformer(h, src_key_padding_mask=key_pad))
    return h, key_pad


@torch.no_grad()
def encode_pooled(model, tok, texts, device, seq_len, bs=128):
    out = []
    for lo in range(0, len(texts), bs):
        h, pad = encode_batch(model, tok, texts[lo:lo + bs], device, seq_len)
        keep = (~pad).unsqueeze(-1).float()
        out.append(((h * keep).sum(1) / keep.sum(1).clamp(min=1)).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def encode_token_matrix(model, tok, texts, device, seq_len, bs=64, cap_len=64):
    """Returns padded token matrices (N, cap_len, D) fp16 + lengths, for the pooler."""
    mats, lens = [], []
    for lo in range(0, len(texts), bs):
        h, pad = encode_batch(model, tok, texts[lo:lo + bs], device, seq_len, max_len=cap_len)
        B, L, D = h.shape
        m = torch.zeros(B, cap_len, D, dtype=torch.float16)
        m[:, :L] = h.half().cpu()
        m[:, :L][pad.cpu()] = 0
        mats.append(m)
        lens.append((~pad).sum(1).cpu())
    return torch.cat(mats), torch.cat(lens)


# ── rung 1: few-shot probes ───────────────────────────────────────────────────

def fewshot_probe(model, tok, cfg, device, ks=(4, 16, 64), n_seeds=5):
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    ds = load_dataset("mteb/banking77")
    Xtr = encode_pooled(model, tok, ds["train"]["text"], device, cfg.seq_len)
    ytr = np.array(ds["train"]["label"])
    Xte = encode_pooled(model, tok, ds["test"]["text"], device, cfg.seq_len)
    yte = np.array(ds["test"]["label"])

    res = {}
    for k in ks:
        accs = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            idx = np.concatenate([rng.choice(np.where(ytr == c)[0],
                                             min(k, (ytr == c).sum()), replace=False)
                                  for c in np.unique(ytr)])
            clf = LogisticRegression(max_iter=2000).fit(Xtr[idx], ytr[idx])
            accs.append(clf.score(Xte, yte))
        res[f"fewshot_b77_k{k}"] = float(np.mean(accs))
        res[f"fewshot_b77_k{k}_std"] = float(np.std(accs))
        print(f"    B77 {k:>3}-shot: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    return res


# ── rung 2: token-level probe (NER) ───────────────────────────────────────────

@torch.no_grad()
def _word_vectors(model, tok, words_batch, device, seq_len):
    """First-subtoken vector per word, via is_split_into_words alignment."""
    enc = model.encoder
    batch = tok(words_batch, is_split_into_words=True, truncation=True,
                max_length=seq_len, padding=True, return_tensors="pt").to(device)
    ids, attn = batch["input_ids"], batch["attention_mask"]
    pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    h = enc.drop(enc.tok_emb(ids) + enc.pos_emb(pos))
    h = enc.norm(enc.transformer(h, src_key_padding_mask=~attn.bool()))
    vecs, keys = [], []
    for i in range(len(words_batch)):
        wids = batch.word_ids(i)
        seen = set()
        for j, w in enumerate(wids):
            if w is not None and w not in seen:
                seen.add(w)
                vecs.append(h[i, j].cpu())
                keys.append((i, w))
    return vecs, keys


def token_probe(model, tok, cfg, device, max_train_tokens=30000):
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    # GPT-2 fast tokenizer needs add_prefix_space for pre-tokenized input.
    tok = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)
    tok.pad_token = tok.eos_token
    # All conll2003 hub repos are script-based (refused by newer `datasets`);
    # fetch HF's auto-converted parquet export to disk and load locally
    # (datasets intercepts huggingface.co URLs as repo paths). Tag col: "tags".
    import urllib.request
    base = "https://huggingface.co/api/datasets/tner/conll2003/parquet/conll2003"
    os.makedirs("data_cache/conll2003", exist_ok=True)
    files = {}
    for split in ("train", "test"):
        local = f"data_cache/conll2003/{split}.parquet"
        if not os.path.exists(local):
            urllib.request.urlretrieve(f"{base}/{split}/0.parquet", local)
        files[split] = local
    ds = load_dataset("parquet", data_files=files)

    def collect(split, cap):
        X, y = [], []
        for lo in range(0, len(split), 64):
            rows = split[lo:lo + 64]
            vecs, keys = _word_vectors(model, tok, rows["tokens"], device, cfg.seq_len)
            for v, (i, w) in zip(vecs, keys):
                if w < len(rows["tags"][i]):
                    X.append(v.numpy()); y.append(rows["tags"][i][w])
            if len(X) >= cap:
                break
        return np.array(X[:cap]), np.array(y[:cap])

    Xtr, ytr = collect(ds["train"], max_train_tokens)
    Xte, yte = collect(ds["test"], 20000)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    f1 = f1_score(yte, clf.predict(Xte), average="macro")
    print(f"    NER token probe: acc={acc:.4f}  macro-F1={f1:.4f}")
    return {"ner_acc": float(acc), "ner_macro_f1": float(f1)}


# ── rung 3: learned attention pooler ──────────────────────────────────────────

class AttnPool(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.w = nn.Linear(d, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self, H, lens):                       # H: (B, L, D)
        mask = torch.arange(H.shape[1], device=H.device)[None] >= lens[:, None]
        a = self.v(torch.tanh(self.w(H))).squeeze(-1).masked_fill(mask, -1e4)
        return (a.softmax(-1).unsqueeze(-1) * H).sum(1)


def _train_head(H, lens, y, n_cls, device, mode, epochs=10, bs=256, lr=1e-3):
    """mode='attn' trains AttnPool+linear; mode='mean' trains linear on mean-pool."""
    d = H.shape[-1]
    pool = AttnPool(d).to(device) if mode == "attn" else None
    head = nn.Linear(d, n_cls).to(device)
    params = list(head.parameters()) + (list(pool.parameters()) if pool else [])
    opt = torch.optim.Adam(params, lr=lr)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for lo in range(0, n, bs):
            i = perm[lo:lo + bs]
            Hb = H[i].to(device).float(); lb = lens[i].to(device); yb = y[i].to(device)
            if pool:
                emb = pool(Hb, lb)
            else:
                m = (torch.arange(Hb.shape[1], device=device)[None] < lb[:, None]).unsqueeze(-1).float()
                emb = (Hb * m).sum(1) / m.sum(1).clamp(min=1)
            loss = F.cross_entropy(head(emb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    def predict(Hs, ls):
        outs = []
        for lo in range(0, len(Hs), 512):
            Hb = Hs[lo:lo + 512].to(device).float(); lb = ls[lo:lo + 512].to(device)
            if pool:
                emb = pool(Hb, lb)
            else:
                m = (torch.arange(Hb.shape[1], device=device)[None] < lb[:, None]).unsqueeze(-1).float()
                emb = (Hb * m).sum(1) / m.sum(1).clamp(min=1)
            outs.append(head(emb).argmax(-1).cpu())
        return torch.cat(outs)
    return predict


def pooler_probe(model, tok, cfg, device):
    from datasets import load_dataset
    ds = load_dataset("mteb/banking77")
    Htr, ltr = encode_token_matrix(model, tok, ds["train"]["text"], device, cfg.seq_len)
    Hte, lte = encode_token_matrix(model, tok, ds["test"]["text"], device, cfg.seq_len)
    ytr = torch.tensor(ds["train"]["label"]); yte = torch.tensor(ds["test"]["label"])
    res = {}
    for mode in ("mean", "attn"):
        torch.manual_seed(0)
        predict = _train_head(Htr, ltr, ytr, 77, device, mode)
        acc = (predict(Hte, lte) == yte).float().mean().item()
        res[f"pooler_b77_{mode}"] = acc
        print(f"    B77 trained-{mode}-pool head: {acc:.4f}")
    res["pooler_b77_gap"] = res["pooler_b77_attn"] - res["pooler_b77_mean"]
    return res


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Beyond-mean-pool usefulness probes.")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--rungs", nargs="+", default=["fewshot", "token", "pooler"],
                    choices=["fewshot", "token", "pooler"])
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    gpu = os.environ.get("GPU", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    os.makedirs("mteb_results/probes", exist_ok=True)

    for path in args.checkpoints:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = LeJEPAText(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        name = os.path.basename(path).replace("_final.pt", "").replace(".pt", "")
        print(f"\n===== {name} (step {ckpt['step']}) =====")

        wb = None
        if not args.no_wandb:
            try:
                import wandb
                from dotenv import load_dotenv
                load_dotenv()
                wandb.login(key=os.getenv("WANDB_API_KEY"))
                wb = wandb.init(entity="austinmyc", project="lejepa",
                                name=f"probes_{name}", config={"checkpoint": path})
            except Exception as e:
                print(f"W&B unavailable ({e}) — stdout/JSON only.")

        results = {}
        runners = {"fewshot": fewshot_probe, "token": token_probe, "pooler": pooler_probe}
        for rung in args.rungs:
            print(f"  -- {rung} --")
            try:
                results.update(runners[rung](model, tok, cfg, device))
            except Exception as e:
                print(f"    {rung} FAILED: {e}")
                results[f"{rung}_error"] = str(e)

        with open(f"mteb_results/probes/{name}.json", "w") as f:
            json.dump(results, f, indent=2)
        if wb:
            wb.log({f"probes/{k}": v for k, v in results.items()
                    if isinstance(v, (int, float))})
            wb.finish()
        print(f"  saved → mteb_results/probes/{name}.json")


if __name__ == "__main__":
    main()
