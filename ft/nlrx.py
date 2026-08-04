"""
NL-RX-SYNTH: natural-language → regex pairs (Locascio et al., EMNLP 2016).

LLM-JEPA's from-scratch pretraining experiment uses this dataset: they pretrain
Llama-3.2-1B from randomly initialised weights and report 54.38 → 60.59 accuracy
with the JEPA term added. We reproduce that and add the control they do not
report — scrambling which regex each description is paired with in the JEPA loss.

Source files are the parallel corpus from the deep-regex repo:
    src.txt  — one NL description per line
    targ.txt — the corresponding regex

Each example provides three tokenizations, because the two loss terms need
different inputs:
    seq_*   full "NL → regex" sequence           → next-token-prediction loss
    text_*  NL only, + k [PRED] tokens appended  → Pred(Enc(Text))
    code_*  regex only                           → Enc(Code)
"""

import os
import torch
from torch.utils.data import Dataset

BASE = ("https://raw.githubusercontent.com/nicholaslocascio/deep-regex/"
        "master/datasets/NL-RX-Synth")
SEP = " => "


def download(cache_dir="./data_cache/nlrx"):
    """Fetch src.txt/targ.txt once; return the parsed (nl, regex) pairs."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for name in ("src.txt", "targ.txt"):
        p = os.path.join(cache_dir, name)
        if not os.path.exists(p):
            import urllib.request
            print(f"downloading {BASE}/{name} → {p}")
            urllib.request.urlretrieve(f"{BASE}/{name}", p)
        paths[name] = p
    with open(paths["src.txt"]) as f:
        src = [l.strip() for l in f]
    with open(paths["targ.txt"]) as f:
        targ = [l.strip() for l in f]
    assert len(src) == len(targ), (len(src), len(targ))
    return [(s, t) for s, t in zip(src, targ) if s and t]


def splits(pairs, seed=1337):
    """Locascio's 65/10/25 train/val/test partition, deterministically shuffled."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(pairs), generator=g).tolist()
    n = len(pairs)
    a, b = int(0.65 * n), int(0.75 * n)
    take = lambda s: [pairs[i] for i in s]
    return take(idx[:a]), take(idx[a:b]), take(idx[b:])


class NLRXDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=128, n_pred_tokens=1,
                 pred_token="[PRED]"):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len
        self.pred_ids = tokenizer.encode(pred_token * n_pred_tokens,
                                         add_special_tokens=False)
        self.pad_id = tokenizer.pad_token_id or 0
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.pairs)

    def _pad(self, ids):
        ids = ids[: self.max_len]
        k = self.max_len - len(ids)
        return (torch.tensor(ids + [self.pad_id] * k, dtype=torch.long),
                torch.tensor([True] * len(ids) + [False] * k))   # True = real token

    def __getitem__(self, i):
        nl, rx = self.pairs[i]
        # Tokenize the prompt as "NL + SEP" (one call) so training and generation
        # see identical token boundaries — encoding NL and SEP separately can
        # merge differently at the seam and put the model off-distribution.
        nl_ids = self.tok.encode(nl + SEP, add_special_tokens=False)
        rx_ids = self.tok.encode(rx, add_special_tokens=False)
        if self.eos_id is not None:      # let the model learn to STOP; without an
            rx_ids = rx_ids + [self.eos_id]   # EOS it runs on and exact match is
                                              # unreachable (only prefix works)

        # Full sequence for next-token prediction; supervise the regex half only
        # (the NL prompt is context, exactly as in supervised NL→regex training).
        seq = (nl_ids + rx_ids)[: self.max_len]
        seq_ids, seq_real = self._pad(seq)
        sup = ([False] * len(nl_ids) + [True] * len(rx_ids))[: self.max_len]
        sup_mask = torch.tensor(sup + [False] * (self.max_len - len(sup)))

        # View A: NL + [PRED] tokens → the predictor's output is the last token's
        # final-layer hidden state (LLM-JEPA reuses the LLM's own weights as the
        # predictor by appending special tokens).
        text_ids, text_real = self._pad(nl_ids[: self.max_len - len(self.pred_ids)]
                                        + self.pred_ids)
        # View B: the regex alone → Enc(Code).
        code_ids, code_real = self._pad(self.tok.encode(rx, add_special_tokens=False))

        return {"seq_ids": seq_ids, "seq_real": seq_real, "sup_mask": sup_mask,
                "text_ids": text_ids, "text_real": text_real,
                "code_ids": code_ids, "code_real": code_real,
                "nl": nl, "regex": rx}


def collate(batch):
    keys = ["seq_ids", "seq_real", "sup_mask", "text_ids", "text_real",
            "code_ids", "code_real"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["nl"] = [b["nl"] for b in batch]
    out["regex"] = [b["regex"] for b in batch]
    return out
