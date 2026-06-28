"""
Data pipeline + masking for the mask/ build.

Datasets are copied from the root data.py (FakeDataset, ShakespeareDataset,
OpenWebTextDataset) and yield flat (seq_len,) token tensors. make_masked_input
is moved here from the root train.py so mask/train.py can import it from data.
"""

import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset


# ── datasets ──────────────────────────────────────────────────────────────────

class FakeDataset(IterableDataset):
    """Random token tensors (seq_len,) — no downloads, for smoke tests."""

    def __init__(self, vocab_size: int = 50257, seq_len: int = 128,
                 n_samples: int = 2000):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.n_samples  = n_samples

    def __iter__(self):
        for _ in range(self.n_samples):
            yield torch.randint(0, self.vocab_size, (self.seq_len,))


class ShakespeareDataset(IterableDataset):
    """tiny-shakespeare (~1MB) tokenised with GPT-2 BPE, yields (seq_len,)."""

    URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    def __init__(self, seq_len: int = 128, cache_dir: str = "./data_cache"):
        super().__init__()
        self.seq_len = seq_len

        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, "shakespeare.txt")
        if not os.path.exists(self.path):
            import urllib.request
            print(f"Downloading tiny-shakespeare → {self.path}")
            urllib.request.urlretrieve(self.URL, self.path)

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.model_max_length = 1_000_000

    def __iter__(self):
        with open(self.path) as f:
            text = f.read()
        tokens = self.tokenizer.encode(text)

        pos = 0
        while True:
            if pos + self.seq_len > len(tokens):
                pos = 0
            seq = tokens[pos: pos + self.seq_len]
            pos += self.seq_len
            yield torch.tensor(seq, dtype=torch.long)


class StreamingCorpusDataset(IterableDataset):
    """
    Packs tokenised documents from a streaming HF corpus into a rolling buffer,
    yields (seq_len,). Used for the full run (server). `corpus` must be a
    namespaced HF repo (e.g. "Skylion007/openwebtext", "HuggingFaceFW/fineweb").
    """

    def __init__(self, corpus: str = "Skylion007/openwebtext",
                 seq_len: int = 128, split: str = "train"):
        super().__init__()
        self.corpus  = corpus
        self.seq_len = seq_len
        self.split   = split

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.model_max_length = 1_000_000

    def __iter__(self):
        # No trust_remote_code (unsupported in current datasets); namespaced
        # parquet repos stream without a loading script.
        dataset = load_dataset(self.corpus, split=self.split, streaming=True)
        buffer = []
        for example in dataset:
            buffer.extend(self.tokenizer.encode(example["text"]))
            while len(buffer) >= self.seq_len:
                seq    = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield torch.tensor(seq, dtype=torch.long)


def get_dataloader(cfg, num_workers: int = 0) -> DataLoader:
    if cfg.fake_data:
        dataset = FakeDataset(vocab_size=cfg.vocab_size, seq_len=cfg.seq_len)
        return DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)

    if cfg.shakespeare:
        dataset = ShakespeareDataset(seq_len=cfg.seq_len, cache_dir=cfg.data_cache)
        return DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)

    dataset = StreamingCorpusDataset(corpus=cfg.corpus, seq_len=cfg.seq_len)
    pin = num_workers > 0
    return DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=num_workers,
        pin_memory=pin, prefetch_factor=4 if num_workers > 0 else None,
    )


# ── masking ───────────────────────────────────────────────────────────────────

def make_masked_input(x: torch.Tensor, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply token masking to a batch of sequences.

    Args:
        x:   (B, L) token ids — the clean input
        cfg: Config with mask_ratio, mask_strategy, mask_token_id

    Returns:
        x_masked: (B, L) with mask_token_id at masked positions
        mask:     (B, L) bool, True at masked positions

    Strategies:
      "random" — independent Bernoulli per token at rate mask_ratio
      "span"   — contiguous spans totalling ~mask_ratio of tokens
      "block"  — one contiguous block of length mask_ratio * L
    """
    B, L = x.shape

    # Dynamic ratio: sample per batch from Uniform(lo, hi) if a range is set.
    if cfg.mask_ratio_range:
        lo, hi = cfg.mask_ratio_range
        ratio = lo + (hi - lo) * torch.rand(1).item()
    else:
        ratio = cfg.mask_ratio

    if cfg.mask_strategy == "random":
        mask = torch.bernoulli(
            torch.full((B, L), ratio, device=x.device)
        ).bool()

    elif cfg.mask_strategy == "span":
        mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        budget = int(ratio * L)
        for b in range(B):
            covered = 0
            while covered < budget:
                span_len = torch.randint(3, 10, (1,)).item()
                start    = torch.randint(0, max(1, L - span_len), (1,)).item()
                end      = min(start + span_len, L)
                mask[b, start:end] = True
                covered  = mask[b].sum().item()

    elif cfg.mask_strategy == "block":
        mask    = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        blk_len = int(ratio * L)
        starts  = torch.randint(0, L - blk_len + 1, (B,))
        for b in range(B):
            mask[b, starts[b]: starts[b] + blk_len] = True

    else:
        raise ValueError(f"Unknown mask_strategy: {cfg.mask_strategy!r}")

    x_masked       = x.clone()
    x_masked[mask] = cfg.mask_token_id
    return x_masked, mask
