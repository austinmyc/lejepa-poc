"""
Streaming data pipeline for OpenWebText.

Streams documents, tokenises with GPT-2 BPE, packs tokens into a rolling
buffer, and yields (B, L) tensors — B=batch, L=seq_len tokens.

No chunking — the training loop receives flat sequences and applies masking
itself via make_masked_input() in train.py.
"""

import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset


class FakeDataset(IterableDataset):
    """
    Yields random token tensors shaped (seq_len,).
    No downloads, no internet. Use for local dev and smoke tests.
    """

    def __init__(self, vocab_size: int = 50257, seq_len: int = 1024,
                 n_samples: int = 2000):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.n_samples  = n_samples

    def __iter__(self):
        for _ in range(self.n_samples):
            yield torch.randint(0, self.vocab_size, (self.seq_len,))


class ShakespeareDataset(IterableDataset):
    """
    Downloads tiny-shakespeare (~1MB) and yields (seq_len,) tensors.
    Good for local dev — fast download, no GPU needed, real text structure.
    """

    URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    def __init__(self, seq_len: int = 1024, cache_dir: str = "./data_cache"):
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
        tokens = self.tokenizer.encode(text)  # ~300k tokens

        pos = 0
        while True:
            if pos + self.seq_len > len(tokens):
                pos = 0   # wrap around
            seq = tokens[pos: pos + self.seq_len]
            pos += self.seq_len
            yield torch.tensor(seq, dtype=torch.long)


class OpenWebTextDataset(IterableDataset):
    """
    Yields (seq_len,) long tensors by packing tokenised documents
    into a rolling buffer and slicing off fixed-length sequences.
    """

    def __init__(self, seq_len: int = 1024, split: str = "train"):
        super().__init__()
        self.seq_len = seq_len
        self.split   = split

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.model_max_length = 1_000_000

    def __iter__(self):
        dataset = load_dataset(
            "openwebtext",
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

        buffer = []
        for example in dataset:
            tokens = self.tokenizer.encode(example["text"])
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len:
                seq    = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield torch.tensor(seq, dtype=torch.long)


def get_dataloader(cfg, num_workers: int = 4) -> DataLoader:
    if cfg.fake_data:
        dataset = FakeDataset(vocab_size=cfg.vocab_size, seq_len=cfg.seq_len)
        return DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)

    if cfg.shakespeare:
        dataset = ShakespeareDataset(seq_len=cfg.seq_len, cache_dir=cfg.data_cache)
        return DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)

    dataset = OpenWebTextDataset(seq_len=cfg.seq_len)
    pin = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        prefetch_factor=4 if num_workers > 0 else None,
    )
