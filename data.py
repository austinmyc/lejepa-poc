"""
Streaming data pipeline for OpenWebText.

Streams documents, tokenises with GPT-2 BPE, packs tokens into a rolling
buffer, and yields (B, N, C) tensors — B=batch, N=num_chunks, C=chunk_size.

No need to download the full 40 GB dataset; HuggingFace streaming handles it.
"""

import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset


class FakeChunkedDataset(IterableDataset):
    """
    Yields random token tensors shaped (num_chunks, chunk_size).
    No downloads, no internet. Use for local dev and smoke tests.
    """

    def __init__(self, vocab_size: int = 50257, chunk_size: int = 64,
                 num_chunks: int = 16, n_samples: int = 2000):
        super().__init__()
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.n_samples  = n_samples

    def __iter__(self):
        for _ in range(self.n_samples):
            yield torch.randint(0, self.vocab_size, (self.num_chunks, self.chunk_size))


class ShakespeareChunked(IterableDataset):
    """
    Downloads tiny-shakespeare (~1MB) and yields (num_chunks, chunk_size) tensors.
    Good for local dev — fast download, no GPU needed, real text structure.
    """

    URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    def __init__(self, chunk_size: int = 64, num_chunks: int = 16, cache_dir: str = "./data_cache"):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.seq_len = chunk_size * num_chunks

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

        # cycle repeatedly so we never run out
        pos = 0
        while True:
            if pos + self.seq_len > len(tokens):
                pos = 0  # wrap around
            seq = tokens[pos: pos + self.seq_len]
            pos += self.seq_len
            t = torch.tensor(seq, dtype=torch.long)
            yield t.reshape(self.num_chunks, self.chunk_size)


class OpenWebTextChunked(IterableDataset):
    """
    Yields (num_chunks, chunk_size) long tensors by packing tokenised documents
    into a rolling buffer and slicing off fixed-length sequences.
    """

    def __init__(self, chunk_size: int = 64, num_chunks: int = 16, split: str = "train"):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.seq_len = chunk_size * num_chunks        # e.g. 1024 tokens
        self.split = split

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.model_max_length = 1_000_000  # don't truncate during tokenisation

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

            # yield as many full sequences as available
            while len(buffer) >= self.seq_len:
                seq = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]

                t = torch.tensor(seq, dtype=torch.long)
                yield t.reshape(self.num_chunks, self.chunk_size)


def get_dataloader(cfg, num_workers: int = 4) -> DataLoader:
    if cfg.fake_data:
        dataset = FakeChunkedDataset(
            vocab_size=cfg.vocab_size,
            chunk_size=cfg.chunk_size,
            num_chunks=cfg.max_chunks,
        )
        return DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)

    if cfg.shakespeare:
        dataset = ShakespeareChunked(
            chunk_size=cfg.chunk_size,
            num_chunks=cfg.max_chunks,
            cache_dir=cfg.data_cache,
        )
        return DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)

    dataset = OpenWebTextChunked(
        chunk_size=cfg.chunk_size,
        num_chunks=cfg.max_chunks,
    )
    pin = num_workers > 0   # pin_memory only useful with CUDA + multiple workers
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        prefetch_factor=4 if num_workers > 0 else None,
    )
