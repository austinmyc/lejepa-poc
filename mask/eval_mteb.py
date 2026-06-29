"""
MTEB evaluation for a trained mask/ checkpoint — the real embedding-quality verdict.

Wraps the encoder as an MTEB-compatible model (the `.encode()` protocol) using the
standard SSL read-out: encoder → mean-pool (or proj). Batched + padding-masked so it
scales to the larger MTEB datasets, unlike the per-sentence loop in eval_sts.py.

    python mask/eval_mteb.py <checkpoint.pt> [--readout encoder|proj]
                                             [--tasks STSBenchmark SICK-R ...]
                                             [--benchmark "MTEB(eng, v2)"]
                                             [--batch-size 128] [--out mteb_results]

Default task set is a fast, multi-type slice (STS + classification + clustering) for
iteration. Pass --benchmark for a named MTEB suite, or --tasks for an explicit list.
Compare against a matched-compute BERT-base baseline — absolute scores mean little
for a 124M encoder trained on ~200M–2B tokens; the delta vs baseline is the signal.

Requires `pip install mteb` (kept out of requirements.txt as an eval-only extra).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from config import Config          # noqa: F401 (needed to unpickle cfg in ckpt)
from model import LeJEPAText

# A quick cross-task slice: two STS, one classification, one clustering.
DEFAULT_TASKS = [
    "STSBenchmark",
    "SICK-R",
    "Banking77Classification",
    "TwentyNewsgroupsClustering",
]


class LeJEPAEncoder:
    """MTEB-compatible wrapper exposing `.encode(sentences) -> np.ndarray`.

    Replicates TokenEncoder.forward but adds a src_key_padding_mask so a padded
    batch mean-pools over real tokens only — the batched equivalent of the
    per-sentence mean-pool in eval_sts.py.
    """

    def __init__(self, model, tok, device, seq_len, readout, batch_size):
        self.model = model
        self.tok = tok
        self.device = device
        self.seq_len = seq_len
        self.readout = readout
        self.batch_size = batch_size

    @torch.no_grad()
    def _encode_batch(self, sentences):
        enc = self.model.encoder
        pad_id = self.tok.eos_token_id
        ids = [self.tok.encode(s)[: self.seq_len] or [pad_id] for s in sentences]
        L = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), L), pad_id, dtype=torch.long, device=self.device)
        key_pad = torch.ones((len(ids), L), dtype=torch.bool, device=self.device)  # True = pad
        for i, seq in enumerate(ids):
            input_ids[i, : len(seq)] = torch.tensor(seq, device=self.device)
            key_pad[i, : len(seq)] = False

        pos = torch.arange(L, device=self.device).unsqueeze(0)
        h = enc.drop(enc.tok_emb(input_ids) + enc.pos_emb(pos))   # drop is identity in eval()
        h = enc.norm(enc.transformer(h, src_key_padding_mask=key_pad))
        if self.readout == "proj":
            h = self.model.proj(h)

        keep = (~key_pad).unsqueeze(-1).float()                   # (B, L, 1)
        emb = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return emb

    @torch.no_grad()
    def encode(self, sentences, *, batch_size=None, **kwargs):
        # **kwargs absorbs task_name / prompt_type / etc. that MTEB may pass.
        bs = batch_size or self.batch_size
        sentences = list(sentences)
        out = [self._encode_batch(sentences[i : i + bs]) for i in range(0, len(sentences), bs)]
        return torch.cat(out).cpu().numpy() if out else np.zeros((0, 1), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="MTEB evaluation for a mask/ checkpoint.")
    ap.add_argument("checkpoint")
    ap.add_argument("--readout", choices=["encoder", "proj"], default="encoder")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Explicit MTEB task names (overrides the default slice).")
    ap.add_argument("--benchmark", default=None,
                    help='Named MTEB benchmark, e.g. "MTEB(eng, v2)" (overrides --tasks).')
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default="mteb_results", help="Output folder for MTEB results.")
    args = ap.parse_args()

    try:
        import mteb
    except ImportError:
        raise SystemExit("MTEB not installed. Run:  pip install mteb")

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = LeJEPAText(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    encoder = LeJEPAEncoder(model, tok, device, cfg.seq_len, args.readout, args.batch_size)

    if args.benchmark:
        tasks = mteb.get_benchmark(args.benchmark)
    else:
        tasks = mteb.get_tasks(tasks=args.tasks or DEFAULT_TASKS)

    out_dir = os.path.join(args.out, f"{cfg.run_name}_{args.readout}")
    print(f"==> Running MTEB ({args.readout} readout, step {ckpt['step']}) → {out_dir}")
    results = mteb.MTEB(tasks=tasks).run(
        encoder, output_folder=out_dir, encode_kwargs={"batch_size": args.batch_size}
    )

    print("\n==> Main scores")
    for r in results:
        try:
            print(f"  {r.task_name:32} {r.get_score():.4f}")
        except Exception:
            print(f"  {getattr(r, 'task_name', r)}  (see {out_dir})")
    print(f"\nFull results written to {out_dir}/")


if __name__ == "__main__":
    main()
