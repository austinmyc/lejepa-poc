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
import json
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
    """MTEB EncoderProtocol-compatible wrapper for LeJEPAText.

    The new MTEB API passes a DataLoader[BatchedInput] to encode(), where each
    batch is a dict with a "sentences" key. We extract sentences, tokenize with
    GPT-2 BPE, mean-pool over real (non-pad) tokens, and return a numpy array.
    """

    def __init__(self, model, tok, device, seq_len, readout, batch_size):
        self.model = model
        self.tok = tok
        self.device = device
        self.seq_len = seq_len
        self.readout = readout
        self.batch_size = batch_size

        try:
            from mteb import ModelMeta
            self.mteb_model_meta = ModelMeta(name="lejepa", revision="0", languages=["eng"])
        except Exception:
            self.mteb_model_meta = None

    @torch.no_grad()
    def _encode_sentences(self, sentences):
        enc = self.model.encoder
        pad_id = self.tok.eos_token_id
        ids = [self.tok.encode(s)[: self.seq_len] or [pad_id] for s in sentences]
        L = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), L), pad_id, dtype=torch.long, device=self.device)
        key_pad = torch.ones((len(ids), L), dtype=torch.bool, device=self.device)
        for i, seq in enumerate(ids):
            input_ids[i, : len(seq)] = torch.tensor(seq, device=self.device)
            key_pad[i, : len(seq)] = False

        pos = torch.arange(L, device=self.device).unsqueeze(0)
        h = enc.drop(enc.tok_emb(input_ids) + enc.pos_emb(pos))
        h = enc.norm(enc.transformer(h, src_key_padding_mask=key_pad))
        if self.readout == "proj":
            h = self.model.proj(h)

        keep = (~key_pad).unsqueeze(-1).float()
        emb = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return emb.cpu().numpy()

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs):
        # inputs is a DataLoader[BatchedInput]; each batch is a dict with "sentences".
        all_embs = []
        for batch in inputs:
            sentences = batch["sentences"] if isinstance(batch, dict) else list(batch)
            all_embs.append(self._encode_sentences(sentences))
        return np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, 1), dtype=np.float32)

    def similarity(self, emb1, emb2):
        e1 = torch.tensor(emb1).float()
        e2 = torch.tensor(emb2).float()
        return F.normalize(e1, dim=-1) @ F.normalize(e2, dim=-1).T

    def similarity_pairwise(self, emb1, emb2):
        e1 = F.normalize(torch.tensor(emb1).float(), dim=-1)
        e2 = F.normalize(torch.tensor(emb2).float(), dim=-1)
        return (e1 * e2).sum(dim=-1)


def run_mteb_eval(
    model,
    cfg,
    step,
    device,
    readout="encoder",
    tasks=None,
    benchmark=None,
    batch_size=128,
    out="mteb_results",
    wandb_run=None,
):
    """
    Run MTEB eval on a live model and optionally log results to wandb.

    Args:
        model:      trained LeJEPAText instance (already on device, will be set to eval())
        cfg:        Config object (needs seq_len, run_name)
        step:       current training step (for naming / wandb logging)
        device:     torch device string
        readout:    "encoder" or "proj"
        tasks:      list of MTEB task names; falls back to DEFAULT_TASKS
        benchmark:  named MTEB benchmark string (overrides tasks)
        batch_size: encoding batch size
        out:        root directory for MTEB result JSON files
        wandb_run:  live wandb run object; if provided, scores are logged as
                    mteb/<task_name> and mteb/mean

    Returns:
        dict mapping task_name -> score (float), or {} on failure
    """
    try:
        import mteb
    except ImportError:
        print("MTEB not installed — skipping eval. Run:  pip install mteb")
        return {}

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    encoder = LeJEPAEncoder(model, tok, device, cfg.seq_len, readout, batch_size)
    model.eval()

    if benchmark:
        task_list = mteb.get_benchmark(benchmark)
    else:
        task_list = mteb.get_tasks(tasks=tasks or DEFAULT_TASKS)

    out_dir = os.path.join(out, f"{cfg.run_name}_{readout}_step{step}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"==> Running MTEB ({readout} readout, step {step}) → {out_dir}")
    results = []
    for task in task_list:
        print(f"  evaluating {task.metadata.name} ...")
        task_results = task.evaluate(encoder, encode_kwargs={"batch_size": batch_size})
        results.append(task_results)

    scores = {}
    print("\n==> MTEB scores")
    for task, r in zip(task_list, results):
        name = task.metadata.name
        try:
            s = r.get_score() if hasattr(r, "get_score") else r
            scores[name] = float(s)
            print(f"  {name:40} {s:.4f}")
        except Exception as e:
            print(f"  {name}  (no score: {e})")

    # Save scores + full result objects to disk.
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "scores.json")
    with open(scores_path, "w") as f:
        json.dump({"step": step, "readout": readout, "scores": scores,
                   "mean": sum(scores.values()) / len(scores) if scores else None}, f, indent=2)
    print(f"\nScores written to {scores_path}")

    # Save per-task full results (each TaskResult has a .to_dict()).
    for task, r in zip(task_list, results):
        try:
            task_path = os.path.join(out_dir, f"{task.metadata.name}.json")
            with open(task_path, "w") as f:
                json.dump(r.to_dict() if hasattr(r, "to_dict") else r, f, indent=2)
        except Exception:
            pass

    if scores and wandb_run is not None:
        log = {f"mteb/{k}": v for k, v in scores.items()}
        log["mteb/mean"] = sum(scores.values()) / len(scores)
        wandb_run.log(log, step=step)
        print(f"  logged {len(log)} MTEB metrics to W&B")

    return scores


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

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = LeJEPAText(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    run_mteb_eval(
        model, cfg, ckpt["step"], device,
        readout=args.readout,
        tasks=args.tasks,
        benchmark=args.benchmark,
        batch_size=args.batch_size,
        out=args.out,
    )


if __name__ == "__main__":
    main()
