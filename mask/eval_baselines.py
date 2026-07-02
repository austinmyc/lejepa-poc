"""
Baseline MTEB evaluation — BERT-base and GPT-2 mean-pool (no fine-tuning).

These are the fair comparison points for LeJEPA: pre-contrastive SSL models
that learn from token prediction, not similarity objectives.

    python mask/eval_baselines.py                    # all baselines, default tasks
    python mask/eval_baselines.py --tasks STSBenchmark SICK-R
    GPU=1 python mask/eval_baselines.py              # run on a specific GPU

Results written to mteb_results/baseline_<model>/ and printed to stdout.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

DEFAULT_TASKS = [
    "STSBenchmark",
    "SICK-R",
    "Banking77Classification",
    "TwentyNewsgroupsClustering",
]

BASELINES = {
    "bert-base": "bert-base-uncased",
    "gpt2":      "gpt2",
}


class MeanPoolEncoder:
    """HuggingFace model wrapped as an MTEB EncoderProtocol.

    Mean-pools over non-padding tokens (attention_mask > 0).
    Works for both BERT-style (with [PAD]) and GPT-2 (left-pad via eos).
    """

    def __init__(self, model_name, device, batch_size=64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name

        # GPT-2 has no pad token — reuse eos.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            from mteb import ModelMeta
            self.mteb_model_meta = ModelMeta(name=model_name, revision="0", languages=["eng"])
        except Exception:
            self.mteb_model_meta = None

    @torch.no_grad()
    def _encode_sentences(self, sentences):
        enc = self.tokenizer(
            list(sentences), padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc)
        # Mean-pool over non-padding tokens.
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return emb.cpu().numpy()

    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs):
        all_embs = []
        for batch in inputs:
            if isinstance(batch, dict):
                for key in ("text", "query", "passage", "sentences", "corpus"):
                    if key in batch:
                        sentences = batch[key]
                        break
                else:
                    sentences = next(v for v in batch.values() if isinstance(v, (list, tuple)))
            else:
                sentences = list(batch)
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


def run_baseline(name, hf_model_id, tasks, device, out_root, batch_size):
    import mteb
    print(f"\n{'='*60}")
    print(f"  Baseline: {name}  ({hf_model_id})")
    print(f"{'='*60}")

    encoder = MeanPoolEncoder(hf_model_id, device, batch_size)
    task_list = mteb.get_tasks(tasks=tasks)
    out_dir = os.path.join(out_root, f"baseline_{name}")
    os.makedirs(out_dir, exist_ok=True)

    scores = {}
    results = []
    for task in task_list:
        print(f"  evaluating {task.metadata.name} ...")
        r = task.evaluate(encoder, encode_kwargs={"batch_size": batch_size})
        results.append(r)
        try:
            subset_scores = [v["main_score"] for v in r.values() if "main_score" in v]
            s = sum(subset_scores) / len(subset_scores)
            scores[task.metadata.name] = s
            print(f"    {task.metadata.name:40} {s:.4f}")
        except Exception as e:
            print(f"    {task.metadata.name}  (no score: {e})")

    # Save summary.
    summary = {"model": hf_model_id, "scores": scores,
                "mean": sum(scores.values()) / len(scores) if scores else None}
    with open(os.path.join(out_dir, "scores.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-task full results.
    for task, r in zip(task_list, results):
        try:
            with open(os.path.join(out_dir, f"{task.metadata.name}.json"), "w") as f:
                json.dump(r, f, indent=2)
        except Exception:
            pass

    print(f"\n  Mean: {summary['mean']:.4f}")
    print(f"  Saved → {out_dir}/scores.json")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(BASELINES.keys()),
                    choices=list(BASELINES.keys()),
                    help="Which baselines to run (default: all)")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="mteb_results")
    args = ap.parse_args()

    gpu = os.environ.get("GPU", "3")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  (GPU={gpu})")

    all_scores = {}
    for name in args.models:
        hf_id = BASELINES[name]
        scores = run_baseline(name, hf_id, args.tasks, device, args.out, args.batch_size)
        all_scores[name] = scores

    # Print comparison table.
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    task_names = args.tasks
    print(f"  {'Model':<12} " + "  ".join(f"{t[:18]:<20}" for t in task_names) + "  Mean")
    for name, scores in all_scores.items():
        vals = [scores.get(t, float("nan")) for t in task_names]
        mean = sum(v for v in vals if not np.isnan(v)) / len(vals)
        print(f"  {name:<12} " + "  ".join(f"{v:<20.4f}" for v in vals) + f"  {mean:.4f}")


if __name__ == "__main__":
    main()
