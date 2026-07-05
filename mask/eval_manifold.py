"""
Post-hoc manifold fitting on frozen sentence embeddings — MTEB comparison.

Ports the cylinder-averaging manifold fitting from
github.com/austinmyc/manifold-fitting-finance (manfit/manfit_ours.py) and
evaluates three transforms of the same frozen embeddings on MTEB tasks:

    raw       — mean-pooled encoder output as-is
    whiten    — linear whitening (BERT-whitening baseline; Su et al. 2021)
    manfit    — nonlinear manifold contraction (ours)

Protocol per task (transductive, standard for this post-processing family —
same setting as BERT-flow/-whitening): pass 1 runs MTEB normally while
recording every embedding; the transform is fitted on that task's own
(unlabeled) embedding cloud; pass 2 re-runs MTEB with the transform applied
inside encode().

    python mask/eval_manifold.py --checkpoint checkpoints_mask/<run>_final.pt
    python mask/eval_manifold.py --baseline bert-base
        [--tasks STSBenchmark SICK-R ...] [--sigma 0.15] [--fit-max 3000]
        [--transforms raw whiten manfit] [--out mteb_results]

Notes:
  - Embeddings are scale-normalized by the cloud's median pairwise distance
    before fitting, so --sigma has the same meaning across models.
  - manfit here is algebraically equivalent to the original (QR eliminated:
    the cylinder test only needs the component along dx and the orthogonal
    residual norm), chunked with torch for speed.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from config import Config          # noqa: F401 (unpickle ckpt cfg)
from model import LeJEPAText
from eval_mteb import LeJEPAEncoder, DEFAULT_TASKS
from eval_baselines import MeanPoolEncoder, BASELINES


# ── manifold fitting (port of manfit_ours.py, vectorized) ────────────────────

@torch.no_grad()
def manfit(sample, points, sig, device="cpu", chunk=256):
    """
    Contract `points` toward the manifold underlying `sample`.

    sample: (N0, D) fitting cloud;  points: (M, D) points to transform.
    Faithful to manfit_ours: r = 5σ/log10(N0), R = 10σ√(log(1/σ))/log10(N0);
    neighborhood = {dist < 2r} ∪ 5-NN; direction dx = x − mean(neighborhood);
    output = mean of samples inside the cylinder {|t| < R, ⊥dist < r} when it
    holds > 10 points, else the neighborhood mean.
    """
    S = torch.as_tensor(sample, dtype=torch.float32, device=device)   # (N0, D)
    X = torch.as_tensor(points, dtype=torch.float32, device=device)   # (M, D)
    N0 = S.shape[0]
    r = 5 * sig / np.log10(N0)
    R = 10 * sig * np.sqrt(np.log(1 / sig)) / np.log10(N0)

    out = torch.empty_like(X)
    for lo in range(0, X.shape[0], chunk):
        x = X[lo:lo + chunk]                                # (C, D)
        d = torch.cdist(x, S)                               # (C, N0)

        # Neighborhood: within 2r, plus 5 nearest as a floor.
        near = d < 2 * r
        knn = d.topk(5, largest=False).indices
        near.scatter_(1, knn, True)

        w = near.float()
        xbar = (w @ S) / w.sum(1, keepdim=True).clamp(min=1)          # (C, D)
        dx = x - xbar + torch.finfo(torch.float32).eps
        u = F.normalize(dx, dim=1)                                    # (C, D)

        # Cylinder test without QR: t = (s−x)·u ; ⊥² = ‖s−x‖² − t².
        t = S @ u.T - (x * u).sum(1)                                  # (N0, C)
        t = t.T                                                       # (C, N0)
        perp2 = d.pow(2) - t.pow(2)
        cyl = (t.abs() < R) & (perp2 < r * r)                         # (C, N0)

        counts = cyl.sum(1)
        wc = cyl.float()
        cyl_mean = (wc @ S) / counts.clamp(min=1).unsqueeze(1)
        out[lo:lo + chunk] = torch.where(counts.unsqueeze(1) > 10, cyl_mean, xbar)
    return out.cpu().numpy()


def whiten_fit(cloud):
    """BERT-whitening: W = U diag(λ^-1/2) Uᵀ fitted on the cloud."""
    mu = cloud.mean(0, keepdims=True)
    cov = np.cov((cloud - mu).T)
    U, s, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(s + 1e-9)) @ U.T
    return lambda E: (E - mu) @ W


# ── encode-wrapping ───────────────────────────────────────────────────────────

class TransformWrapper:
    """Wraps an MTEB encoder; records raw embeddings and/or applies a transform."""

    def __init__(self, base):
        self.base = base
        self.transform = None      # callable (M, D) -> (M, D), or None
        self.recorded = []
        self.mteb_model_meta = getattr(base, "mteb_model_meta", None)

    def encode(self, inputs, **kwargs):
        emb = self.base.encode(inputs, **kwargs)
        if self.transform is None:
            self.recorded.append(np.asarray(emb))
            return emb
        return self.transform(np.asarray(emb))

    def similarity(self, a, b):
        a, b = torch.tensor(a).float(), torch.tensor(b).float()
        return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T

    def similarity_pairwise(self, a, b):
        a = F.normalize(torch.tensor(a).float(), dim=-1)
        b = F.normalize(torch.tensor(b).float(), dim=-1)
        return (a * b).sum(dim=-1)


def task_score(result):
    subset = [v["main_score"] for v in result.values() if "main_score" in v]
    return sum(subset) / len(subset)


def main():
    ap = argparse.ArgumentParser(description="Post-hoc manifold fitting on MTEB.")
    ap.add_argument("--checkpoint", default=None, help="mask/ checkpoint (.pt)")
    ap.add_argument("--baseline", default=None, choices=list(BASELINES.keys()))
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--transforms", nargs="+", default=["raw", "whiten", "manfit"],
                    choices=["raw", "whiten", "manfit"])
    ap.add_argument("--sigma", type=float, default=0.15,
                    help="manfit σ in median-distance-normalized space (0.01–1).")
    ap.add_argument("--fit-max", type=int, default=3000,
                    help="Max cloud points used to fit manfit/whitening.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default="mteb_results")
    args = ap.parse_args()

    gpu = os.environ.get("GPU", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"

    import mteb

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = LeJEPAText(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        base = LeJEPAEncoder(model, tok, device, cfg.seq_len, "encoder", args.batch_size)
        model_name = f"{cfg.run_name}_step{ckpt['step']}"
    elif args.baseline:
        base = MeanPoolEncoder(BASELINES[args.baseline], device, args.batch_size)
        model_name = f"baseline_{args.baseline}"
    else:
        raise SystemExit("Provide --checkpoint or --baseline.")

    out_dir = os.path.join(args.out, f"manifold_{model_name}")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    all_scores = {t: {} for t in args.tasks}

    for task_name in args.tasks:
        task = mteb.get_tasks(tasks=[task_name])[0]
        wrapper = TransformWrapper(base)

        # Pass 1: raw scores + record the task's embedding cloud.
        print(f"\n==> {task_name}: pass 1 (raw + record)")
        res_raw = task.evaluate(wrapper, encode_kwargs={"batch_size": args.batch_size})
        all_scores[task_name]["raw"] = task_score(res_raw)
        cloud = np.concatenate(wrapper.recorded, axis=0)
        print(f"    raw={all_scores[task_name]['raw']:.4f}   cloud={cloud.shape}")

        # Scale-normalize so σ is comparable across models.
        sub = cloud[rng.choice(len(cloud), min(2000, len(cloud)), replace=False)]
        scale = float(np.median(np.linalg.norm(sub[:, None, :] - sub[None, ::7, :], axis=-1)))
        fit_cloud = cloud[rng.choice(len(cloud), min(args.fit_max, len(cloud)),
                                     replace=False)] / scale

        for tf in args.transforms:
            if tf == "raw":
                continue
            if tf == "whiten":
                w = whiten_fit(cloud)
                wrapper.transform = w
            else:
                wrapper.transform = lambda E: manfit(
                    fit_cloud, np.asarray(E) / scale, args.sigma, device=device) * scale
            print(f"==> {task_name}: pass 2 ({tf})")
            res = task.evaluate(wrapper, encode_kwargs={"batch_size": args.batch_size})
            all_scores[task_name][tf] = task_score(res)
            print(f"    {tf}={all_scores[task_name][tf]:.4f}")
            wrapper.transform = None
            wrapper.recorded = []   # don't re-record during/after transform passes

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'task':32} " + "  ".join(f"{t:>8}" for t in args.transforms))
    means = {t: [] for t in args.transforms}
    for task_name, scores in all_scores.items():
        row = []
        for t in args.transforms:
            s = scores.get(t, float("nan"))
            means[t].append(s)
            row.append(f"{s:8.4f}")
        print(f"{task_name:32} " + "  ".join(row))
    print(f"{'MEAN':32} " + "  ".join(f"{np.nanmean(means[t]):8.4f}" for t in args.transforms))

    payload = {"model": model_name, "sigma": args.sigma, "fit_max": args.fit_max,
               "scores": all_scores,
               "means": {t: float(np.nanmean(means[t])) for t in args.transforms}}
    with open(os.path.join(out_dir, f"scores_sig{args.sigma}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out_dir}/scores_sig{args.sigma}.json")


if __name__ == "__main__":
    main()
