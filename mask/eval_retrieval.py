"""
Cross-view retrieval eval for frozen paired-JEPA checkpoints.

Why this and not GSM8K / NL-Regex / Spider (LLM-JEPA's tasks): those are
GENERATIVE and need a pretrained decoder — impossible on our from-scratch 768-d
bidirectional ENCODER checkpoints. The encoder-appropriate analogue that ALSO
directly tests the paired hypothesis is bitext retrieval: freeze the encoder,
embed held-out (view_A, view_B) pairs, and ask whether A retrieves its true B.

This is the metric that answers "did the pairing teach the correspondence?" —
which MTEB (generic sentence similarity) does NOT. The sharp read:
    anchored  >  shuffled   → real pairs taught the A↔B mapping (pairing helped),
                              even if MTEB looked flat.
    anchored  ≈  shuffled   → the pairing carried no retrievable signal.

Runs on ANY checkpoint saved by train.py / train_paired.py (dict with
"model"+"config"), including Part I single-text encoders (a non-paired baseline).

Usage:
    python mask/eval_retrieval.py --ckpt checkpoints_mask/paired_..._code_anchored_final.pt \
        --pair-source code --n-pairs 1000 --wandb
"""

import argparse
import os
import torch
import torch.nn.functional as F

from config import Config
from model  import LeJEPAText, _masked_mean
from data   import PAIR_PRESETS, PairedCorpusDataset


def load_model(ckpt_path, device):
    # weights_only=False: the checkpoint pickles the Config dataclass instance.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = LeJEPAText(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, cfg


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


@torch.no_grad()
def embed(model, ids, pad, device, bs=128):
    """Frozen encoder mean-pool (the MTEB readout), L2-normalized. (N,L)->(N,D)."""
    outs = []
    for i in range(0, ids.shape[0], bs):
        x = ids[i:i + bs].to(device)
        p = pad[i:i + bs].to(device)
        h = model.encoder(x, key_padding_mask=p)          # (b,L,D)
        outs.append(_masked_mean(h, p).float().cpu())     # (b,D) mean over valid
    return F.normalize(torch.cat(outs), dim=-1)


def retrieval(EA, EB):
    """recall@1/@10 and MRR, averaged over both directions. EA,EB: (N,D) normed."""
    def one_way(Q, K):
        sims = Q @ K.T                                    # (N,N)
        diag = sims.diag().unsqueeze(1)
        ranks = (sims > diag).sum(1) + 1                  # 1-indexed rank of true match
        return ((ranks <= 1).float().mean().item(),
                (ranks <= 10).float().mean().item(),
                (1.0 / ranks.float()).mean().item())
    ab = one_way(EA, EB)                                  # A→B
    ba = one_way(EB, EA)                                  # B→A
    return [(a + b) / 2 for a, b in zip(ab, ba)], ab, ba


def main():
    p = argparse.ArgumentParser(description="Cross-view retrieval on a frozen checkpoint.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pair-source", default="code", help="PAIR_PRESETS key (retrieval corpus).")
    p.add_argument("--split", default="", help="Held-out split; '' = auto (test/validation, else train).")
    p.add_argument("--n-pairs", type=int, default=1000, help="Retrieval pool size (chance recall@1 = 1/N).")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run-name", default="")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    model, cfg = load_model(a.ckpt, device)

    repo, col_a, col_b, _, config = PAIR_PRESETS[a.pair_source]
    split = pick_split(repo, config, a.split or None)
    print(f"ckpt={os.path.basename(a.ckpt)} | corpus={a.pair_source} ({repo}, split={split}) | N={a.n_pairs}")

    ds = PairedCorpusDataset(repo, col_a, col_b, seq_len=a.seq_len,
                             split=split, pad_id=cfg.mask_token_id, config=config)
    ids_a, pad_a, ids_b, pad_b = [], [], [], []
    for rec in ds:
        ids_a.append(rec[0]); pad_a.append(rec[1]); ids_b.append(rec[2]); pad_b.append(rec[3])
        if len(ids_a) >= a.n_pairs:
            break
    ids_a = torch.stack(ids_a); pad_a = torch.stack(pad_a)
    ids_b = torch.stack(ids_b); pad_b = torch.stack(pad_b)

    EA = embed(model, ids_a, pad_a, device)
    EB = embed(model, ids_b, pad_b, device)
    (r1, r10, mrr), ab, ba = retrieval(EA, EB)
    print(f"  recall@1={r1:.4f}  recall@10={r10:.4f}  MRR={mrr:.4f}  "
          f"(chance r@1={1.0/EA.shape[0]:.4f})")
    print(f"  A→B r@1={ab[0]:.4f}  B→A r@1={ba[0]:.4f}")

    if a.wandb:
        import wandb
        from dotenv import load_dotenv
        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        name = a.run_name or ("retr_" + os.path.splitext(os.path.basename(a.ckpt))[0])
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=name,
                   config={"ckpt": a.ckpt, "pair_source": a.pair_source, "split": split,
                           "n_pairs": EA.shape[0]})
        wandb.log({"retr/recall@1": r1, "retr/recall@10": r10, "retr/mrr": mrr,
                   "retr/AtoB_r@1": ab[0], "retr/BtoA_r@1": ba[0]})
        wandb.finish()


if __name__ == "__main__":
    main()
