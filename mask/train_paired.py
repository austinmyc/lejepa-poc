"""
Cross-view paired-JEPA training — EXPERIMENT_PLAN cell 1 (view gap YES,
abstraction gap NO). The counterpart to train.py: instead of masked-vs-clean
views of the SAME text (no information asymmetry, RQ1's dead cell), the two views
are DIFFERENT texts forming a genuine semantic pair (docstring↔code,
doc↔summary). This is the LLM-JEPA regime run FROM SCRATCH — the sharp test of
whether a real view gap alone rescues latent prediction, or whether the
abstraction gap (a pretrained backbone) is also necessary.

Loss:
    pred, target = model.forward_paired(x_a, x_b, pad_a, pad_b)   # both pooled
    loss = mse_weight * MSE(pred, target[stop-grad])              # cross-view
         + lam        * SIGReg(z_b)                               # target geometry
         + mlm_beta   * CE(decode(enc(mask(x_a))), x_a)           # optional anchor

Arms this scaffold expresses (all vs a matched pure-MLM baseline from train.py):
    pure cross-view    : --mlm-beta 0            (does a view gap escape 2/P?)
    anchored cross-view: --mlm-beta 1 --mlm-head encoder  (LLM-JEPA in miniature)
    shuffled control   : --shuffle-pairs         (chance-floor analogue: break A↔B)

Run:
    python train_paired.py --pair-source code --mlm-beta 1 --mlm-head encoder \
        --steps 30000 --wandb --mteb --run-name paired_code_anchored
"""

import os
import torch
import torch.nn.functional as F

from config import Config
from model  import LeJEPAText
from sigreg import sigreg_loss
from data   import get_paired_dataloader, make_masked_input
from eval_mteb import run_mteb_eval
# Reuse the single-text loop's helpers verbatim — same schedule, geometry probe,
# checkpoint format (importing does not run train.py; it is __main__-guarded).
from train import get_lr, space_geometry, save_checkpoint


def train_paired(cfg: Config):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"

    model = LeJEPAText(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = get_paired_dataloader(cfg)
    print(f"Device: {device}  |  paired: {cfg.pair_source} "
          f"({'SHUFFLED' if cfg.shuffle_pairs else 'real pairs'})  |  "
          f"mlm_beta={cfg.mlm_beta} head={cfg.mlm_head} mse_w={cfg.mse_weight} lam={cfg.lam}")
    print(f"Parameters: {model.count_params():,}")

    anchored = cfg.mlm_beta > 0
    if anchored:
        assert cfg.mlm_head == "encoder", \
            "paired CE anchor decodes from encoder space — use --mlm-head encoder"
        assert model.decoder is not None, "mlm_beta>0 but no decoder head was built"

    if cfg.use_wandb:
        import wandb
        from dotenv import load_dotenv
        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project,
                   name=cfg.run_name, config=cfg.__dict__)

    data_iter = iter(loader)
    for step in range(cfg.max_steps):
        try:
            ids_a, pad_a, ids_b, pad_b = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            ids_a, pad_a, ids_b, pad_b = next(data_iter)
        ids_a, pad_a = ids_a.to(device), pad_a.to(device)
        ids_b, pad_b = ids_b.to(device), pad_b.to(device)

        # Control arm: break the pairing by permuting view B within the batch.
        if cfg.shuffle_pairs:
            perm = torch.randperm(ids_b.shape[0], device=device)
            ids_b, pad_b = ids_b[perm], pad_b[perm]

        lr = get_lr(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr

        pred, target, z_a, z_b = model.forward_paired(ids_a, ids_b, pad_a, pad_b)

        if cfg.normalize_target:
            mse = F.mse_loss(F.normalize(pred, dim=-1), F.normalize(target, dim=-1))
        else:
            mse = F.mse_loss(pred, target)

        # SIGReg over the TARGET view's valid (non-pad) token projections.
        valid_b = z_b[~pad_b]                              # (N, P)
        reg = sigreg_loss(valid_b, num_slices=cfg.sigreg_num_slices,
                          num_points=cfg.sigreg_num_points, global_step=step)
        loss = cfg.mse_weight * mse + cfg.lam * reg

        # Optional BERT-style CE anchor on view A (separate masked pass) — turns
        # "pure cross-view" into "anchored cross-view" (the LLM-JEPA arm).
        ce = torch.tensor(0.0, device=device)
        if anchored:
            x_a_masked, mask_a = make_masked_input(ids_a, cfg)
            mask_a = mask_a & ~pad_a                       # never supervise on pad
            if mask_a.any():
                h_a_m = model.encoder(x_a_masked, key_padding_mask=pad_a)
                logits = model.decoder(h_a_m[mask_a])
                ce = F.cross_entropy(logits, ids_a[mask_a])
                loss = loss + cfg.mlm_beta * ce

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0:
            print(f"step {step:05d} | loss {loss.item():.4f} | mse {mse.item():.4f} | "
                  f"reg {reg.item():.4f} | ce {ce.item():.4f} | lr {lr:.2e}")
            if cfg.use_wandb:
                wandb.log({"loss": loss.item(), "mse": mse.item(),
                           "reg": reg.item(), "ce": ce.item(), "lr": lr}, step=step)

        if step % cfg.rank_every == 0:
            zg = space_geometry(z_b.detach())              # target proj (want isotropic)
            hg = space_geometry(z_a.detach())              # context view geometry
            P = z_b.shape[-1]
            print(f"  [diag] tgt_iso {zg['iso']:.2f} (rank {zg['rank']}/{P}, "
                  f"cos {zg['mean_cos']:.3f}) | ctx_eff_rank {hg['eff_rank']:.0f} | "
                  f"tgt_rms {zg['rms']:.2f}")
            if cfg.use_wandb:
                wandb.log({"tgt_rank": zg["rank"], "tgt_eff_rank": zg["eff_rank"],
                           "tgt_iso": zg["iso"], "tgt_mean_cos": zg["mean_cos"],
                           "tgt_rms": zg["rms"], "ctx_eff_rank": hg["eff_rank"]}, step=step)

        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            save_checkpoint(model, cfg, step)

    save_checkpoint(model, cfg, cfg.max_steps, final=True)

    if cfg.run_mteb:
        device_str = str(next(model.parameters()).device)
        wandb_run = wandb.run if cfg.use_wandb else None
        run_mteb_eval(model, cfg, cfg.max_steps, device_str, wandb_run=wandb_run)

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Cross-view paired-JEPA (cell 1).")
    p.add_argument("--steps",       type=int,   default=Config.max_steps)
    p.add_argument("--pair-source", type=str,   default=Config.pair_source,
                   help="PAIR_PRESETS key: 'code' | 'simplify' (or set --pair-repo).")
    p.add_argument("--pair-repo",   type=str,   default=Config.pair_repo,
                   help="Override HF parquet repo (namespaced). '' = use preset.")
    p.add_argument("--pair-col-a",  type=str,   default=Config.pair_col_a)
    p.add_argument("--pair-col-b",  type=str,   default=Config.pair_col_b)
    p.add_argument("--pair-split",  type=str,   default=Config.pair_split)
    p.add_argument("--shuffle-pairs", action="store_true",
                   help="Control: permute view B within batch (break A↔B pairing).")
    p.add_argument("--lr",          type=float, default=Config.lr)
    p.add_argument("--warmup-steps", type=int,  default=Config.warmup_steps)
    p.add_argument("--lam",         type=float, default=Config.lam)
    p.add_argument("--sigreg-grad-scale", type=float, default=Config.sigreg_grad_scale)
    p.add_argument("--d-model",     type=int,   default=Config.d_model)
    p.add_argument("--d-proj",      type=int,   default=Config.d_proj)
    p.add_argument("--n-heads",     type=int,   default=Config.n_heads)
    p.add_argument("--enc-layers",  type=int,   default=Config.enc_layers)
    p.add_argument("--pred-layers", type=int,   default=Config.pred_layers)
    p.add_argument("--batch-size",  type=int,   default=Config.batch_size)
    p.add_argument("--seq-len",     type=int,   default=Config.seq_len)
    p.add_argument("--latent-space", type=str,  default=Config.latent_space,
                   choices=["proj", "encoder"])
    p.add_argument("--no-normalize-target", action="store_true")
    p.add_argument("--mlm-beta",    type=float, default=Config.mlm_beta,
                   help="CE anchor weight on view A (0 = pure cross-view JEPA).")
    p.add_argument("--mlm-head",    type=str,   default="encoder",
                   choices=["pred", "encoder"])
    p.add_argument("--mse-weight",  type=float, default=Config.mse_weight)
    p.add_argument("--save-every",  type=int,   default=Config.save_every)
    p.add_argument("--seed",        type=int,   default=Config.seed)
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--mteb",        action="store_true")
    p.add_argument("--run-name",    type=str,   default="paired_jepa")
    a = p.parse_args()

    train_paired(Config(
        max_steps=a.steps,
        pair_source=a.pair_source, pair_repo=a.pair_repo,
        pair_col_a=a.pair_col_a, pair_col_b=a.pair_col_b, pair_split=a.pair_split,
        shuffle_pairs=a.shuffle_pairs,
        lr=a.lr, warmup_steps=a.warmup_steps,
        lam=a.lam, sigreg_grad_scale=a.sigreg_grad_scale,
        d_model=a.d_model, d_proj=a.d_proj,
        n_heads=a.n_heads, enc_layers=a.enc_layers, pred_layers=a.pred_layers,
        batch_size=a.batch_size, seq_len=a.seq_len,
        latent_space=a.latent_space,
        normalize_target=not a.no_normalize_target,
        mlm_beta=a.mlm_beta, mlm_head=a.mlm_head, mse_weight=a.mse_weight,
        save_every=a.save_every, seed=a.seed,
        use_wandb=a.wandb, run_mteb=a.mteb, run_name=a.run_name,
    ))
