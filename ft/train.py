"""
DLLM-JEPA finetuning (reproduction) + CoT-as-action (extension).

    L_total = L_diff + λ · L_JEPA

Arms (see PLAN.md; --action / --lam / --cot-in-target select them):
    A0 sft        --lam 0                          plain diffusion SFT baseline
    A1 jepa       --lam 1 --action none            DLLM-JEPA reproduction (GATE)
    A2 jepa_cot   --lam 1 --action cot             CoT as the predictor's action
    A3 jepa_shuf  --lam 1 --action shuffled        faithfulness control
    A4 jepa_rand  --lam 1 --action random          any-conditioning control
    A5 cot_sft    --lam 0 --cot-in-target          plain CoT-SFT comparator

Run (smoke, tiny model, CPU/1 GPU):
    python ft/train.py --smoke
Real:
    python ft/train.py --model GSAI-ML/LLaDA-8B-Instruct --lam 1 --action none \
        --run-name A1_jepa --wandb
"""

import argparse, math, os, sys, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from data   import GSM8KDataset, collate
from model  import DLLMJepa, diffusion_loss, make_view, jepa_loss


def resolve_mask_id(tok, model, cfg):
    if cfg.mask_token_id:
        return cfg.mask_token_id
    if getattr(tok, "mask_token_id", None) is not None:
        return tok.mask_token_id
    mid = getattr(model.config, "mask_token_id", None)
    if mid is not None:
        return mid
    raise ValueError("Could not resolve a mask token id — pass --mask-token-id "
                     "(LLaDA-8B uses 126336).")


def build(cfg, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=cfg.trust_remote)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, cfg.dtype) if device != "cpu" else torch.float32
    backbone = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, trust_remote_code=cfg.trust_remote, torch_dtype=dtype)

    if cfg.grad_ckpt and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model
        targets = [s for s in cfg.lora_targets.split(",") if s] or None
        backbone = get_peft_model(backbone, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=targets, task_type="CAUSAL_LM"))
        backbone.print_trainable_parameters()

    hidden = getattr(backbone.config, "hidden_size", None) or backbone.config.d_model
    model = DLLMJepa(backbone, cfg, hidden).to(device)
    return tok, model, hidden


def train(cfg, smoke=False):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model, hidden = build(cfg, device)
    mask_id = resolve_mask_id(tok, model.backbone, cfg)
    print(f"device={device} hidden={hidden} mask_id={mask_id} λ={cfg.lam} "
          f"action={cfg.action} t=({cfg.t_low},{cfg.t_high}) k={cfg.pred_layers}")

    ds = GSM8KDataset(tok, cfg, split="train")
    if smoke:
        ds.rows = ds.rows.select(range(8))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = max(1, len(dl) // cfg.grad_accum * cfg.epochs)
    warm = max(1, int(total * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    if cfg.lam > 0:
        model.ema_init()

    if cfg.use_wandb:
        import wandb
        from dotenv import load_dotenv
        load_dotenv(); wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project,
                   name=cfg.run_name, config=cfg.__dict__)

    step = 0
    os.makedirs(cfg.out_dir, exist_ok=True)
    for ep in range(cfg.epochs):
        for i, b in enumerate(dl):
            ids  = b["input_ids"].to(device)
            resp = b["resp_mask"].to(device)
            pad  = b["pad_mask"].to(device)

            # ── L_diff: standard LLaDA SFT (t ~ U(0,1) keeps the denoiser
            # healthy at every mask rate, which iterative generation needs).
            if cfg.diff_noise == "uniform":
                l_diff, *_ = diffusion_loss(model, ids, resp, pad, mask_id)
            else:                                    # reuse the t_L context pass
                t_ctx = torch.full((ids.shape[0],), cfg.t_low, device=device)
                l_diff, *_ = diffusion_loss(model, ids, resp, pad, mask_id, t=t_ctx)

            # ── L_JEPA: two views of the SAME input at t_L / t_H.
            l_jepa = torch.zeros((), device=device)
            if cfg.lam > 0:
                ctx_ids, ctx_valid = make_view(ids, resp, pad, cfg.t_low,  mask_id)
                tgt_ids, tgt_valid = make_view(ids, resp, pad, cfg.t_high, mask_id)
                z_ctx = model.state(ctx_ids, ctx_valid, (~pad).long())
                z_tgt = model.target_state(tgt_ids, tgt_valid, (~pad).long())
                a = model.action_vector(b["cot_ids"].to(device),
                                        b["cot_pad"].to(device), cfg.action)
                l_jepa = jepa_loss(model.predictor(z_ctx, a), z_tgt)

            loss = (l_diff + cfg.lam * l_jepa) / cfg.grad_accum
            loss.backward()

            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                if cfg.lam > 0:
                    model.ema_update()
                step += 1
                if step % cfg.log_every == 0:
                    print(f"ep{ep} step {step}/{total} | diff {l_diff.item():.4f} | "
                          f"jepa {l_jepa.item():.4f} | lr {sched.get_last_lr()[0]:.2e}")
                    if cfg.use_wandb:
                        wandb.log({"l_diff": l_diff.item(), "l_jepa": l_jepa.item(),
                                   "lr": sched.get_last_lr()[0]}, step=step)
            if smoke and i >= 3:
                break
        if smoke:
            break

    path = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(path, exist_ok=True)
    (model.backbone.save_pretrained(path) if cfg.use_lora
     else torch.save(model.backbone.state_dict(), os.path.join(path, "model.pt")))
    torch.save({"predictor": model.predictor.state_dict(), "cfg": cfg},
               os.path.join(path, "jepa_head.pt"))
    print(f"saved → {path}")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DLLM-JEPA finetuning (+ CoT action).")
    p.add_argument("--model", default=Config.model_name)
    p.add_argument("--lam", type=float, default=Config.lam)
    p.add_argument("--action", default=Config.action,
                   choices=["none", "cot", "shuffled", "random"])
    p.add_argument("--cot-in-target", action="store_true", help="A5: plain CoT-SFT.")
    p.add_argument("--t-low", type=float, default=Config.t_low)
    p.add_argument("--t-high", type=float, default=Config.t_high)
    p.add_argument("--pred-layers", type=int, default=Config.pred_layers)
    p.add_argument("--ema-decay", type=float, default=Config.ema_decay)
    p.add_argument("--diff-noise", default=Config.diff_noise, choices=["uniform", "context"])
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--grad-accum", type=int, default=Config.grad_accum)
    p.add_argument("--max-len", type=int, default=Config.max_len)
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--mask-token-id", type=int, default=Config.mask_token_id)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--out-dir", default=Config.out_dir)
    p.add_argument("--run-name", default=Config.run_name)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny model + 4 steps — verifies plumbing without a GPU.")
    a = p.parse_args()

    cfg = Config(
        # Smoke uses real gpt2 (full 50257 vocab so GSM8K token ids are in range;
        # it is NOT a diffusion LM — this only exercises the plumbing).
        model_name="gpt2" if a.smoke else a.model,
        lam=a.lam, action=a.action, cot_in_target=a.cot_in_target,
        t_low=a.t_low, t_high=a.t_high, pred_layers=a.pred_layers,
        ema_decay=a.ema_decay, diff_noise=a.diff_noise,
        lr=a.lr, epochs=a.epochs, batch_size=a.batch_size, grad_accum=a.grad_accum,
        max_len=64 if a.smoke else a.max_len,
        max_cot_len=64 if a.smoke else Config.max_cot_len,
        use_lora=not a.no_lora and not a.smoke,
        grad_ckpt=Config.grad_ckpt and not a.smoke,
        trust_remote=not a.smoke,
        mask_token_id=50256 if a.smoke else a.mask_token_id,
        seed=a.seed, out_dir=a.out_dir, run_name=a.run_name, use_wandb=a.wandb,
        log_every=1 if a.smoke else Config.log_every,
    )
    train(cfg, smoke=a.smoke)
