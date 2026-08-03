"""
LLM-JEPA from-scratch pretraining on NL-RX-SYNTH — reproduction + the missing control.

Method (arXiv 2509.14252, verified against the paper):

    L = L_NTP(sequence)  +  λ · d( Pred(Enc(Text)) , Enc(Code) )

    Enc(x)   = hidden state of the LAST token, LAST layer
    Pred(·)  = the LLM itself with k [PRED] tokens appended to the input; the
               last [PRED] token's final-layer hidden state is the prediction
               (LLM-JEPA reuses the LLM's internal weights as the predictor)
    d        = cosine distance, 1 − cos
    λ        ∈ {0.5, 1, 2, 4}
    NO stop-gradient and NO EMA — both views carry gradients (collapse is
    prevented by the NTP term, not by a teacher).

Arms:
    B0  --lam 0                      baseline, NTP only          (paper: 54.38)
    B1  --lam 1                      LLM-JEPA                    (paper: 60.59)
    B2  --lam 1 --shuffle-pairs      THE MISSING CONTROL: the JEPA term pairs
                                     each description with a RANDOM OTHER
                                     example's regex. The NTP term is untouched,
                                     so only the JEPA term's correspondence is
                                     destroyed. If B2 ≈ B1, the reported gain is
                                     not coming from the semantic pairing.

Run:  python ft/train_llmjepa.py --lam 1 --run-name B1_jepa --wandb
      python ft/train_llmjepa.py --smoke        # tiny config, no GPU needed
"""

import argparse, math, os, sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlrx import download, splits, NLRXDataset, collate, SEP

PRED_TOKEN = "[PRED]"


@torch.no_grad()
def quick_eval(model, tok, pairs, device, n=200, max_new=48, bs=32):
    """Greedy exact-match accuracy on a held-out slice, run during training.

    Worth the cost here: a 1B model on 6.5k examples overfits heavily, so the
    VALIDATION CURVE is itself evidence about the mechanism — if the JEPA term's
    benefit is regularisation rather than semantic pairing, the shuffled arm's
    curve should track the real one.
    """
    model.eval()
    was_cache, was_side = model.config.use_cache, tok.padding_side
    model.config.use_cache, tok.padding_side = True, "left"
    data = pairs[:n]
    correct = 0
    for i in range(0, len(data), bs):
        chunk = data[i: i + bs]
        enc = tok([nl + SEP for nl, _ in chunk], return_tensors="pt",
                  padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        correct += sum(g.strip().split("\n")[0].strip() == gold
                       for (_, gold), g in zip(chunk, gen))
    model.config.use_cache, tok.padding_side = was_cache, was_side
    model.train()
    return correct / max(1, len(data))


def build_model(model_id, n_pred_tokens, smoke=False, device="cpu"):
    """Llama-3.2-1B architecture, RANDOMLY INITIALISED (the paper pretrains from
    random weights). Default id is a non-gated mirror so no HF token is needed."""
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": [PRED_TOKEN]})

    cfg = AutoConfig.from_pretrained(model_id)
    if smoke:                                   # tiny stand-in for plumbing tests
        cfg.num_hidden_layers, cfg.hidden_size = 2, 128
        cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
        cfg.intermediate_size = 256
    model = AutoModelForCausalLM.from_config(cfg)      # ← random init, not pretrained
    model.resize_token_embeddings(len(tok))
    return tok, model


def encode_last(model, ids, real):
    """Enc(x) = last REAL token's final-layer hidden state. (B,L)→(B,D)."""
    out = model(input_ids=ids, attention_mask=real.long(), output_hidden_states=True)
    h = out.hidden_states[-1]                                   # (B, L, D)
    last = real.long().sum(1).clamp(min=1) - 1                  # index of last real token
    return h[torch.arange(h.shape[0], device=h.device), last]   # (B, D)


def train(a):
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = build_model(a.model, a.n_pred_tokens, a.smoke, device)
    dtype = torch.bfloat16 if (device == "cuda" and not a.fp32) else torch.float32
    model = model.to(device=device, dtype=dtype)
    if a.grad_ckpt and not a.smoke:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} dtype={dtype} params={n_params/1e6:.0f}M "
          f"λ={a.lam} shuffle={a.shuffle_pairs} k_pred={a.n_pred_tokens}")

    train_pairs, val_pairs, _ = splits(download(a.data_cache))
    if a.smoke:
        train_pairs, val_pairs = train_pairs[:16], val_pairs[:4]
    ds = NLRXDataset(train_pairs, tok, a.max_len, a.n_pred_tokens, PRED_TOKEN)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                    drop_last=True)
    print(f"train examples: {len(ds)}  steps/epoch: {len(dl)}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay,
                            betas=(0.9, 0.95))
    total = max(1, len(dl) * a.epochs)
    warm = max(1, int(total * 0.03))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    if a.wandb:
        import wandb
        from dotenv import load_dotenv
        load_dotenv(); wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(entity="austinmyc", project="lejepa-ft", name=a.run_name,
                   config=vars(a) | {"params": n_params})

    step = 0
    for ep in range(a.epochs):
        for b in dl:
            seq, seq_real = b["seq_ids"].to(device), b["seq_real"].to(device)
            sup = b["sup_mask"].to(device)

            # ── L_NTP: predict the regex half of the sequence.
            logits = model(input_ids=seq, attention_mask=seq_real.long()).logits
            tgt_pos = sup[:, 1:]                                # shift for next-token
            if tgt_pos.any():
                l_ntp = F.cross_entropy(logits[:, :-1][tgt_pos].float(), seq[:, 1:][tgt_pos])
            else:
                l_ntp = logits.sum() * 0.0

            # ── L_JEPA: cosine distance between the predictor's output and the
            # other view's embedding. No stop-grad, no EMA (per the paper).
            l_jepa = torch.zeros((), device=device)
            if a.lam > 0:
                text_ids, text_real = b["text_ids"].to(device), b["text_real"].to(device)
                code_ids, code_real = b["code_ids"].to(device), b["code_real"].to(device)
                if a.shuffle_pairs:      # ← the control: wrong regex for each description
                    perm = torch.randperm(code_ids.shape[0], device=device)
                    code_ids, code_real = code_ids[perm], code_real[perm]
                z_pred = encode_last(model, text_ids, text_real)   # Pred(Enc(Text))
                z_code = encode_last(model, code_ids, code_real)   # Enc(Code)
                l_jepa = (1 - F.cosine_similarity(z_pred.float(), z_code.float(), -1)).mean()

            loss = (l_ntp + a.lam * l_jepa) / a.grad_accum
            loss.backward()

            if (step + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            sched.step(); step += 1

            if step % a.log_every == 0:
                print(f"ep{ep} step {step}/{total} | ntp {l_ntp.item():.4f} | "
                      f"jepa {l_jepa.item():.4f} | lr {sched.get_last_lr()[0]:.2e}")
                if a.wandb:
                    wandb.log({"l_ntp": l_ntp.item(), "l_jepa": l_jepa.item(),
                               "lr": sched.get_last_lr()[0]}, step=step)
            if a.smoke and step >= 4:
                break

        # ── validation accuracy each epoch (the overfitting/regularisation curve)
        if a.eval_every and (ep + 1) % a.eval_every == 0:
            acc = quick_eval(model, tok, val_pairs, device, n=a.eval_n,
                             bs=a.batch_size)
            print(f"ep{ep} VAL exact-match {acc:.4f}")
            if a.wandb:
                wandb.log({"val/exact": acc, "epoch": ep + 1}, step=step)
        if a.smoke:
            break

    out = os.path.join(a.out_dir, a.run_name)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    print(f"saved → {out}")
    if a.wandb:
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LLM-JEPA from-scratch NL-RX reproduction.")
    p.add_argument("--model", default="unsloth/Llama-3.2-1B",
                   help="Architecture source (config+tokenizer only; weights are RANDOM). "
                        "Non-gated mirror of meta-llama/Llama-3.2-1B.")
    p.add_argument("--lam", type=float, default=1.0, help="JEPA weight; 0 = baseline.")
    p.add_argument("--shuffle-pairs", action="store_true",
                   help="THE CONTROL: pair each description with a random other regex "
                        "in the JEPA term only.")
    p.add_argument("--n-pred-tokens", type=int, default=1, help="k [PRED] tokens.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--data-cache", default="./data_cache/nlrx")
    p.add_argument("--out-dir", default="./ft_checkpoints")
    p.add_argument("--run-name", default="llmjepa_nlrx")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2,
                   help="Run validation accuracy every N epochs (0 = off).")
    p.add_argument("--eval-n", type=int, default=200, help="Validation subset size.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.log_every, a.eval_every, a.eval_n = 1, 1, 4
    train(a)
