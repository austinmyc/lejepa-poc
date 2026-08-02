"""
GSM8K eval for masked-diffusion LMs: iterative unmasking + accuracy.

Generation follows the standard LLaDA sampler (the paper: 128 diffusion steps,
greedy): start from a fully-masked response block, and at each step decode all
positions greedily but only KEEP the most-confident fraction, re-masking the
rest for the next step (low-confidence remasking).

    python ft/eval_gsm8k.py --model GSAI-ML/LLaDA-8B-Instruct \
        --adapter ft_checkpoints/A1_jepa --n-shot 4 --eval-n 200 --wandb
"""

import argparse, os, re, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from data   import GSM8KDataset, split_answer, few_shot_prefix

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_pred(text):
    """Prefer the '#### x' form; else the last number produced."""
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = NUM_RE.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else ""


def equal(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (ValueError, TypeError):
        return False


@torch.no_grad()
def generate(model, tok, prompt_ids, mask_id, gen_len, steps, device):
    """Iterative unmasking with low-confidence remasking. prompt_ids: (1, P)."""
    P = prompt_ids.shape[1]
    x = torch.cat([prompt_ids,
                   torch.full((1, gen_len), mask_id, device=device, dtype=torch.long)], 1)
    steps = max(1, min(steps, gen_len))
    for i in range(steps):
        masked = x[0, P:] == mask_id
        if not masked.any():
            break
        logits = model(input_ids=x).logits[0, P:]                 # (gen_len, V)
        probs = F.softmax(logits.float(), -1)
        conf, pred = probs.max(-1)
        conf = conf.masked_fill(~masked, -1.0)                    # only fill masked
        # Unmask a growing fraction so all positions are decided by the last step.
        k = max(1, int(masked.sum().item() / (steps - i)))
        idx = conf.topk(k).indices
        row = x[0, P:]
        row[idx] = pred[idx]
        x[0, P:] = row
    return tok.decode(x[0, P:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser(description="GSM8K accuracy for a (finetuned) diffusion LM.")
    p.add_argument("--model", default=Config.model_name)
    p.add_argument("--adapter", default="", help="LoRA dir from train.py ('' = base model).")
    p.add_argument("--n-shot", type=int, default=Config.n_shot)
    p.add_argument("--eval-n", type=int, default=200, help="0 = full test set.")
    p.add_argument("--gen-len", type=int, default=Config.gen_len)
    p.add_argument("--gen-steps", type=int, default=Config.gen_steps)
    p.add_argument("--mask-token-id", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run-name", default="")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()

    mask_id = a.mask_token_id or getattr(tok, "mask_token_id", None) \
              or getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("Pass --mask-token-id (LLaDA-8B uses 126336).")

    from datasets import load_dataset
    test = load_dataset(Config.dataset, Config.dataset_config, split="test")
    train = load_dataset(Config.dataset, Config.dataset_config, split="train")
    n = a.eval_n or len(test)
    prefix = few_shot_prefix(train, a.n_shot, with_cot=True)

    correct = 0
    for i in range(n):
        row = test[i]
        _, gold = split_answer(row["answer"])
        prompt = prefix + GSM8KDataset.PROMPT.format(q=row["question"].strip())
        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device=device)
        out = generate(model, tok, ids, mask_id, a.gen_len, a.gen_steps, device)
        correct += equal(extract_pred(out), gold)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n}] running acc {correct/(i+1):.4f}")

    acc = correct / n
    print(f"GSM8K {a.n_shot}-shot accuracy: {acc:.4f}  ({correct}/{n})  adapter={a.adapter or 'base'}")

    if a.wandb:
        import wandb
        from dotenv import load_dotenv
        load_dotenv(); wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(entity=Config.wandb_entity, project=Config.wandb_project,
                   name=a.run_name or f"eval_{os.path.basename(a.adapter) or 'base'}",
                   config={"adapter": a.adapter, "n_shot": a.n_shot, "n": n})
        wandb.log({"gsm8k/acc": acc, "gsm8k/n": n})
        wandb.finish()


if __name__ == "__main__":
    main()
