"""
NL-RX-SYNTH eval: generate a regex from each description, score exact match.

    python ft/eval_nlrx.py --ckpt ft_checkpoints/B1_jepa --split test --wandb

Note on the metric: Locascio et al. score DFA-equivalence (semantically identical
regexes count as correct). Exact string match is a strict lower bound on that;
`--dfa` additionally accepts regexes that behave identically on a random string
sample, which approximates DFA-equivalence without a DFA library.
"""

import argparse, os, random, re, string, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlrx import download, splits, SEP


def behavioural_match(a, b, n=200, seed=0):
    """
    Approximate DFA-equivalence: do the two regexes accept the same strings?

    Guard against DEGENERATE agreement — a garbage prediction that matches
    nothing trivially "agrees" with a gold regex that also matches none of the
    sample. We therefore require the gold regex to accept at least one sampled
    string; if it accepts none, the test is uninformative and we fall back to
    exact string equality.
    """
    try:
        ra, rb = re.compile(a), re.compile(b)
    except re.error:
        return False
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + " .,-"
    agree, gold_hits = True, 0
    for _ in range(n):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
        ma, mb = bool(ra.fullmatch(s)), bool(rb.fullmatch(s))
        gold_hits += mb
        if ma != mb:
            agree = False
            break
    if gold_hits == 0:                       # uninformative sample
        return a == b
    return agree


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--eval-n", type=int, default=0, help="0 = whole split.")
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dfa", action="store_true", help="Also count behavioural matches.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run-name", default="")
    a = p.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        a.ckpt, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device).eval()
    model.config.use_cache = True
    tok.padding_side = "left"                       # correct batched generation

    tr, va, te = splits(download())
    data = {"train": tr, "val": va, "test": te}[a.split]
    if a.eval_n:
        data = data[: a.eval_n]

    exact = behav = 0
    for i in range(0, len(data), a.batch_size):
        chunk = data[i: i + a.batch_size]
        prompts = [nl + SEP for nl, _ in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        for (nl, gold), g in zip(chunk, gen):
            pred = g.strip().split("\n")[0].strip()
            if pred == gold:
                exact += 1; behav += 1
            elif a.dfa and behavioural_match(pred, gold):
                behav += 1
        done = min(i + a.batch_size, len(data))
        if done % (a.batch_size * 10) == 0:
            print(f"  [{done}/{len(data)}] exact {exact/done:.4f}")

    n = len(data)
    print(f"NL-RX {a.split}: exact-match {exact/n:.4f} ({exact}/{n})"
          + (f" | behavioural {behav/n:.4f}" if a.dfa else ""))

    if a.wandb:
        import wandb
        from dotenv import load_dotenv
        load_dotenv(); wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(entity="austinmyc", project="lejepa-ft",
                   name=a.run_name or f"eval_{os.path.basename(a.ckpt)}",
                   config={"ckpt": a.ckpt, "split": a.split, "n": n})
        wandb.log({"nlrx/exact": exact / n, "nlrx/behavioural": behav / n})
        wandb.finish()


if __name__ == "__main__":
    main()
