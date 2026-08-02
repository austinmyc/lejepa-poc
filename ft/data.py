"""
GSM8K loading for DLLM-JEPA finetuning.

GSM8K's `answer` field is a chain-of-thought followed by "#### <final>". We split
it into (cot, final) so the CoT can be used three ways:
  - as the predictor's ACTION (A2/A3, never in the generation context),
  - not at all (A0/A1),
  - inside the generation target (A5 = plain CoT-SFT, the honest comparator).

Each example yields token ids plus a `resp_mask` marking the response span —
only response tokens are ever noised by the diffusion loss (LLaDA SFT
convention: the prompt is always clean).
"""

import re
import torch
from torch.utils.data import Dataset

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")


def split_answer(ans: str):
    """GSM8K answer → (cot_text, final_answer_string)."""
    m = ANS_RE.search(ans)
    final = m.group(1).strip().replace(",", "") if m else ""
    cot = ANS_RE.split(ans)[0].strip()
    # Drop GSM8K's calculator annotations <<...>> — they are not natural CoT.
    cot = re.sub(r"<<[^>]*>>", "", cot).strip()
    return cot, final


class GSM8KDataset(Dataset):
    """
    Returns per example:
        input_ids  (L,)   prompt + response, right-padded
        resp_mask  (L,)   True on response tokens (the maskable span)
        pad_mask   (L,)   True on padding
        cot_ids    (C,)   tokenised CoT (the action input), right-padded
        cot_pad    (C,)   True on padding
        final      str    gold final answer (eval only)
    """

    PROMPT = "Question: {q}\nAnswer:"

    def __init__(self, tokenizer, cfg, split="train"):
        from datasets import load_dataset
        self.tok = tokenizer
        self.cfg = cfg
        self.rows = load_dataset(cfg.dataset, cfg.dataset_config, split=split)

    def __len__(self):
        return len(self.rows)

    def _pad(self, ids, n, pad_id):
        ids = ids[:n]
        k = n - len(ids)
        return (torch.tensor(ids + [pad_id] * k, dtype=torch.long),
                torch.tensor([False] * len(ids) + [True] * k))

    def __getitem__(self, i):
        row = self.rows[i]
        cot, final = split_answer(row["answer"])
        prompt = self.PROMPT.format(q=row["question"].strip())

        # A5 (cot_in_target): the CoT is part of what the model must generate.
        # A0–A4: the response is the final answer only; the CoT is routed to the
        # predictor's action input and never appears in the generation context.
        response = f" {cot}\n#### {final}" if self.cfg.cot_in_target else f" {final}"

        p_ids = self.tok.encode(prompt, add_special_tokens=False)
        r_ids = self.tok.encode(response, add_special_tokens=False)[: self.cfg.max_len - 1]
        # The response must survive truncation — it is the only maskable span, so
        # a right-truncated example would contribute ZERO diffusion loss. Keep the
        # response whole and left-truncate the prompt (keep its tail: the question
        # end + "Answer:" is what conditions generation).
        room = self.cfg.max_len - len(r_ids)
        if len(p_ids) > room:
            p_ids = p_ids[-room:]
        ids = p_ids + r_ids

        pad_id = self.tok.pad_token_id or 0
        input_ids, pad_mask = self._pad(ids, self.cfg.max_len, pad_id)
        resp = ([False] * len(p_ids) + [True] * len(r_ids))[: self.cfg.max_len]
        resp_mask = torch.tensor(resp + [False] * (self.cfg.max_len - len(resp)))

        cot_ids, cot_pad = self._pad(
            self.tok.encode(cot, add_special_tokens=False), self.cfg.max_cot_len, pad_id)

        return {"input_ids": input_ids, "resp_mask": resp_mask, "pad_mask": pad_mask,
                "cot_ids": cot_ids, "cot_pad": cot_pad, "final": final}


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch])
           for k in ["input_ids", "resp_mask", "pad_mask", "cot_ids", "cot_pad"]}
    out["final"] = [b["final"] for b in batch]
    return out


def few_shot_prefix(train_rows, k, with_cot=True):
    """k-shot prompt prefix (the paper's primary metric is 4-shot)."""
    parts = []
    for r in list(train_rows)[:k]:
        cot, final = split_answer(r["answer"])
        body = f" {cot}\n#### {final}" if with_cot else f" {final}"
        parts.append(GSM8KDataset.PROMPT.format(q=r["question"].strip()) + body)
    return "\n\n".join(parts) + "\n\n" if parts else ""
