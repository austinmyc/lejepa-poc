"""
Nearest-neighbour evaluation for supervisor demo.

Loads a checkpoint, encodes a set of test sentences, and shows
the top-3 nearest neighbours for each query.

Eval read-out (§3b of research plan):
    encoder(x) → (1, L, D) → mean over L → (1, D) sequence embedding
    The predictor is NOT used at eval time.

Usage:
    python eval_neighbors.py --ckpt checkpoints/ckpt_050000.pt
    python eval_neighbors.py --ckpt checkpoints/ckpt_050000.pt --umap
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from config import Config
from model  import LeJEPAText


# ── test sentences ────────────────────────────────────────────────────────────

TEST_SENTENCES = [
    # science
    "The researchers published their findings in Nature.",
    "Scientists announced a breakthrough in quantum computing.",
    "The study revealed new insights into protein folding mechanisms.",
    # sports
    "The team won the championship after a dramatic final match.",
    "The player scored three goals in the last ten minutes.",
    "Fans celebrated in the streets after the historic victory.",
    # cooking
    "Preheat the oven to 180 degrees and grease the baking tin.",
    "Add the flour and butter and mix until the dough is smooth.",
    "Simmer the sauce on low heat for twenty minutes until thick.",
    # politics
    "The parliament voted to approve the new budget proposal.",
    "The prime minister addressed the nation about the economic crisis.",
    "Negotiations between the two parties collapsed over taxation.",
    # random / unrelated
    "The cat sat quietly on the warm windowsill.",
    "Heavy rainfall is expected across the northern regions tomorrow.",
]


# ── encoding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_sentences(sentences, model, tokenizer, cfg, device):
    """
    Encode each sentence to a single embedding via mean-pooling over tokens.

    Sentences are truncated / padded to cfg.seq_len tokens.
    Returns (N, D) tensor.
    """
    model.eval()
    embs = []
    for sent in sentences:
        ids = tokenizer.encode(sent)[:cfg.seq_len]
        if len(ids) < cfg.seq_len:
            ids = ids + [tokenizer.eos_token_id] * (cfg.seq_len - len(ids))

        x   = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
        out = model.encoder(x)   # (1, L, D)
        emb = out.mean(dim=1)    # (1, D)  mean-pool over tokens
        embs.append(emb.squeeze(0))

    return torch.stack(embs)   # (N, D)


# ── nearest neighbours ────────────────────────────────────────────────────────

def nearest_neighbours(embs, sentences, top_k=3):
    normed = F.normalize(embs.float(), dim=-1)
    sim    = normed @ normed.T
    print("\n" + "=" * 70)
    print("NEAREST NEIGHBOURS (cosine similarity)")
    print("=" * 70)
    for i, sent in enumerate(sentences):
        scores = sim[i].clone()
        scores[i] = -1.0   # exclude self
        top_idx = scores.topk(top_k).indices.tolist()
        print(f"\nQuery: {sent!r}")
        for rank, j in enumerate(top_idx, 1):
            print(f"  {rank}. [{sim[i][j]:.3f}]  {sentences[j]!r}")
    print("=" * 70)


# ── UMAP plot ─────────────────────────────────────────────────────────────────

def plot_umap(embs, sentences):
    try:
        import umap
        import matplotlib.pyplot as plt
    except ImportError:
        print("UMAP plot skipped — install umap-learn and matplotlib")
        return

    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    coords  = reducer.fit_transform(embs.float().cpu().numpy())

    colours = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    labels  = ["science", "sports", "cooking", "politics", "misc"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for cat_i in range(len(labels)):
        start = cat_i * 3
        end   = start + 3
        ax.scatter(coords[start:end, 0], coords[start:end, 1],
                   c=colours[cat_i], label=labels[cat_i], s=100, zorder=3)
        for j in range(start, min(end, len(sentences))):
            ax.annotate(sentences[j][:30] + "…", coords[j], fontsize=6, alpha=0.7)

    ax.legend()
    ax.set_title("UMAP of LeJEPA token embeddings (mean-pooled)")
    plt.tight_layout()
    plt.savefig("umap_embeddings.png", dpi=150)
    print("UMAP plot saved → umap_embeddings.png")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--umap", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt  = torch.load(args.ckpt, map_location=device)
    cfg   = ckpt["config"]
    model = LeJEPAText(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from step {ckpt['step']}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    embs      = encode_sentences(TEST_SENTENCES, model, tokenizer, cfg, device)
    nearest_neighbours(embs, TEST_SENTENCES)

    if args.umap:
        plot_umap(embs, TEST_SENTENCES)
