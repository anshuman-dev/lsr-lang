#!/usr/bin/env python3
"""
Compare baseline VAE vs CLIP-VAE on latent space quality.

Metrics:
  1. k-NN classification accuracy from latent codes (5-class episode-progress bins).
  2. Planning success rate (loaded from eval JSON if available).
  3. Side-by-side latent space scatter plots saved to --out.

Example:
    python -m scripts.compare \
        --train data/raw/box_stacking_normal_task_2500.pkl \
        --val   data/raw/box_stacking_normal_task_holdout.pkl \
        --vae_ckpt results/vae/vae_best.pt \
        --clip_vae_ckpt results/clip_vae/clip_vae_best.pt \
        --clip_text_emb results/clip_vae/text_emb.npy \
        --z_dim 4 --out results
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from data.dataset import LSRGraphData
from models.clip_vae import ClipVAE
from models.vae import VAE


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode(model, graph_data, device, text_emb=None):
    imgs = torch.from_numpy(graph_data.images.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    mus = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(imgs), 256):
            batch = imgs[i:i+256].to(device)
            if text_emb is not None:
                t_emb = text_emb.expand(batch.size(0), -1)
                _, mu, _ = model.encode(batch, t_emb)
            else:
                _, mu, _ = model.encode(batch)
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


def degree_labels(graph_data: LSRGraphData, n_bins: int = 5) -> np.ndarray:
    """Bin node degree into n_bins buckets as classification target."""
    degree = np.zeros(graph_data.n_states, dtype=np.float32)
    for src, dst, _ in graph_data.transitions:
        degree[src] += 1
        degree[dst] += 1
    edges = np.percentile(degree, np.linspace(0, 100, n_bins + 1))
    labels = np.digitize(degree, edges[1:-1]).astype(int)
    return labels


def knn_accuracy(zs_tr, y_tr, zs_te, y_te, k=5) -> float:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(zs_tr, y_tr)
    return clf.score(zs_te, y_te)


def project_2d(zs):
    if zs.shape[1] == 2:
        return zs
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(zs)


def plot_side_by_side(zs_base, zs_clip, labels, transitions, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, zs, title in zip(axes, [zs_base, zs_clip], ["Baseline VAE", "CLIP-VAE"]):
        coords = project_2d(zs)
        for src, dst, _ in transitions[:300]:
            ax.plot([coords[src, 0], coords[dst, 0]],
                    [coords[src, 1], coords[dst, 1]],
                    c="lightgray", lw=0.5, alpha=0.35, zorder=1)
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=labels, cmap="tab10",
                        s=28, alpha=0.85, linewidths=0.4, edgecolors="white", zorder=2)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
        ax.grid(True, alpha=0.25)
    plt.colorbar(sc, ax=axes[1], label="Degree bucket")
    plt.suptitle("Latent space comparison — box stacking (normal)", fontsize=14)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def compare(args):
    device = get_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_gd = LSRGraphData(args.train)
    val_gd   = LSRGraphData(args.val)

    train_labels = degree_labels(train_gd)
    val_labels   = degree_labels(val_gd)

    # Baseline VAE
    vae = VAE(z_dim=args.z_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    zs_base_tr = encode(vae, train_gd, device)
    zs_base_te = encode(vae, val_gd,   device)

    # CLIP-VAE
    text_emb = torch.tensor(
        np.load(args.clip_text_emb), dtype=torch.float32
    ).to(device)
    clip_vae = ClipVAE(z_dim=args.z_dim).to(device)
    clip_vae.load_state_dict(torch.load(args.clip_vae_ckpt, map_location=device))
    zs_clip_tr = encode(clip_vae, train_gd, device, text_emb)
    zs_clip_te = encode(clip_vae, val_gd,   device, text_emb)

    acc_base = knn_accuracy(zs_base_tr, train_labels, zs_base_te, val_labels)
    acc_clip = knn_accuracy(zs_clip_tr, train_labels, zs_clip_te, val_labels)

    print("\n=== Comparison ===")
    print(f"k-NN accuracy — Baseline VAE : {acc_base:.3f}")
    print(f"k-NN accuracy — CLIP-VAE     : {acc_clip:.3f}")

    plan_results = {}
    for name, path in [("baseline", "results/eval/results.json"),
                        ("clip",     "results/eval_clip/results.json")]:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                plan_results[name] = json.load(f)["summary"]["planning_success_rate"]
            print(f"Planning success ({name}): {plan_results[name]:.3f}")

    results = {
        "knn_accuracy": {"baseline": acc_base, "clip": acc_clip},
        "planning_success": plan_results,
    }
    with open(out / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_side_by_side(
        zs_base_te, zs_clip_te,
        val_labels,
        val_gd.transitions,
        out / "latent_comparison.png",
    )
    print(f"Results → {out}/comparison.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",          required=True)
    parser.add_argument("--val",            required=True)
    parser.add_argument("--vae_ckpt",       required=True)
    parser.add_argument("--clip_vae_ckpt",  required=True)
    parser.add_argument("--clip_text_emb",  required=True)
    parser.add_argument("--z_dim",          type=int, default=4)
    parser.add_argument("--out",            default="results")
    args = parser.parse_args()
    compare(args)


if __name__ == "__main__":
    main()
