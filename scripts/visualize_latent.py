#!/usr/bin/env python3
"""
Visualize the learned latent space over unique training states.

Example:
    python -m scripts.visualize_latent \
        --train data/raw/box_stacking_normal_task_2500.pkl \
        --vae_ckpt results/vae/vae_best.pt \
        --out results/latent_space_baseline.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset import LSRGraphData
from models.vae import VAE


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_states(vae: VAE, graph_data: LSRGraphData, device: torch.device) -> np.ndarray:
    imgs = torch.from_numpy(graph_data.images.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    mus = []
    vae.eval()
    with torch.no_grad():
        for i in range(0, len(imgs), 256):
            _, mu, _ = vae.encode(imgs[i:i+256].to(device))
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


def project_2d(zs: np.ndarray) -> np.ndarray:
    if zs.shape[1] == 2:
        return zs
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(zs)


def visualize(args):
    device = get_device()

    vae = VAE(z_dim=args.z_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))

    graph_data = LSRGraphData(args.train)
    zs = encode_states(vae, graph_data, device)
    coords = project_2d(zs)

    # Colour by node connectivity (degree in transition graph) as a proxy
    # for how "central" each state is in the task graph
    degree = np.zeros(len(zs), dtype=np.float32)
    for src, dst, _ in graph_data.transitions:
        degree[src] += 1
        degree[dst] += 1

    fig, ax = plt.subplots(figsize=(8, 7))

    # Draw edges
    for src, dst, _ in graph_data.transitions[:300]:  # subsample for readability
        ax.plot(
            [coords[src, 0], coords[dst, 0]],
            [coords[src, 1], coords[dst, 1]],
            c="lightgray", linewidth=0.5, alpha=0.4, zorder=1,
        )

    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=degree, cmap="plasma",
        s=30, alpha=0.85, linewidths=0.5, edgecolors="white", zorder=2,
    )
    plt.colorbar(sc, ax=ax, label="Node degree (number of transitions)")
    method = "PCA" if zs.shape[1] > 2 else "direct"
    ax.set_title(
        f"Latent space — {Path(args.vae_ckpt).parent.name}\n"
        f"288 unique states, 1629 transitions ({method} projection)",
        fontsize=11,
    )
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.grid(True, alpha=0.25)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--z_dim",    type=int, default=4)
    parser.add_argument("--out",      default="results/latent_space_baseline.png")
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
