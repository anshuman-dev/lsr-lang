#!/usr/bin/env python3
"""
Create a training-evolution GIF showing how the latent space develops.

Reads epoch-snapshot .npy files saved by train_vae.py or train_clip_vae.py
and stitches them into an animated GIF.

Example:
    python scripts/animate.py \
        --snapshot_dir results/vae/checkpoints \
        --data data/raw/box_stacking_normal.pkl \
        --out results/training_evolution.gif
"""

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data.dataset import LSRGraphData


def get_progress_labels(n_states: int) -> np.ndarray:
    # Simple sequential colouring — unique states don't have a canonical order,
    # so we just colour by index as a visual variety proxy.
    return np.linspace(0, 1, n_states, dtype=np.float32)


def project_2d(zs: np.ndarray) -> np.ndarray:
    if zs.shape[1] == 2:
        return zs
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(zs)


def render_frame(zs: np.ndarray, progress: np.ndarray, epoch: int, total: int) -> Image.Image:
    coords = project_2d(zs)

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=progress, cmap="viridis",
        s=4, alpha=0.7, linewidths=0,
    )
    ax.set_title(f"Epoch {epoch}/{total}", fontsize=11)
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(coords[:, 0].min() - 0.5, coords[:, 0].max() + 0.5)
    ax.set_ylim(coords[:, 1].min() - 0.5, coords[:, 1].max() + 0.5)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P")


def animate(args):
    snap_dir = Path(args.snapshot_dir)
    snaps = sorted(snap_dir.glob("latents_epoch*.npy"))
    if not snaps:
        raise FileNotFoundError(f"No latent snapshots found in {snap_dir}")

    gd = LSRGraphData(args.data)
    progress = get_progress_labels(gd.n_states)

    # Infer total epochs from last snapshot filename
    last_epoch = int(snaps[-1].stem.split("epoch")[-1])

    print(f"Rendering {len(snaps)} frames...")
    frames = []
    for snap in snaps:
        epoch = int(snap.stem.split("epoch")[-1])
        zs = np.load(snap)
        # Trim if mismatch (e.g. different shuffle order)
        n = min(len(zs), len(progress))
        if n == 0:
            continue
        frame = render_frame(zs[:n], progress[:n], epoch, last_epoch)
        frames.append(frame)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out),
        save_all=True,
        append_images=frames[1:],
        duration=args.frame_ms,
        loop=0,
    )
    print(f"Saved {out}  ({len(frames)} frames, {args.frame_ms}ms/frame)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_dir", required=True)
    parser.add_argument("--data", "--train", dest="data", required=True)
    parser.add_argument("--out",          default="results/training_evolution.gif")
    parser.add_argument("--frame_ms",     type=int, default=300,
                        help="Milliseconds per frame")
    args = parser.parse_args()
    animate(args)


if __name__ == "__main__":
    main()
