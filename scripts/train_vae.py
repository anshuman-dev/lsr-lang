#!/usr/bin/env python3
"""
Train the baseline convolutional VAE.

Example:
    python scripts/train_vae.py \
        --train data/raw/box_stacking_normal_task_2500.pkl \
        --val   data/raw/box_stacking_normal_task_holdout.pkl \
        --z_dim 4 --epochs 100 --out results/vae
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import LSRDataset, LSRGraphData
from models.vae import VAE, elbo_loss


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_unique_states(vae: VAE, graph_data: LSRGraphData, device: torch.device) -> np.ndarray:
    """Encode all unique state images → (N_states, z_dim) mu array."""
    vae.eval()
    imgs = graph_data.images                        # (N, H, W, C) uint8
    imgs_t = torch.from_numpy(imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    mus = []
    with torch.no_grad():
        for i in range(0, len(imgs_t), 256):
            batch = imgs_t[i:i+256].to(device)
            _, mu, _ = vae.encode(batch)
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


def train(args):
    device = get_device()
    print(f"Device: {device}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    train_ds = LSRDataset(args.train)
    val_ds   = LSRDataset(args.val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Also load unique-state graph data for latent snapshots + final LSR export
    graph_data = LSRGraphData(args.train)
    print(f"Unique states: {graph_data.n_states}  |  Transitions: {len(graph_data.transitions)}")

    vae = VAE(z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        vae.train()
        t0 = time.time()
        tr_loss = tr_recon = tr_kl = 0.0

        for obs_t, _, _ in train_loader:
            obs_t = obs_t.to(device)
            recon, mu, logvar = vae(obs_t)
            loss, recon_l, kl_l = elbo_loss(recon, obs_t, mu, logvar, beta=args.beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            tr_loss  += loss.item()
            tr_recon += recon_l.item()
            tr_kl    += kl_l.item()

        scheduler.step()
        n = len(train_loader)

        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_t, _, _ in val_loader:
                obs_t = obs_t.to(device)
                recon, mu, logvar = vae(obs_t)
                l, _, _ = elbo_loss(recon, obs_t, mu, logvar, beta=args.beta)
                val_loss += l.item()
        val_loss /= len(val_loader)

        log = {
            "epoch": epoch,
            "train_loss": tr_loss / n,
            "train_recon": tr_recon / n,
            "train_kl": tr_kl / n,
            "val_loss": val_loss,
        }
        history.append(log)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={log['train_loss']:.4f} "
            f"(recon={log['train_recon']:.4f} kl={log['train_kl']:.4f}) | "
            f"val={val_loss:.4f} | {time.time()-t0:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(vae.state_dict(), out / "vae_best.pt")

        if epoch % args.save_every == 0:
            ckpt = ckpt_dir / f"vae_epoch{epoch:04d}.pt"
            torch.save(vae.state_dict(), ckpt)
            snap_zs = encode_unique_states(vae, graph_data, device)
            np.save(ckpt_dir / f"latents_epoch{epoch:04d}.npy", snap_zs)

    torch.save(vae.state_dict(), out / "vae_final.pt")
    with open(out / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save unique-state latents for LSR graph building
    print("Encoding unique states with best checkpoint...")
    vae.load_state_dict(torch.load(out / "vae_best.pt", map_location=device))
    all_zs = encode_unique_states(vae, graph_data, device)
    np.save(out / "train_latents.npy", all_zs)

    episodes, actions = graph_data.episodes_and_actions()
    np.save(out / "train_episodes.npy",  np.array(episodes, dtype=object), allow_pickle=True)
    np.save(out / "train_actions.npy",   actions)
    print(f"Saved train_latents.npy  shape={all_zs.shape}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      required=True)
    parser.add_argument("--val",        required=True)
    parser.add_argument("--z_dim",      type=int,   default=4)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--beta",       type=float, default=1.0)
    parser.add_argument("--save_every", type=int,   default=10)
    parser.add_argument("--out",        default="results/vae")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
