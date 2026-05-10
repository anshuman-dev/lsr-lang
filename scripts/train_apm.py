#!/usr/bin/env python3
"""
Train the Action Proposal Module (APM).

Example:
    python -m scripts.train_apm \
        --train data/raw/box_stacking_normal_task_2500.pkl \
        --val   data/raw/box_stacking_normal_task_holdout.pkl \
        --vae_ckpt results/vae/vae_best.pt \
        --z_dim 4 --action_dim 4 \
        --out results/apm
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import LSRDataset
from models.apm import APM, apm_loss
from models.vae import VAE


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_latent_dataset(
    vae: VAE,
    ds: LSRDataset,
    device: torch.device,
    batch_size: int = 256,
) -> TensorDataset:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    zs_t, zs_next, acts = [], [], []
    vae.eval()
    with torch.no_grad():
        for obs_t, obs_next, action in loader:
            _, mu_t,    _ = vae.encode(obs_t.to(device))
            _, mu_next, _ = vae.encode(obs_next.to(device))
            zs_t.append(mu_t.cpu())
            zs_next.append(mu_next.cpu())
            acts.append(action)
    return TensorDataset(torch.cat(zs_t), torch.cat(zs_next), torch.cat(acts))


def train(args):
    device = get_device()
    print(f"Device: {device}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vae = VAE(z_dim=args.z_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    train_ds = LSRDataset(args.train)
    val_ds   = LSRDataset(args.val)

    print("Encoding training set...")
    train_lat = build_latent_dataset(vae, train_ds, device)
    val_lat   = build_latent_dataset(vae, val_ds,   device)

    train_loader = DataLoader(train_lat, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_lat,   batch_size=args.batch_size, shuffle=False)

    apm = APM(z_dim=args.z_dim, action_dim=args.action_dim, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(apm.parameters(), lr=args.lr)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        apm.train()
        t0 = time.time()
        tr_loss = 0.0
        for z_t, z_next, action in train_loader:
            z_t, z_next, action = z_t.to(device), z_next.to(device), action.to(device)
            loss = apm_loss(apm(z_t, z_next), action)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        apm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z_t, z_next, action in val_loader:
                z_t, z_next, action = z_t.to(device), z_next.to(device), action.to(device)
                val_loss += apm_loss(apm(z_t, z_next), action).item()

        log = {
            "epoch": epoch,
            "train_loss": tr_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
        }
        history.append(log)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={log['train_loss']:.5f} | val={log['val_loss']:.5f} | {time.time()-t0:.1f}s"
        )

        if log["val_loss"] < best_val:
            best_val = log["val_loss"]
            torch.save(apm.state_dict(), out / "apm_best.pt")

    torch.save(apm.state_dict(), out / "apm_final.pt")
    with open(out / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      required=True)
    parser.add_argument("--val",        required=True)
    parser.add_argument("--vae_ckpt",   required=True)
    parser.add_argument("--z_dim",      type=int,   default=4)
    parser.add_argument("--action_dim", type=int,   default=4)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--out",        default="results/apm")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
