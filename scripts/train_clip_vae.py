#!/usr/bin/env python3
"""
Train the CLIP-conditioned VAE.

Example:
    python -m scripts.train_clip_vae \
        --train data/raw/box_stacking_normal_task_2500.pkl \
        --val   data/raw/box_stacking_normal_task_holdout.pkl \
        --z_dim 4 --epochs 100 --out results/clip_vae
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import LSRDataset, LSRGraphData
from models.clip_vae import ClipVAE, encode_text
from models.vae import elbo_loss


TASK_LABELS = [
    "stack the red box on the blue box",
    "place the object on top of the other",
    "move the block onto the stack",
]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_unique(model, graph_data, text_emb, device):
    imgs = torch.from_numpy(graph_data.images.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    mus = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(imgs), 256):
            batch = imgs[i:i+256].to(device)
            t_emb = text_emb.expand(batch.size(0), -1)
            _, mu, _ = model.encode(batch, t_emb)
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
    graph_data = LSRGraphData(args.train)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Encoding task labels with CLIP...")
    text_emb = encode_text(TASK_LABELS, device).mean(0, keepdim=True)  # (1, 512)
    np.save(out / "text_emb.npy", text_emb.cpu().numpy())

    model = ClipVAE(z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss = tr_recon = tr_kl = 0.0

        for obs_t, _, _ in train_loader:
            obs_t = obs_t.to(device)
            t_emb = text_emb.expand(obs_t.size(0), -1)
            recon, mu, logvar = model(obs_t, t_emb)
            loss, rl, kl = elbo_loss(recon, obs_t, mu, logvar, beta=args.beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss  += loss.item()
            tr_recon += rl.item()
            tr_kl    += kl.item()

        scheduler.step()
        n = len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_t, _, _ in val_loader:
                obs_t = obs_t.to(device)
                t_emb = text_emb.expand(obs_t.size(0), -1)
                recon, mu, logvar = model(obs_t, t_emb)
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
            torch.save(model.state_dict(), out / "clip_vae_best.pt")

        if epoch % args.save_every == 0:
            ckpt = ckpt_dir / f"clip_vae_epoch{epoch:04d}.pt"
            torch.save(model.state_dict(), ckpt)
            snap = encode_unique(model, graph_data, text_emb, device)
            np.save(ckpt_dir / f"latents_epoch{epoch:04d}.npy", snap)

    torch.save(model.state_dict(), out / "clip_vae_final.pt")
    with open(out / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(torch.load(out / "clip_vae_best.pt", map_location=device))
    all_zs = encode_unique(model, graph_data, text_emb, device)
    np.save(out / "train_latents.npy", all_zs)

    episodes, actions = graph_data.episodes_and_actions()
    np.save(out / "train_episodes.npy", np.array(episodes, dtype=object), allow_pickle=True)
    np.save(out / "train_actions.npy",  actions)
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
    parser.add_argument("--out",        default="results/clip_vae")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
