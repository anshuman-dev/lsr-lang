#!/usr/bin/env python3
"""
Evaluate planning success rate.

Pipeline:
  1. Load VAE + APM.
  2. Encode unique training states → build LSR graph.
  3. For each test episode's (start, goal) pair, plan and check success.

Example:
    python -m scripts.evaluate \
        --train data/raw/box_stacking_normal_task_2500.pkl \
        --val   data/raw/box_stacking_normal_task_holdout.pkl \
        --vae_ckpt results/vae/vae_best.pt \
        --apm_ckpt results/apm/apm_best.pt \
        --out results/eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from data.dataset import LSRDataset, LSRGraphData
from models.apm import APM
from models.lsr import LSR
from models.vae import VAE


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_unique(vae: VAE, graph_data: LSRGraphData, device: torch.device) -> np.ndarray:
    imgs = torch.from_numpy(graph_data.images.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    mus = []
    vae.eval()
    with torch.no_grad():
        for i in range(0, len(imgs), 256):
            _, mu, _ = vae.encode(imgs[i:i+256].to(device))
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


def evaluate(args):
    device = get_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plans_dir = out / "planning_examples"
    plans_dir.mkdir(exist_ok=True)

    vae = VAE(z_dim=args.z_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()

    apm = APM(z_dim=args.z_dim, action_dim=args.action_dim).to(device)
    apm.load_state_dict(torch.load(args.apm_ckpt, map_location=device))
    apm.eval()

    # Build LSR from unique training states
    print("Building LSR graph from training data...")
    graph_data = LSRGraphData(args.train)
    latents = encode_unique(vae, graph_data, device)
    episodes, actions = graph_data.episodes_and_actions()

    lsr = LSR(k=args.k)
    lsr.build(latents, episodes, actions)
    lsr.save(str(out / "lsr_graph.pkl"))
    print(f"Graph: {len(lsr.graph)} nodes with edges, {len(lsr.valid_transitions)} valid transitions")

    # Evaluate on val/holdout set
    val_ds = LSRDataset(args.val)
    n_success, n_total = 0, 0
    path_lengths = []
    results = []

    # Group val transitions into "episodes" by source state for meaningful start/goal pairs
    # For flat transition data, we treat each transition as its own planning problem
    for idx, (obs_t, obs_goal, action) in enumerate(val_ds):
        if idx >= args.max_pairs:
            break

        with torch.no_grad():
            _, z_start, _ = vae.encode(obs_t.unsqueeze(0).to(device))
            _, z_goal_enc, _ = vae.encode(obs_goal.unsqueeze(0).to(device))

        z_start_np = z_start.cpu().numpy().squeeze()
        z_goal_np  = z_goal_enc.cpu().numpy().squeeze()

        result = lsr.plan(z_start_np, z_goal_np)
        n_total += 1

        if result is None:
            results.append({"idx": idx, "success": False, "path_len": None})
            continue

        path, latent_path = result
        path_lengths.append(len(path))

        # Success if goal reconstruction matches ground truth
        with torch.no_grad():
            z_end = torch.tensor(latent_path[-1], dtype=torch.float32).unsqueeze(0).to(device)
            recon = vae.decode(z_end)
            mse = torch.mean((recon.cpu() - obs_goal.unsqueeze(0)) ** 2).item()

        success = mse < args.recon_thresh
        if success:
            n_success += 1

        results.append({"idx": idx, "success": success, "path_len": len(path), "goal_mse": mse})

        if idx < args.save_examples:
            _save_plan(vae, apm, obs_t, obs_goal, latent_path, device,
                       plans_dir / f"pair_{idx:04d}")

    rate = n_success / max(n_total, 1)
    avg_len = float(np.mean(path_lengths)) if path_lengths else 0.0

    summary = {
        "planning_success_rate": rate,
        "n_success": n_success,
        "n_total": n_total,
        "avg_path_length": avg_len,
        "recon_thresh": args.recon_thresh,
    }
    print(f"\n=== Results ===")
    print(f"Success rate : {rate:.3f}  ({n_success}/{n_total})")
    print(f"Avg path len : {avg_len:.1f}")

    with open(out / "results.json", "w") as f:
        json.dump({"summary": summary, "per_pair": results}, f, indent=2)
    print(f"Saved → {out}/results.json")


def _save_plan(vae, apm, obs_start, obs_goal, latent_path, device, save_dir):
    save_dir.mkdir(exist_ok=True)
    frames = [obs_start]
    with torch.no_grad():
        for z_np in latent_path:
            z = torch.tensor(z_np, dtype=torch.float32).unsqueeze(0).to(device)
            frames.append(vae.decode(z).cpu().squeeze(0))
        frames.append(obs_goal)
    strip = torch.stack(frames)
    save_image(strip, str(save_dir / "plan_strip.png"), nrow=len(strip))

    actions_pred = []
    with torch.no_grad():
        for i in range(len(latent_path) - 1):
            z_i = torch.tensor(latent_path[i],     dtype=torch.float32).unsqueeze(0).to(device)
            z_j = torch.tensor(latent_path[i + 1], dtype=torch.float32).unsqueeze(0).to(device)
            a = apm(z_i, z_j).cpu().squeeze().tolist()
            actions_pred.append(a if isinstance(a, list) else [a])
    with open(save_dir / "actions.json", "w") as f:
        json.dump(actions_pred, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",        required=True)
    parser.add_argument("--val",          required=True)
    parser.add_argument("--vae_ckpt",     required=True)
    parser.add_argument("--apm_ckpt",     required=True)
    parser.add_argument("--z_dim",        type=int,   default=4)
    parser.add_argument("--action_dim",   type=int,   default=4)
    parser.add_argument("--k",            type=int,   default=10)
    parser.add_argument("--recon_thresh", type=float, default=0.05)
    parser.add_argument("--max_pairs",    type=int,   default=500)
    parser.add_argument("--save_examples",type=int,   default=10)
    parser.add_argument("--out",          default="results/eval")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
