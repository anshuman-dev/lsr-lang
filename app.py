"""
HuggingFace Spaces — Gradio demo for lsr-lang.

Shows:
  1. Latent space with planned path highlighted.
  2. Decoded visual plan (start → intermediate frames → goal).
  3. Predicted actions at each step.
  4. Toggle between baseline VAE and CLIP-VAE.

Expects pre-trained artifacts in results/:
  results/vae/vae_best.pt
  results/clip_vae/clip_vae_best.pt
  results/clip_vae/text_emb.npy
  results/eval/lsr_graph.pkl  (or rebuilt on start-up)
"""

import json
import os
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models.apm import APM
from models.clip_vae import ClipVAE
from models.lsr import LSR
from models.vae import VAE

# ---- Config ----
Z_DIM = int(os.getenv("Z_DIM", "4"))
ACTION_DIM = int(os.getenv("ACTION_DIM", "4"))
DATA_PATH = os.getenv("DATA_PATH", "data/raw/box_stacking_normal.pkl")

DEVICE = torch.device("cpu")    # HF Spaces runs CPU; MPS/CUDA not available there

# ---- Load models ----

def _load_model(ckpt: str, model: torch.nn.Module) -> torch.nn.Module:
    if Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


vae      = _load_model("results/vae/vae_best.pt", VAE(z_dim=Z_DIM))
clip_vae = _load_model("results/clip_vae/clip_vae_best.pt", ClipVAE(z_dim=Z_DIM))
apm      = _load_model("results/apm/apm_best.pt", APM(z_dim=Z_DIM, action_dim=ACTION_DIM))

text_emb_path = "results/clip_vae/text_emb.npy"
text_emb = (
    torch.tensor(np.load(text_emb_path), dtype=torch.float32)
    if Path(text_emb_path).exists()
    else None
)

lsr_path = "results/eval/lsr_graph.pkl"
lsr: LSR = LSR.load(lsr_path) if Path(lsr_path).exists() else LSR()


# ---- Helpers ----

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _encode(img_pil: Image.Image, use_clip: bool, lang: str):
    img = img_pil.convert("RGB").resize((64, 64))
    x = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        if use_clip and text_emb is not None:
            if lang.strip():
                from models.clip_vae import encode_text
                t_emb = encode_text([lang], DEVICE)
            else:
                t_emb = text_emb
            _, mu, _ = clip_vae.encode(x, t_emb)
        else:
            _, mu, _ = vae.encode(x)
    return mu.squeeze().numpy()


def _decode(z_np: np.ndarray, use_clip: bool) -> Image.Image:
    z = torch.tensor(z_np, dtype=torch.float32).unsqueeze(0)
    model = clip_vae if use_clip else vae
    with torch.no_grad():
        out = model.decode(z)
    return _tensor_to_pil(out)


def _latent_plot(latent_path: np.ndarray) -> Image.Image:
    if lsr.latents is None:
        return None

    from sklearn.decomposition import PCA
    all_z = lsr.latents
    coords = PCA(n_components=2).fit_transform(all_z) if all_z.shape[1] > 2 else all_z

    # Project path nodes
    path_idx = [
        int(lsr.tree.query(z.reshape(1, -1))[1][0]) for z in latent_path
    ] if lsr.tree is not None else []

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.4, c="lightgray")
    if path_idx:
        path_coords = coords[path_idx]
        ax.plot(path_coords[:, 0], path_coords[:, 1], "r-o", markersize=6, linewidth=2, label="plan")
        ax.scatter(path_coords[[0], 0], path_coords[[0], 1], c="green", s=80, zorder=5, label="start")
        ax.scatter(path_coords[[-1], 0], path_coords[[-1], 1], c="blue", s=80, zorder=5, label="goal")
    ax.legend(fontsize=9)
    ax.set_title("Latent space + planned path")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = __import__("io").BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ---- Main inference function ----

def run_plan(start_img, goal_img, language_goal, use_clip_str):
    use_clip = use_clip_str == "CLIP-VAE"

    if start_img is None or goal_img is None:
        return [], None, "Please provide both a start and goal image."

    z_start = _encode(Image.fromarray(start_img), use_clip, language_goal)
    z_goal  = _encode(Image.fromarray(goal_img),  use_clip, language_goal)

    result = lsr.plan(z_start, z_goal)
    if result is None:
        return [], None, "No path found in latent roadmap."

    path, latent_path = result

    # Decode plan frames
    plan_imgs = [_tensor_to_pil(
        torch.tensor(np.array(Image.fromarray(start_img).resize((64, 64))),
                     dtype=torch.float32).permute(2, 0, 1) / 255.0
    )]
    for z_np in latent_path:
        plan_imgs.append(_decode(z_np, use_clip))
    plan_imgs.append(_tensor_to_pil(
        torch.tensor(np.array(Image.fromarray(goal_img).resize((64, 64))),
                     dtype=torch.float32).permute(2, 0, 1) / 255.0
    ))

    # Predict actions
    actions_text = []
    with torch.no_grad():
        for i in range(len(latent_path) - 1):
            z_i = torch.tensor(latent_path[i],     dtype=torch.float32).unsqueeze(0)
            z_j = torch.tensor(latent_path[i + 1], dtype=torch.float32).unsqueeze(0)
            a = apm(z_i, z_j).squeeze().tolist()
            actions_text.append(f"Step {i+1}: " + ", ".join(f"{v:.3f}" for v in a))

    info = (
        f"Model: {'CLIP-VAE' if use_clip else 'Baseline VAE'}\n"
        f"Path length: {len(path)} nodes\n\n"
        "Predicted actions:\n" + "\n".join(actions_text)
    )

    latent_plot = _latent_plot(latent_path)
    return plan_imgs, latent_plot, info


# ---- Gradio UI ----

with gr.Blocks(title="LSR-Lang: Visual Action Planning") as demo:
    gr.Markdown("## LSR-Lang — Latent Space Roadmap with Language Conditioning")
    gr.Markdown(
        "Upload a **start** and **goal** image of a manipulation scene. "
        "Optionally provide a language description. "
        "The model plans a visual path through latent space and predicts robot actions."
    )

    with gr.Row():
        with gr.Column():
            start_img  = gr.Image(label="Start image",  type="numpy")
            goal_img   = gr.Image(label="Goal image",   type="numpy")
            lang_goal  = gr.Textbox(label="Language goal (optional)",
                                    placeholder="stack the red box on the blue box")
            model_sel  = gr.Radio(
                choices=["Baseline VAE", "CLIP-VAE"],
                value="Baseline VAE",
                label="Model",
            )
            run_btn = gr.Button("Plan", variant="primary")

        with gr.Column():
            plan_gallery = gr.Gallery(label="Visual plan", columns=8, height="auto")
            latent_plot  = gr.Image(label="Latent space + path")
            info_box     = gr.Textbox(label="Info / actions", lines=10)

    run_btn.click(
        fn=run_plan,
        inputs=[start_img, goal_img, lang_goal, model_sel],
        outputs=[plan_gallery, latent_plot, info_box],
    )

    gr.Examples(
        examples=[["", "", "stack the red box on the blue box", "CLIP-VAE"]],
        inputs=[start_img, goal_img, lang_goal, model_sel],
    )


if __name__ == "__main__":
    demo.launch()
