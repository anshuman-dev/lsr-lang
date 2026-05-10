"""
HuggingFace Spaces — Gradio demo for lsr-lang.

Models are downloaded from singhanshuman/lsr-lang-models on first run.
Compatible with Gradio 4.x and 6.x.
"""

import io
import os
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

DEVICE = torch.device("cpu")
Z_DIM = 4
ACTION_DIM = 4
MODEL_REPO = "singhanshuman/lsr-lang-models"

from models.apm import APM
from models.clip_vae import ClipVAE
from models.lsr import LSR
from models.vae import VAE


def _get(filename: str) -> str:
    return hf_hub_download(repo_id=MODEL_REPO, filename=filename)


def _load(ckpt: str, model: torch.nn.Module) -> torch.nn.Module:
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


print("Loading models…")
vae      = _load(_get("vae_best.pt"),      VAE(z_dim=Z_DIM))
clip_vae = _load(_get("clip_vae_best.pt"), ClipVAE(z_dim=Z_DIM))
apm      = _load(_get("apm_best.pt"),      APM(z_dim=Z_DIM, action_dim=ACTION_DIM))
text_emb = torch.tensor(np.load(_get("text_emb.npy")), dtype=torch.float32)
lsr      = LSR.load(_get("lsr_graph.pkl"))
print("Ready.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((64, 64))
    return torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze().permute(1, 2, 0).clamp(0, 1).detach().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _encode(img: Image.Image, use_clip: bool, lang: str) -> np.ndarray:
    x = _pil_to_tensor(img)
    with torch.no_grad():
        if use_clip:
            t_emb = text_emb.clone()
            if lang.strip():
                from models.clip_vae import encode_text
                t_emb = encode_text([lang], DEVICE)
            _, mu, _ = clip_vae.encode(x, t_emb.expand(1, -1))
        else:
            _, mu, _ = vae.encode(x)
    return mu.squeeze().numpy()


def _decode(z: np.ndarray, use_clip: bool) -> Image.Image:
    zt = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
    model = clip_vae if use_clip else vae
    with torch.no_grad():
        return _tensor_to_pil(model.decode(zt))


def _latent_plot(latent_path: np.ndarray) -> Image.Image:
    if lsr.latents is None:
        return None
    all_z = lsr.latents
    if all_z.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2).fit(all_z)
        coords      = pca.transform(all_z)
        path_coords = pca.transform(latent_path)
    else:
        coords      = all_z
        path_coords = latent_path

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.35, c="lightgray")
    ax.plot(path_coords[:, 0], path_coords[:, 1], "r-o", ms=7, lw=2, label="plan")
    ax.scatter(*path_coords[0],  c="limegreen", s=100, zorder=5, label="start")
    ax.scatter(*path_coords[-1], c="royalblue", s=100, zorder=5, label="goal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_title("Latent space + planned path")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── inference ─────────────────────────────────────────────────────────────────

def run_plan(start_np, goal_np, lang, model_choice):
    use_clip = model_choice == "CLIP-VAE"
    if start_np is None or goal_np is None:
        return [], None, "Please upload both a start and a goal image."

    start_pil = Image.fromarray(start_np)
    goal_pil  = Image.fromarray(goal_np)

    z_start = _encode(start_pil, use_clip, lang)
    z_goal  = _encode(goal_pil,  use_clip, lang)

    result = lsr.plan(z_start, z_goal)
    if result is None:
        return [], None, "No path found in the latent roadmap."

    path, latent_path = result

    plan_imgs = [start_pil.resize((64, 64))]
    for z in latent_path:
        plan_imgs.append(_decode(z, use_clip))
    plan_imgs.append(goal_pil.resize((64, 64)))

    action_lines = []
    with torch.no_grad():
        for i in range(len(latent_path) - 1):
            zi = torch.tensor(latent_path[i],     dtype=torch.float32).unsqueeze(0)
            zj = torch.tensor(latent_path[i + 1], dtype=torch.float32).unsqueeze(0)
            a  = apm(zi, zj).squeeze().tolist()
            vals = a if isinstance(a, list) else [a]
            action_lines.append(f"Step {i+1}: [{', '.join(f'{v:.3f}' for v in vals)}]")

    info = (
        f"Model : {'CLIP-VAE' if use_clip else 'Baseline VAE'}\n"
        f"Path  : {len(path)} nodes\n\n"
        "Predicted actions:\n" + "\n".join(action_lines)
    )
    return plan_imgs, _latent_plot(latent_path), info


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="lsr-lang — Visual Action Planning") as demo:
    gr.Markdown(
        "## lsr-lang — Latent Space Roadmap with Language Conditioning\n"
        "Upload **start** and **goal** images from a box-stacking scene. "
        "The system plans a visual path through latent space and predicts robot actions.\n\n"
        "**[GitHub](https://github.com/anshuman-dev/lsr-lang)** · "
        "**[Paper — LSR-v2 (IEEE T-RO 2023)](https://arxiv.org/abs/2103.02554)**"
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp_start  = gr.Image(label="Start image",  type="numpy")
            inp_goal   = gr.Image(label="Goal image",   type="numpy")
            inp_lang   = gr.Textbox(
                label="Language goal (optional, CLIP-VAE only)",
                placeholder="stack the red box on the blue box",
            )
            inp_model  = gr.Radio(
                choices=["Baseline VAE", "CLIP-VAE"],
                value="Baseline VAE",
                label="Model",
            )
            btn = gr.Button("Plan →", variant="primary")

        with gr.Column(scale=2):
            out_gallery = gr.Gallery(label="Visual plan (start → plan → goal)", columns=12)
            out_latent  = gr.Image(label="Latent space + path")
            out_info    = gr.Textbox(label="Predicted actions", lines=8)

    btn.click(
        fn=run_plan,
        inputs=[inp_start, inp_goal, inp_lang, inp_model],
        outputs=[out_gallery, out_latent, out_info],
    )

if __name__ == "__main__":
    demo.launch()
