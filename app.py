"""
HuggingFace Spaces — Gradio demo for lsr-lang.

Models are downloaded from singhanshuman/lsr-lang-models on first run.
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

# ── device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")
Z_DIM = 4
ACTION_DIM = 4
MODEL_REPO = "singhanshuman/lsr-lang-models"

# ── download weights ─────────────────────────────────────────────────────────
def _get(filename: str) -> str:
    return hf_hub_download(repo_id=MODEL_REPO, filename=filename)

# ── imports after sys.path is stable ─────────────────────────────────────────
from models.apm import APM
from models.clip_vae import ClipVAE
from models.lsr import LSR
from models.vae import VAE


def _load(ckpt: str, model: torch.nn.Module) -> torch.nn.Module:
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


print("Loading models from HF Hub...")
vae      = _load(_get("vae_best.pt"),      VAE(z_dim=Z_DIM))
clip_vae = _load(_get("clip_vae_best.pt"), ClipVAE(z_dim=Z_DIM))
apm      = _load(_get("apm_best.pt"),      APM(z_dim=Z_DIM, action_dim=ACTION_DIM))
text_emb = torch.tensor(np.load(_get("text_emb.npy")), dtype=torch.float32)
lsr      = LSR.load(_get("lsr_graph.pkl"))
print("Ready.")


# ── helpers ───────────────────────────────────────────────────────────────────
def _pil_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    img = img_pil.convert("RGB").resize((64, 64))
    return torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _encode_img(img_pil: Image.Image, use_clip: bool, lang: str):
    x = _pil_to_tensor(img_pil)
    with torch.no_grad():
        if use_clip:
            t_emb = text_emb
            if lang.strip():
                from models.clip_vae import encode_text
                t_emb = encode_text([lang], DEVICE)
            t_emb = t_emb.expand(1, -1)
            _, mu, _ = clip_vae.encode(x, t_emb)
        else:
            _, mu, _ = vae.encode(x)
    return mu.squeeze().numpy()


def _decode(z_np: np.ndarray, use_clip: bool) -> Image.Image:
    z = torch.tensor(z_np, dtype=torch.float32).unsqueeze(0)
    model = clip_vae if use_clip else vae
    with torch.no_grad():
        return _tensor_to_pil(model.decode(z))


def _latent_plot(latent_path: np.ndarray) -> Image.Image:
    if lsr.latents is None:
        return None
    all_z = lsr.latents
    if all_z.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2).fit(all_z)
        coords = pca.transform(all_z)
        path_coords = pca.transform(latent_path)
    else:
        coords = all_z
        path_coords = latent_path

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.35, c="lightgray")
    ax.plot(path_coords[:, 0], path_coords[:, 1], "r-o", ms=7, lw=2, label="plan")
    ax.scatter(*path_coords[0],  c="green", s=90, zorder=5, label="start")
    ax.scatter(*path_coords[-1], c="blue",  s=90, zorder=5, label="goal")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    ax.set_title("Latent space + planned path")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── main inference ────────────────────────────────────────────────────────────
def run_plan(start_img, goal_img, language_goal, model_choice):
    use_clip = model_choice == "CLIP-VAE"
    if start_img is None or goal_img is None:
        return [], None, "Provide both a start and a goal image."

    z_start = _encode_img(Image.fromarray(start_img), use_clip, language_goal)
    z_goal  = _encode_img(Image.fromarray(goal_img),  use_clip, language_goal)

    result = lsr.plan(z_start, z_goal)
    if result is None:
        return [], None, "No path found in the latent roadmap."

    path, latent_path = result

    plan_imgs = [Image.fromarray(start_img).resize((64, 64))]
    for z_np in latent_path:
        plan_imgs.append(_decode(z_np, use_clip))
    plan_imgs.append(Image.fromarray(goal_img).resize((64, 64)))

    action_lines = []
    with torch.no_grad():
        for i in range(len(latent_path) - 1):
            z_i = torch.tensor(latent_path[i],     dtype=torch.float32).unsqueeze(0)
            z_j = torch.tensor(latent_path[i + 1], dtype=torch.float32).unsqueeze(0)
            a = apm(z_i, z_j).squeeze().tolist()
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
        "Upload **start** and **goal** images of a box-stacking scene. "
        "The model plans a visual path through latent space and predicts robot actions."
    )
    with gr.Row():
        with gr.Column(scale=1):
            start_img   = gr.Image(label="Start image",  type="numpy")
            goal_img    = gr.Image(label="Goal image",   type="numpy")
            lang_goal   = gr.Textbox(
                label="Language goal (optional, CLIP-VAE only)",
                placeholder="stack the red box on the blue box",
            )
            model_radio = gr.Radio(
                ["Baseline VAE", "CLIP-VAE"], value="Baseline VAE", label="Model"
            )
            run_btn = gr.Button("Plan →", variant="primary")
        with gr.Column(scale=2):
            gallery     = gr.Gallery(label="Visual plan", columns=10, height="auto")
            latent_plot = gr.Image(label="Latent space + path")
            info_box    = gr.Textbox(label="Actions", lines=8)

    run_btn.click(
        fn=run_plan,
        inputs=[start_img, goal_img, lang_goal, model_radio],
        outputs=[gallery, latent_plot, info_box],
    )

    gr.Markdown(
        "**Code:** [github.com/anshuman-dev/lsr-lang](https://github.com/anshuman-dev/lsr-lang) · "
        "**Paper:** [LSR-v2 (IEEE T-RO 2023)](https://arxiv.org/abs/2103.02554)"
    )

if __name__ == "__main__":
    demo.launch()
