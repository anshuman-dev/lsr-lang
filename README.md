# lsr-lang

From-scratch reproduction of the [Latent Space Roadmap (LSR-v2)](https://arxiv.org/abs/2103.02554) paper, extended with CLIP language conditioning.

Given a start image and a goal image of a robot manipulation scene, the system plans a sequence of intermediate images and predicts the robot actions between them.

**[Live Demo](https://huggingface.co/spaces/anshuman-dev/lsr-lang) · [Project Page](https://anshuman-dev.github.io/lsr-lang) · [Report](report/report.pdf)**

---

## Architecture

```
Image (64×64) ──► ConvEncoder ──► z (4-d) ──► ConvDecoder ──► Reconstruction
                      ▲
               CLIP text emb          (CLIP-VAE variant only)
               512-d → 64-d

Latent cloud ──► k-NN graph ──► Dijkstra ──► latent path ──► visual plan
                 (edges = training transitions only)

(z_i, z_{i+1}) ──► APM MLP ──► action
```

**Three components trained in sequence:**

1. **VAE** — convolutional β-VAE, z_dim=4, trained with ELBO loss
2. **LSR Graph** — k-NN graph on encoded training frames, filtered to valid transitions, Dijkstra planner
3. **APM** — MLP predicting continuous actions from adjacent latent pairs

**CLIP extension** — frozen `openai/clip-vit-base-patch32` text encoder, 512→64 projection concatenated with CNN features.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.11, PyTorch 2.3.1. Works on Apple Silicon (MPS), CUDA, and CPU.

---

## Data

```bash
python data/get_datasets.py --dest data/raw --datasets normal hard rope
```

Downloads the three official LSR-v2 datasets (box stacking normal/hard, rope-box).

---

## Training

```bash
# 1. Baseline VAE
python scripts/train_vae.py \
    --data data/raw/box_stacking_normal.pkl \
    --z_dim 4 --epochs 100 --out results/vae

# 2. CLIP-conditioned VAE
python scripts/train_clip_vae.py \
    --data data/raw/box_stacking_normal.pkl \
    --z_dim 4 --epochs 100 --out results/clip_vae

# 3. Action Proposal Module (uses frozen VAE)
python scripts/train_apm.py \
    --data data/raw/box_stacking_normal.pkl \
    --vae_ckpt results/vae/vae_best.pt \
    --z_dim 4 --action_dim 4 --out results/apm
```

---

## Evaluation

```bash
# Planning success rate + build LSR graph
python scripts/evaluate.py \
    --data data/raw/box_stacking_normal.pkl \
    --vae_ckpt results/vae/vae_best.pt \
    --apm_ckpt results/apm/apm_best.pt \
    --out results/eval

# Latent space scatter plot
python scripts/visualize_latent.py \
    --data data/raw/box_stacking_normal.pkl \
    --vae_ckpt results/vae/vae_best.pt \
    --out results/latent_space_baseline.png

# Training evolution GIF
python scripts/animate.py \
    --snapshot_dir results/vae/checkpoints \
    --data data/raw/box_stacking_normal.pkl \
    --out results/training_evolution.gif

# Baseline vs CLIP comparison
python scripts/compare.py \
    --data data/raw/box_stacking_normal.pkl \
    --vae_ckpt results/vae/vae_best.pt \
    --clip_vae_ckpt results/clip_vae/clip_vae_best.pt \
    --clip_text_emb results/clip_vae/text_emb.npy \
    --out results
```

---

## Tests

```bash
pytest tests/ -v
```

33 unit tests covering VAE forward/backward, LSR graph construction and planning, and APM.

---

## Implementation notes

**Edge filtering.** The k-NN graph is built with k=10, but an edge (i, j) is kept only if (i, j) appears as consecutive timesteps in a training trajectory. This means the planner can't take shortcuts through physically unreachable states, at the cost of a sparser graph. If planning fails on a given dataset, increasing k is the right lever.

**MPS.** SciPy's KDTree runs on CPU (numpy arrays). All PyTorch ops use the MPS backend when available. If you hit an unsupported-op warning, the operation silently falls back to CPU — check `PYTORCH_ENABLE_MPS_FALLBACK=1` in your env.

**CLIP text embedding.** The same embedding is broadcast across every batch during training — it's a task-level descriptor, not per-image. Multiple paraphrases of the task are averaged before training starts for slightly more robust conditioning.

**What didn't work.** On single-task datasets the CLIP conditioning has limited benefit because the text embedding is constant across all training samples — the encoder can learn to ignore it. The comparison is more interesting on the rope-box dataset where two qualitatively different objects appear.

---

## Project structure

```
lsr-lang/
├── data/
│   ├── get_datasets.py      # download LSR-v2 datasets
│   └── dataset.py           # PyTorch Dataset
├── models/
│   ├── vae.py               # convolutional VAE
│   ├── clip_vae.py          # CLIP-conditioned VAE
│   ├── lsr.py               # graph + Dijkstra planner
│   └── apm.py               # action proposal MLP
├── scripts/
│   ├── train_vae.py
│   ├── train_clip_vae.py
│   ├── train_apm.py
│   ├── evaluate.py
│   ├── visualize_latent.py
│   ├── animate.py
│   └── compare.py
├── tests/                   # 33 unit tests
├── app.py                   # HuggingFace Spaces Gradio demo
├── docs/index.html          # GitHub Pages
├── report/report.tex        # IEEE two-column report
├── requirements.txt
└── hf_requirements.txt
```

---

## License

MIT
