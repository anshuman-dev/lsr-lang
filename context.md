# Project Context: LSR-V2 Reproduction + Language Extension

## Who is building this

**Anshuman Singh** — applying for a PhD position at KTH under **Prof. Danica Kragic Jensfelt** (Robot Learning for Manipulation, focus on VLA models for tabletop manipulation).

GitHub: https://github.com/anshuman-dev
Website: https://singhanshuman.me

### Existing repos (for narrative continuity):
- `rfmp-ellipsoids` — Riemannian Flow Matching on SPD manifolds for robot manipulability transfer. Three models, baselines, 30 unit tests, LaTeX report. References IROS 2024 papers from Kragic's lab circle.
- `gphlvm-repro` — From-scratch reproduction of Jaquier et al. ICML 2024 (Kragic collaborator). Lorentz geometry, Riemannian Adam, 74 unit tests, training GIF, HuggingFace Spaces demo, GitHub Pages.

Both repos follow the same style: proper structure, tests, reproducible scripts, write-ups, demos.

---

## What we are building

**Repo name:** `lsr-lang` (or `lsr-clip`)

A from-scratch reproduction of the **Latent Space Roadmap (LSR-v2)** paper from Kragic's lab, extended with CLIP language conditioning — making it a primitive Vision-Language-Action (VLA) system.

### Paper being reproduced:
> Lippi, Poklukar, Welle, Varava, Yin, Marino, Kragic.
> "Enabling Visual Action Planning for Object Manipulation through Latent Space Roadmap."
> IEEE Transactions on Robotics, vol. 39, no. 1, pp. 57–75, 2023.

ArXiv: https://arxiv.org/abs/2103.02554
Official code (NOT to copy, only for reference): https://github.com/visual-action-planning/lsr-v2-code
Paper website: https://visual-action-planning.github.io/lsr-v2/

---

## Part 1 — LSR-v2 Core Reproduction (from scratch)

### Core idea:
Given start and goal images of a manipulation scene, plan a sequence of intermediate images (visual plan) and predict the actions between them.

### Architecture:

**1. Mapping Module (MM) — VAE**
- Convolutional VAE that encodes 64x64 RGB images into a low-dimensional latent space (z_dim=2 or 4)
- Trained with standard ELBO loss (reconstruction + KL)
- The latent space should cluster manipulation states meaningfully

**2. Latent Space Roadmap (LSR) — Graph**
- After VAE training, encode the entire training dataset into latent space
- Build a k-NN graph on latent points (k=5 or 10)
- Add edges only between states that are "reachable" (i.e., have a valid action between them in training data)
- At inference: given start z_s and goal z_g, find shortest path through graph (Dijkstra/A*)
- Decode path nodes back to images → visual plan

**3. Action Proposal Module (APM)**
- MLP that takes a pair of latent states (z_i, z_{i+1}) and predicts the action u between them
- Trained supervised on (z_i, z_{i+1}, u) triplets from training data

### Dataset:
Use the official LSR-v2 datasets (freely downloadable):
- Box stacking normal task
- Box stacking hard task  
- Rope-box hybrid task (deformable object)

Download script from: https://github.com/visual-action-planning/lsr-v2-code/blob/main/datasets/get_datasets.py

Each sample: (image_t, image_{t+1}, action_u) where images are 64x64 RGB and actions are continuous robot commands.

### Metrics to reproduce:
- Planning success rate on test set
- Latent space visualization (2D scatter, colored by object position or task state)
- Qualitative visual plans (start → intermediate frames → goal)

---

## Part 2 — Language Extension (the VLA twist)

### Core idea:
Condition the VAE encoder on a CLIP text embedding so the latent space is organized by both visual content AND language goal description. This turns the system into a primitive VLA: image + language → latent plan → action sequence.

### Architecture change:
- Keep the same VAE decoder
- Encoder: concatenate image features with CLIP text embedding (e.g. "stack the red box on the blue box")
- Use `openai/clip-vit-base-patch32` via HuggingFace — frozen, no fine-tuning needed
- CLIP text embedding (512-dim) projected down to 64-dim via linear layer, concatenated with CNN image features before the latent bottleneck

### Language labels for box stacking dataset:
- "stack the red box on the blue box"
- "place the object on top"
- Simple templates — the dataset has known task structure so labels can be auto-generated

### Comparison:
Train both:
1. Baseline VAE (Part 1, no language)
2. CLIP-conditioned VAE (Part 2)

Compare latent space quality: do semantically similar states cluster better with language conditioning?
Metric: k-NN classification accuracy of task state from latent code.

---

## Part 3 — Optional: Flow Matching Encoder

If time permits, add a third variant:
- Replace VAE with a **Riemannian Flow Matching** encoder (connecting to `rfmp-ellipsoids` repo)
- This ties all three repos into one narrative arc

---

## Repo Structure

```
lsr-lang/
├── data/
│   └── get_datasets.py          # download LSR-v2 datasets
├── models/
│   ├── vae.py                   # convolutional VAE (baseline)
│   ├── clip_vae.py              # CLIP-conditioned VAE
│   ├── lsr.py                   # graph construction + Dijkstra planning
│   └── apm.py                   # action proposal MLP
├── scripts/
│   ├── train_vae.py             # train baseline VAE
│   ├── train_clip_vae.py        # train CLIP-conditioned VAE
│   ├── train_apm.py             # train action proposal network
│   ├── evaluate.py              # planning success rate
│   ├── visualize_latent.py      # 2D latent space scatter plots
│   ├── animate.py               # training evolution GIF
│   └── compare.py               # baseline vs CLIP comparison
├── tests/
│   ├── test_vae.py
│   ├── test_lsr.py
│   └── test_apm.py
├── app.py                       # HuggingFace Spaces Gradio demo
├── docs/                        # GitHub Pages
│   └── index.html
├── report/
│   ├── report.tex
│   └── report.pdf
├── results/
│   ├── latent_space_baseline.png
│   ├── latent_space_clip.png
│   ├── training_evolution.gif
│   └── planning_examples/
├── README.md
├── requirements.txt
└── hf_requirements.txt
```

---

## HuggingFace Spaces Demo (app.py)

Interactive Gradio demo:
- User selects a start image and goal image from the test set (dropdown or gallery)
- User optionally types a language goal (e.g. "stack the red box on the blue box")
- Model runs LSR planning → shows:
  1. The 2D latent space with the planned path highlighted
  2. The sequence of decoded images (visual plan)
  3. The predicted actions at each step
- Toggle between baseline VAE and CLIP-conditioned VAE to compare

---

## GitHub Pages (docs/index.html)

Static page showing:
- What the project is (1 paragraph)
- Training evolution GIF (latent space building up, like gphlvm-repro's Poincaré disk GIF)
- Results table (baseline vs CLIP)
- Links: paper, HF demo, code

Style: same clean aesthetic as https://anshuman-dev.github.io/gphlvm-repro/

---

## Code Style Requirements

Follow exactly the same conventions as the existing repos:

**From rfmp-ellipsoids:**
- Proper module structure with `models/`, `scripts/`, `tests/`, `data/`, `report/`
- Unit tests for every mathematical component (at least 30)
- Shell script to reproduce everything
- LaTeX report (5 pages, two-column, IEEE style)
- Honest reporting — if CLIP doesn't improve things, say so and explain why

**From gphlvm-repro:**
- Training evolution GIF
- HuggingFace Spaces demo with `app.py`
- GitHub Pages at `docs/`
- Annotated result visualizations
- Key implementation decisions documented in README (what was tricky, what bugs were found)

**General:**
- Python 3.11
- PyTorch (MPS compatible for Apple Silicon)
- Tested with `pytest tests/`
- `requirements.txt` pinned versions
- MIT license

---

## Narrative for Application

This repo completes a trilogy:

1. `gphlvm-repro` — reproduced Jaquier et al. ICML 2024 (hyperbolic geometry for grasp taxonomy, Kragic collaborator)
2. `rfmp-ellipsoids` — implemented Riemannian flow matching for manipulability ellipsoids (IROS 2024 paper from Kragic's circle)
3. `lsr-lang` (this repo) — reproduces Kragic's own LSR-v2 paper and extends it toward VLA with language conditioning

The arc shows: geometric ML → manipulation-specific geometry → visual action planning → language-conditioned planning (VLA). Exactly the research trajectory the PhD is about.

---

## Key Papers to Read Before Building

1. LSR-v2: https://arxiv.org/abs/2103.02554 (the paper being reproduced)
2. LSR-v1: https://arxiv.org/abs/2003.08974 (original IROS 2020 version)
3. CLIP: https://arxiv.org/abs/2103.00020 (for the language extension)
4. Diffusion Policy: https://arxiv.org/abs/2303.04137 (context for where this field is going)

---

## Priority Order

1. Get dataset download working
2. Train baseline VAE and visualize latent space
3. Build LSR graph + Dijkstra planner
4. Train APM
5. Evaluate planning success rate
6. Add CLIP conditioning
7. Compare and make figures
8. Training GIF
9. HuggingFace demo
10. GitHub Pages
11. LaTeX report
12. Tests (write alongside, not after)