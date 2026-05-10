"""
LSR-v2 dataset loader for the official flat-transition format.

Each .pkl file is a list of tuples:
    (obs_t, obs_{t+1}, valid_flag, action, state_t, state_{t+1})

where:
    obs_t / obs_{t+1} : (256, 256, 3) uint8 RGB images
    valid_flag        : int  — 1 = real transition, 0 = null/initial observation
    action            : (4,) int64  — discrete robot command
    state_t           : (9,) int64  — symbolic grid state at t
    state_{t+1}       : (9,) int64  — symbolic grid state at t+1

Images are resized to 64×64 and normalised to [0, 1].

Two datasets are provided per task:
    *_2500.pkl   — training transitions
    *_holdout.pkl — held-out transitions for evaluation
"""

import pickle
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resize(img_np: np.ndarray, size: int = 64) -> np.ndarray:
    return np.array(Image.fromarray(img_np).resize((size, size), Image.BILINEAR))


class LSRDataset(Dataset):
    """
    Yields (obs_t, obs_{t+1}, action) for VAE / APM training.

    Only valid transitions (valid_flag == 1) are included.
    Images are resized to 64×64 and returned as (C, H, W) float tensors in [0,1].
    Actions are returned as float32 vectors.

    Attributes
    ----------
    pairs       : list of (obs_t_np, obs_tp1_np, action_np)
    episodes    : list of 2-element index lists — each valid transition is
                  its own "episode" [i, i+1] for LSR graph construction.
                  The latents array passed to LSR.build() has shape (2*N, z_dim)
                  where N = len(pairs); latent[2i] encodes pairs[i][0] and
                  latent[2i+1] encodes pairs[i][1].
    """

    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        img_size: int = 64,
    ):
        self.transform = transform
        self.img_size = img_size

        self.pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        # For LSR: each transition is a 2-node mini-episode
        self.episodes: List[List[int]] = []

        raw = self._load(Path(data_path))
        self._build(raw)

    def _load(self, path: Path) -> list:
        with open(path, "rb") as f:
            return pickle.load(f)

    def _build(self, raw: list):
        global_node = 0
        for item in raw:
            obs_t, obs_tp1, valid_flag, action, state_t, state_tp1 = item
            if int(valid_flag) == 0:
                continue
            if np.all(np.array(action) == -1):
                continue

            obs_t_r   = _resize(np.array(obs_t),   self.img_size)
            obs_tp1_r = _resize(np.array(obs_tp1), self.img_size)
            act       = np.array(action, dtype=np.float32)

            self.pairs.append((obs_t_r, obs_tp1_r, act))
            # Two consecutive nodes: [global_node, global_node+1]
            self.episodes.append([global_node, global_node + 1])
            global_node += 2

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        obs_t, obs_tp1, action = self.pairs[idx]

        t  = torch.from_numpy(obs_t.astype(np.float32)   / 255.0).permute(2, 0, 1)
        tp = torch.from_numpy(obs_tp1.astype(np.float32) / 255.0).permute(2, 0, 1)
        a  = torch.from_numpy(action)

        if self.transform is not None:
            t  = self.transform(t)
            tp = self.transform(tp)

        return t, tp, a


class LSRGraphData:
    """
    State-deduplicated view of the dataset for LSR graph construction.

    The box-stacking dataset has 288 unique symbolic states.  Many transitions
    share the same start/end state, so we deduplicate and give each unique
    state exactly one representative image.

    Attributes
    ----------
    images       : np.ndarray (N_states, H, W, C) uint8
    state_keys   : list[tuple]  — state vector for each unique state, len=N_states
    transitions  : list[(src_idx, dst_idx, action_np)]
    """

    def __init__(self, data_path: str, img_size: int = 64):
        raw = pickle.load(open(data_path, "rb"))
        self.img_size = img_size

        # Deduplicate states
        state_to_idx: dict = {}
        unique_imgs: List[np.ndarray] = []
        self.state_keys: List[tuple] = []

        self.transitions: List[Tuple[int, int, np.ndarray]] = []

        for item in raw:
            obs_t, obs_tp1, valid_flag, action, state_t, state_tp1 = item
            if int(valid_flag) == 0 or np.all(np.array(action) == -1):
                continue

            key_t  = tuple(np.array(state_t).tolist())
            key_tp = tuple(np.array(state_tp1).tolist())

            for key, obs in [(key_t, obs_t), (key_tp, obs_tp1)]:
                if key not in state_to_idx:
                    idx = len(unique_imgs)
                    state_to_idx[key] = idx
                    unique_imgs.append(_resize(np.array(obs), img_size))
                    self.state_keys.append(key)

            src = state_to_idx[key_t]
            dst = state_to_idx[key_tp]
            act = np.array(action, dtype=np.float32)
            self.transitions.append((src, dst, act))

        self.images = np.stack(unique_imgs, axis=0)   # (N_states, H, W, C)

    @property
    def n_states(self) -> int:
        return len(self.images)

    def episodes_and_actions(self):
        """
        Returns (episodes, actions) suitable for LSR.build().

        episodes : list of [src, dst] — one per training transition
        actions  : np.ndarray (2*N_states, action_dim) — actions[src] set for each edge
        """
        n = self.n_states
        action_dim = self.transitions[0][2].shape[0] if self.transitions else 4
        actions = np.zeros((n, action_dim), dtype=np.float32)
        episodes = []
        for src, dst, act in self.transitions:
            actions[src] = act
            episodes.append([src, dst])
        return episodes, actions
