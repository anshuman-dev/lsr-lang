"""Unit tests for the Latent Space Roadmap (LSR) graph and planner."""

import numpy as np
import pytest

from models.lsr import LSR


# ---- helpers ----

def _simple_lsr(k=5) -> LSR:
    """
    4-node chain: 0-1-2-3 with unit spacing.
    Each step is also a training transition.
    """
    latents = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ], dtype=np.float32)
    episodes = [[0, 1, 2, 3]]
    actions = np.zeros((4, 2), dtype=np.float32)
    actions[0] = [1.0, 0.0]
    actions[1] = [1.0, 0.0]
    actions[2] = [1.0, 0.0]
    actions[3] = [0.0, 0.0]   # last frame has no forward action

    lsr = LSR(k=k)
    lsr.build(latents, episodes, actions)
    return lsr


# ---- build ----

def test_build_populates_latents():
    lsr = _simple_lsr()
    assert lsr.latents is not None
    assert lsr.latents.shape == (4, 2)

def test_build_creates_tree():
    lsr = _simple_lsr()
    assert lsr.tree is not None

def test_build_records_valid_transitions():
    lsr = _simple_lsr()
    assert (0, 1) in lsr.valid_transitions
    assert (1, 2) in lsr.valid_transitions
    assert (2, 3) in lsr.valid_transitions

def test_build_does_not_add_invalid_edges():
    # 0-1-2-3 chain; (0,3) should NOT be a direct edge
    lsr = _simple_lsr()
    has_0_3 = any(j == 3 for j, _ in lsr.graph.get(0, []))
    assert not has_0_3

def test_build_multiple_episodes():
    latents = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    episodes = [[0, 1], [2, 3]]
    actions = np.zeros((4, 2), dtype=np.float32)
    lsr = LSR(k=4)
    lsr.build(latents, episodes, actions)
    assert (0, 1) in lsr.valid_transitions
    assert (2, 3) in lsr.valid_transitions


# ---- plan ----

def test_plan_returns_tuple_for_valid_path():
    lsr = _simple_lsr()
    result = lsr.plan(np.array([0.0, 0.0]), np.array([3.0, 0.0]))
    assert result is not None
    path, latent_path = result
    assert isinstance(path, list)
    assert latent_path.shape[1] == 2

def test_plan_path_starts_at_start():
    lsr = _simple_lsr()
    path, _ = lsr.plan(np.array([0.0, 0.0]), np.array([3.0, 0.0]))
    assert path[0] == 0

def test_plan_path_ends_at_goal():
    lsr = _simple_lsr()
    path, _ = lsr.plan(np.array([0.0, 0.0]), np.array([3.0, 0.0]))
    assert path[-1] == 3

def test_plan_same_start_goal():
    lsr = _simple_lsr()
    result = lsr.plan(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert result is not None
    path, _ = result
    assert len(path) == 1

def test_plan_returns_none_when_no_path():
    # Two isolated chains with no connecting edges
    latents = np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32)
    episodes = [[0, 1], [2, 3]]
    actions = np.zeros((4, 1), dtype=np.float32)
    lsr = LSR(k=1)
    lsr.build(latents, episodes, actions)
    result = lsr.plan(np.array([0.0]), np.array([10.0]))
    assert result is None

def test_plan_before_build_raises():
    lsr = LSR()
    with pytest.raises(RuntimeError):
        lsr.plan(np.array([0.0, 0.0]), np.array([1.0, 0.0]))


# ---- actions ----

def test_get_actions_for_path_length():
    lsr = _simple_lsr()
    path, _ = lsr.plan(np.array([0.0, 0.0]), np.array([3.0, 0.0]))
    actions = lsr.get_actions_for_path(path)
    assert len(actions) == len(path) - 1

def test_get_actions_for_path_not_none_on_valid_chain():
    lsr = _simple_lsr()
    path, _ = lsr.plan(np.array([0.0, 0.0]), np.array([2.0, 0.0]))
    actions = lsr.get_actions_for_path(path)
    assert all(a is not None for a in actions)


# ---- persistence ----

def test_save_load_roundtrip(tmp_path):
    lsr = _simple_lsr()
    p = str(tmp_path / "lsr.pkl")
    lsr.save(p)
    loaded = LSR.load(p)
    result = loaded.plan(np.array([0.0, 0.0]), np.array([3.0, 0.0]))
    assert result is not None
