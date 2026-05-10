"""Unit tests for the Action Proposal Module (APM)."""

import pytest
import torch

from models.apm import APM, apm_loss


@pytest.fixture
def apm():
    return APM(z_dim=4, action_dim=4, hidden_dim=64)


def test_output_shape(apm):
    z_i = torch.randn(8, 4)
    z_j = torch.randn(8, 4)
    out = apm(z_i, z_j)
    assert out.shape == (8, 4)

def test_batch_size_1(apm):
    z_i = torch.randn(1, 4)
    z_j = torch.randn(1, 4)
    out = apm(z_i, z_j)
    assert out.shape == (1, 4)

def test_different_z_dim():
    model = APM(z_dim=2, action_dim=6, hidden_dim=32)
    z_i = torch.randn(4, 2)
    z_j = torch.randn(4, 2)
    out = model(z_i, z_j)
    assert out.shape == (4, 6)

def test_different_action_dim():
    model = APM(z_dim=4, action_dim=7)
    z_i = torch.randn(3, 4)
    z_j = torch.randn(3, 4)
    assert model(z_i, z_j).shape == (3, 7)

def test_loss_positive(apm):
    z_i = torch.randn(8, 4)
    z_j = torch.randn(8, 4)
    target = torch.randn(8, 4)
    pred = apm(z_i, z_j)
    loss = apm_loss(pred, target)
    assert loss.item() >= 0.0

def test_loss_zero_perfect_prediction():
    model = APM(z_dim=4, action_dim=4, hidden_dim=32)
    target = torch.ones(4, 4)
    # Manually set output to target
    loss = apm_loss(target, target)
    assert loss.item() < 1e-8

def test_gradient_flows(apm):
    z_i = torch.randn(8, 4)
    z_j = torch.randn(8, 4)
    target = torch.randn(8, 4)
    pred = apm(z_i, z_j)
    loss = apm_loss(pred, target)
    loss.backward()
    for name, p in apm.named_parameters():
        assert p.grad is not None, f"No grad for {name}"

def test_output_changes_with_input(apm):
    apm.eval()
    z_i = torch.randn(4, 4)
    z_j_a = torch.randn(4, 4)
    z_j_b = torch.randn(4, 4)
    out_a = apm(z_i, z_j_a)
    out_b = apm(z_i, z_j_b)
    assert not torch.allclose(out_a, out_b)

def test_hidden_dim_affects_params():
    small = APM(z_dim=4, action_dim=4, hidden_dim=32)
    large = APM(z_dim=4, action_dim=4, hidden_dim=512)
    n_small = sum(p.numel() for p in small.parameters())
    n_large = sum(p.numel() for p in large.parameters())
    assert n_large > n_small
