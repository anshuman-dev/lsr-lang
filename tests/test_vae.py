"""Unit tests for the convolutional VAE."""

import pytest
import torch
import torch.nn as nn

from models.vae import VAE, ConvEncoder, ConvDecoder, elbo_loss


# ---- fixtures ----

@pytest.fixture
def vae2():
    return VAE(z_dim=2)

@pytest.fixture
def vae4():
    return VAE(z_dim=4)

@pytest.fixture
def batch():
    return torch.randn(8, 3, 64, 64).clamp(0, 1)


# ---- encoder ----

def test_encoder_mu_shape(vae4, batch):
    mu, logvar = vae4.encoder(batch)
    assert mu.shape == (8, 4)

def test_encoder_logvar_shape(vae4, batch):
    mu, logvar = vae4.encoder(batch)
    assert logvar.shape == (8, 4)

def test_encoder_z_dim_2():
    enc = ConvEncoder(z_dim=2)
    x = torch.randn(4, 3, 64, 64)
    mu, _ = enc(x)
    assert mu.shape == (4, 2)

def test_encoder_differentiable(vae4, batch):
    batch.requires_grad_(False)
    mu, logvar = vae4.encoder(batch)
    loss = mu.sum()
    loss.backward()
    # No error means gradients flow


# ---- decoder ----

def test_decoder_output_shape(vae4):
    z = torch.randn(8, 4)
    out = vae4.decode(z)
    assert out.shape == (8, 3, 64, 64)

def test_decoder_output_in_01(vae4):
    z = torch.randn(16, 4)
    out = vae4.decode(z)
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6

def test_decoder_z_dim_2():
    dec = ConvDecoder(z_dim=2)
    z = torch.randn(4, 2)
    out = dec(z)
    assert out.shape == (4, 3, 64, 64)


# ---- reparameterize ----

def test_reparameterize_train_adds_noise(vae4):
    vae4.train()
    mu = torch.zeros(100, 4)
    logvar = torch.zeros(100, 4)
    z = vae4.reparameterize(mu, logvar)
    assert not torch.allclose(z, mu)    # noise added

def test_reparameterize_eval_returns_mu(vae4):
    vae4.eval()
    mu = torch.randn(8, 4)
    logvar = torch.zeros(8, 4)
    z = vae4.reparameterize(mu, logvar)
    assert torch.allclose(z, mu)


# ---- forward ----

def test_vae_forward_shapes(vae4, batch):
    recon, mu, logvar = vae4(batch)
    assert recon.shape  == batch.shape
    assert mu.shape     == (8, 4)
    assert logvar.shape == (8, 4)

def test_vae_forward_z2(vae2, batch):
    recon, mu, logvar = vae2(batch)
    assert mu.shape == (8, 2)

def test_vae_forward_single_image(vae4):
    x = torch.randn(1, 3, 64, 64).clamp(0, 1)
    recon, mu, logvar = vae4(x)
    assert recon.shape == (1, 3, 64, 64)


# ---- ELBO loss ----

def test_elbo_loss_returns_three_values(vae4, batch):
    recon, mu, logvar = vae4(batch)
    result = elbo_loss(recon, batch, mu, logvar)
    assert len(result) == 3

def test_elbo_total_is_sum_of_parts(vae4, batch):
    recon, mu, logvar = vae4(batch)
    total, recon_l, kl_l = elbo_loss(recon, batch, mu, logvar, beta=1.0)
    assert torch.isclose(total, recon_l + kl_l, atol=1e-4)

def test_elbo_beta_scales_kl(vae4, batch):
    recon, mu, logvar = vae4(batch)
    _, _, kl1 = elbo_loss(recon, batch, mu, logvar, beta=1.0)
    _, _, kl2 = elbo_loss(recon, batch, mu, logvar, beta=2.0)
    # kl term itself is the same; the *total* differs
    total1, _, _ = elbo_loss(recon, batch, mu, logvar, beta=1.0)
    total2, _, _ = elbo_loss(recon, batch, mu, logvar, beta=2.0)
    assert total2 > total1 or torch.isclose(kl1, torch.zeros(1))

def test_elbo_recon_loss_positive(vae4, batch):
    recon, mu, logvar = vae4(batch)
    _, recon_l, _ = elbo_loss(recon, batch, mu, logvar)
    assert recon_l.item() >= 0.0

def test_elbo_perfect_recon_has_low_recon_loss():
    x = torch.rand(4, 3, 64, 64)
    mu = torch.zeros(4, 4)
    logvar = torch.zeros(4, 4)
    total, recon_l, _ = elbo_loss(x, x, mu, logvar)
    assert recon_l.item() < 1e-6


# ---- gradient flow ----

def test_gradient_flows_through_vae(vae4, batch):
    vae4.train()
    recon, mu, logvar = vae4(batch)
    loss, _, _ = elbo_loss(recon, batch, mu, logvar)
    loss.backward()
    for name, p in vae4.named_parameters():
        assert p.grad is not None, f"No grad for {name}"

def test_encode_is_deterministic_in_eval(vae4, batch):
    vae4.eval()
    with torch.no_grad():
        z1, _, _ = vae4.encode(batch)
        z2, _, _ = vae4.encode(batch)
    assert torch.allclose(z1, z2)
