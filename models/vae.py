import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, z_dim: int = 4, img_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),   # 64→32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),              # 32→16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),             # 16→8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),            # 8→4
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, z_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, z_dim: int = 4, img_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),          # 4→8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),           # 8→16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),            # 16→32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),  # 32→64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.net(h)


class VAE(nn.Module):
    def __init__(self, z_dim: int = 4, img_channels: int = 3):
        super().__init__()
        self.encoder = ConvEncoder(z_dim, img_channels)
        self.decoder = ConvDecoder(z_dim, img_channels)
        self.z_dim = z_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def encode(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar


def elbo_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
    return recon_loss + beta * kl, recon_loss, kl
