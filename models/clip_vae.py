"""
CLIP-conditioned VAE.

The encoder concatenates a projected CLIP text embedding with the CNN
image features before the latent bottleneck.  The decoder is identical
to the baseline VAE.

CLIP is loaded once and kept frozen; only the linear projector and the
rest of the VAE are trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vae import ConvDecoder

CLIP_DIM = 512      # openai/clip-vit-base-patch32 text embedding size
PROJ_DIM = 64       # projected text feature size


class TextProjector(nn.Module):
    def __init__(self, in_dim: int = CLIP_DIM, out_dim: int = PROJ_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(text_emb), inplace=True)


class ClipCondEncoder(nn.Module):
    def __init__(
        self,
        z_dim: int = 4,
        img_channels: int = 3,
        proj_dim: int = PROJ_DIM,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.text_proj = TextProjector(CLIP_DIM, proj_dim)
        feat_dim = 256 * 4 * 4 + proj_dim
        self.fc_mu = nn.Linear(feat_dim, z_dim)
        self.fc_logvar = nn.Linear(feat_dim, z_dim)

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor):
        img_feat = self.conv(x).flatten(1)
        txt_feat = self.text_proj(text_emb)
        h = torch.cat([img_feat, txt_feat], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class ClipVAE(nn.Module):
    def __init__(self, z_dim: int = 4, img_channels: int = 3):
        super().__init__()
        self.encoder = ClipCondEncoder(z_dim, img_channels)
        self.decoder = ConvDecoder(z_dim, img_channels)
        self.z_dim = z_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def encode(self, x: torch.Tensor, text_emb: torch.Tensor):
        mu, logvar = self.encoder(x, text_emb)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, text_emb: torch.Tensor):
        z, mu, logvar = self.encode(x, text_emb)
        recon = self.decode(z)
        return recon, mu, logvar


# ------------------------------------------------------------------
# CLIP helper — load once, freeze, cache text embeddings
# ------------------------------------------------------------------

_clip_model = None
_clip_tokenizer = None


def get_clip(device: torch.device):
    global _clip_model, _clip_tokenizer
    if _clip_model is None:
        from transformers import CLIPModel, CLIPTokenizer
        _clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        for p in _clip_model.parameters():
            p.requires_grad_(False)
    return _clip_model.to(device), _clip_tokenizer


@torch.no_grad()
def encode_text(texts: list, device: torch.device) -> torch.Tensor:
    model, tokenizer = get_clip(device)
    tokens = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    out = model.get_text_features(**tokens)
    # get_text_features returns a plain tensor in newer transformers
    emb = out if isinstance(out, torch.Tensor) else out.pooler_output
    emb = emb / emb.norm(dim=-1, keepdim=True)     # L2-normalise
    return emb.float()
