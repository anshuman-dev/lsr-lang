import torch
import torch.nn as nn
import torch.nn.functional as F


class APM(nn.Module):
    """Action Proposal Module: predicts the action bridging two latent states."""

    def __init__(self, z_dim: int = 4, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.z_dim = z_dim
        self.action_dim = action_dim

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_i, z_j], dim=-1))


def apm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)
