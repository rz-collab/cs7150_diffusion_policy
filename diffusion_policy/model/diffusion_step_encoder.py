import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_length: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_length = max_length

    def forward(self, x):
        """
        Convert diffusion timesteps into sinusoidal positional embeddings

        Args:
        - x: Tensor of shape (B,) containing diffusion steps

        Return:
        - pe: Tensor of shape (B, dim)
        """
        half_dim = self.dim // 2
        i_range = torch.arange(half_dim, device=x.device).unsqueeze(0)  # (1, half_dim)
        pos_range = x.unsqueeze(1)  # (B, 1)

        pe_arg = pos_range / (
            self.max_length ** (2 * i_range / self.dim)
        )  # (B, half_dim)

        pe = torch.stack((pe_arg.sin(), pe_arg.cos()), dim=-1)
        pe = torch.flatten(pe, start_dim=-2)  # (B, dim)
        return pe


class DiffusionStepEncoder(nn.Module):
    def __init__(self, diff_step_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPositionEmbedding(diff_step_dim),
            nn.Linear(diff_step_dim, diff_step_dim * 4),
            nn.Mish(),
            nn.Linear(diff_step_dim * 4, diff_step_dim),
        )

    def forward(self, t):
        return self.net(t)
