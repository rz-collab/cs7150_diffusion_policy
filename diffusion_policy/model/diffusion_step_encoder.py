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
