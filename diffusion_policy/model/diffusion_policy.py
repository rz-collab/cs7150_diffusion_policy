import torch
import torch.nn as nn
from diffusion_policy.model.visual_encoder import get_resnet, replace_bn_with_gn
from diffusion_policy.model.denoiser import ConditionalUnet1D

"""
Wrapper class that wraps all components (diffusion denoiser, diff step encoder, visual encoder) into a single pytorch model
"""


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


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        action_dim,
        state_obs_dim=0,
        obs_horizon=2,
        diff_step_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    ):
        # Visual Encoder: Resnet18 with Group Normalization, Outputs feature dim=512
        self.visual_encoder = get_resnet("resnet18")
        self.visual_encoder = replace_bn_with_gn(self.visual_encoder)

        # Diffusion step encoder:
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPositionEmbedding(diff_step_dim),
            nn.Linear(diff_step_dim, diff_step_dim * 4),
            nn.Mish(),
            nn.Linear(diff_step_dim * 4, diff_step_dim),
        )

        visual_obs_dim = 512
        cond_dim = (visual_obs_dim + state_obs_dim) * obs_horizon + diff_step_dim

        # Diffusion denoiser
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            cond_dim=cond_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

    def forward(
        self,
        noisy_action_seq,
        diffusion_step,
        visual_obs_seq,
        state_obs_seq=None,
    ):
        """
        Args:
            noisy_action_seq: (B, pred_action_horizon, action_dim)
            diffusion_step: (B,)
            visual_obs_seq: (B, obs_horizon, C, H, W)
            state_obs_seq: (B, obs_horizon, state_dim)
        Returns:
            (B, pred_action_horizon, action_dim) predicted noise
        """

        # Observation embedding:
        # Process each image independently with vision encoder
        imgs = visual_obs_seq
        B = imgs.shape[0]
        imgs = imgs.flatten(end_dim=1)  # (B * obs_horizon, 3, H, W)
        imgs_embeds = self.visual_encoder(imgs)  # (B * obs_horizon, 512)
        imgs_embeds = imgs_embeds.reshape((B, -1))  # (B, obs_horizon * 512)
        obs_embeds = imgs_embeds

        # Concatenate with state observations if provided.
        if state_obs_seq is not None:
            state_embeds = state_obs_seq.reshape((B, -1))
            obs_embeds = torch.cat([imgs_embeds, state_embeds], dim=-1)

        # Diffusion step embedding (B, diff_step_dim)
        diff_step_embed = self.diffusion_step_encoder(diffusion_step)

        # Concatenate both embeddings as condition input (B, cond_dim)
        cond = torch.cat((diff_step_embed, obs_embeds), dim=-1)

        # Predict noise of the given noisy action sequence
        pred_action_seq_noise = self.noise_pred_net(
            noisy_action_seq,
            cond,
        )

        return pred_action_seq_noise
