# ---
# Generated: 2026-04-06 | claude-opus-4-6
# Prompt: Wrapper class that wraps all components (diffusion denoiser, diff step
#         encoder, visual encoder) into a single pytorch model.
# Modifications:
#   2026-04-14 | Prompt: Add exchangeable language conditioning | Added optional
#               language encoder ("clip" or "text") with configurable projection
#               dim and freeze flag.  Model stores its constructor config as
#               model_config for checkpoint serialization.
#   2026-04-14 | Prompt: Add CLIP vision encoder option | Added use_clip_vision
#               flag to swap ResNet-18 for CLIP ViT-B/32 vision encoder, and
#               freeze_visual_encoder flag to control whether it trains.
#   2026-04-14 | Prompt: Consolidate encoder settings into single string |
#               Replaced separate use_clip_vision / lang_encoder_type / freeze
#               flags with a single encoder_type string ("resnet_only",
#               "clip_text", "clip_both", "resnet_and_text") and one
#               freeze_encoders bool.
# ---

import torch
import torch.nn as nn
from diffusion_policy.model.visual_encoder import (
    get_visual_encoder,
    PretrainedVisualEncoder,
    PRETRAINED_VISION_MODELS,
)
from diffusion_policy.model.denoiser import ConditionalUnet1D
from diffusion_policy.model.diffusion_step_encoder import DiffusionStepEncoder
from diffusion_policy.model.language_encoder import LanguageEncoder


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


VALID_ENCODER_TYPES = {"resnet_only", "clip_text", "clip_both", "resnet_and_text"}


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        state_obs_dim: int = 0,
        obs_horizon: int = 2,
        diff_step_dim: int = 256,
        down_dims: list[int] = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
        encoder_type: str = "resnet_only",
        pretrained_model: str = "clip-vit-b-32",
        lang_proj_dim: int = 256,
        freeze_encoders: bool = False,
    ):
        """
        Args:
            encoder_type: Selects vision and language encoders.
                "resnet_only"    — ResNet-18 vision, no language.
                "clip_text"      — ResNet-18 vision + pretrained text encoder.
                "clip_both"      — Pretrained vision + pretrained text encoder.
                "resnet_and_text" — ResNet-18 vision + standalone text encoder.
            pretrained_model: Which pretrained model to use for vision/language
                when encoder_type is "clip_text" or "clip_both".
                See PRETRAINED_VISION_MODELS in visual_encoder.py for options.
            lang_proj_dim: Output dimension of the language projection MLP.
            freeze_encoders: If True, freeze pretrained encoder weights
                             (vision and/or language) so only the projection
                             MLP and the UNet train.
        """
        super().__init__()

        if encoder_type not in VALID_ENCODER_TYPES:
            raise ValueError(
                f"Unknown encoder_type '{encoder_type}'. "
                f"Valid options: {sorted(VALID_ENCODER_TYPES)}"
            )

        # Store config for checkpoint serialization
        self.model_config: dict = {
            "action_dim": action_dim,
            "state_obs_dim": state_obs_dim,
            "obs_horizon": obs_horizon,
            "diff_step_dim": diff_step_dim,
            "down_dims": down_dims,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "encoder_type": encoder_type,
            "pretrained_model": pretrained_model,
            "lang_proj_dim": lang_proj_dim,
            "freeze_encoders": freeze_encoders,
        }

        # Load the full pretrained model once when using clip_both,
        # so vision and text encoders share the exact same weights.
        shared_text_model = None
        if encoder_type == "clip_both":
            config = PRETRAINED_VISION_MODELS[pretrained_model]
            family: str = config["family"]

            if family == "clip":
                from transformers import CLIPModel

                full_model = CLIPModel.from_pretrained(config["hf_name"])
            elif family == "siglip":
                from transformers import SiglipModel

                full_model = SiglipModel.from_pretrained(config["hf_name"])

            self.visual_encoder: nn.Module = PretrainedVisualEncoder(
                model_key=pretrained_model,
                vision_model=full_model.vision_model,
                visual_projection=full_model.visual_projection,
            )
            shared_text_model = full_model.text_model
            visual_obs_dim: int = self.visual_encoder.output_dim
        else:
            self.visual_encoder = get_visual_encoder()
            visual_obs_dim = 512  # ResNet-18 output dim

        if freeze_encoders:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

        # Diffusion step encoder:
        self.diffusion_step_encoder = DiffusionStepEncoder(diff_step_dim)

        cond_dim = (visual_obs_dim + state_obs_dim) * obs_horizon + diff_step_dim

        # Language encoder (optional, depends on encoder_type)
        lang_backend: str | None = None
        if encoder_type in ("clip_text", "clip_both"):
            lang_backend = "clip"
        elif encoder_type == "resnet_and_text":
            lang_backend = "text"

        self.lang_encoder: LanguageEncoder | None = None
        if lang_backend is not None:
            self.lang_encoder = LanguageEncoder(
                encoder_type=lang_backend,
                proj_dim=lang_proj_dim,
                freeze=freeze_encoders,
                pretrained_model=pretrained_model,
                text_model=shared_text_model,
            )
            cond_dim += lang_proj_dim

        # Diffusion denoiser
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            cond_dim=cond_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

        print(f"Encoder type: {encoder_type}, pretrained model: {pretrained_model}")
        print(f"Condition features dimension: {cond_dim}")
        print(f"Number of Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        noisy_action_seq: torch.Tensor,
        diffusion_step: torch.Tensor,
        visual_obs_seq: torch.Tensor,
        state_obs_seq: torch.Tensor | None = None,
        task_description: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            noisy_action_seq: (B, pred_action_horizon, action_dim)
            diffusion_step: (B,)
            visual_obs_seq: (B, obs_horizon, C, H, W)
            state_obs_seq: (B, obs_horizon, state_dim)
            task_description: List of B task description strings. Encoded
                              internally by the language encoder + projection.
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

        # Build conditioning vector
        cond_parts: list[torch.Tensor] = [diff_step_embed, obs_embeds]

        # Language conditioning: encode raw text or pad with zeros when dropped
        if self.lang_encoder is not None:
            if task_description is not None:
                raw_embedding = self.lang_encoder.encode_text(
                    task_description, noisy_action_seq.device
                )
                lang_cond = self.lang_encoder(raw_embedding)  # (B, lang_proj_dim)
            else:
                lang_cond = torch.zeros(
                    B, self.lang_encoder.proj_dim, device=noisy_action_seq.device
                )
            cond_parts.append(lang_cond)

        # Concatenate all conditioning inputs (B, cond_dim)
        cond = torch.cat(cond_parts, dim=-1)

        # Predict noise of the given noisy action sequence
        pred_action_seq_noise = self.noise_pred_net(
            noisy_action_seq,
            cond,
        )

        return pred_action_seq_noise
