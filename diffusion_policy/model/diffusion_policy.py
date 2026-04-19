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
#   2026-04-15 | Prompt: Add DINOv2 encoder support | Added "dino_only",
#               "dino_clip_text", and "dino_text" encoder types. DINOv2 vision
#               is paired with no language, a pretrained CLIP/SigLIP text
#               encoder, or a standalone text encoder respectively.
#   2026-04-15 | Prompt: Replace encoder_type with vision/text encoder keys |
#               Replaced encoder_type, pretrained_model, text_pretrained_model
#               with two simple params: vision_encoder (model key or None for
#               ResNet-18) and text_encoder (model key, "text", or None for no
#               language). When both point to the same clip/siglip model,
#               weights are shared. Old checkpoint model_configs are converted
#               automatically.
#   2026-04-15 | Prompt: Add vision projection layer | Added vision_proj_dim
#               parameter (default 512) that adds a trainable projection MLP
#               to pretrained vision encoders, matching ResNet-18 output dim.
#               Does not apply to ResNet-18 itself. Legacy checkpoint configs
#               default to None (no projection) for backwards compatibility.
#   2026-04-17 | Prompt: Option B freeze_backbone — projection layer should not
#               freeze with the backbone. Replaced the blanket parameters() loop
#               with a call to visual_encoder.freeze_backbone() for pretrained
#               encoders (which only freezes the backbone, not self.proj).
#               Falls back to freezing all parameters for ResNet-18, which has
#               no projection layer.
#   2026-04-17 | Prompt: Update ResNet for VisualEncoder ABC — all encoders now
#               subclass VisualEncoder. Updated self.visual_encoder type
#               annotation to VisualEncoder, removed hasattr fallback (ResNet
#               raises NotImplementedError), and replaced hardcoded ResNet-18
#               output_dim with self.visual_encoder.output_dim.
#   2026-04-18 | Prompt: Fix AttributeError on SiglipModel.visual_projection |
#               Used getattr(..., None) when passing visual_projection in the
#               shared-weights branch so SigLIP (which has no projection layer)
#               falls through to the nn.Identity() fallback in PretrainedVisualEncoder.
# ---

import torch
import torch.nn as nn
from diffusion_policy.model.encoder_base import Encoder
from diffusion_policy.model.visual_encoder import (
    get_visual_encoder,
    PretrainedVisualEncoder,
    DINOv2VisualEncoder,
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


def _convert_legacy_model_config(cfg: dict) -> dict:
    """Convert old encoder_type-based model configs to vision/text_encoder style."""
    if "vision_encoder" in cfg:
        # Convert old freeze_encoders to split flags if needed
        if "freeze_encoders" in cfg:
            freeze: bool = cfg.pop("freeze_encoders")
            cfg.setdefault("freeze_vision_encoder", freeze)
            cfg.setdefault("freeze_text_encoder", freeze)
        # Old checkpoints without vision projection — default to None
        cfg.setdefault("vision_proj_dim", None)
        return cfg

    encoder_type: str = cfg.pop("encoder_type", "resnet_only")
    pretrained_model: str = cfg.pop("pretrained_model", "clip-vit-b-32")
    text_pretrained_model: str | None = cfg.pop("text_pretrained_model", None)
    freeze = cfg.pop("freeze_encoders", False)
    cfg["freeze_vision_encoder"] = freeze
    cfg["freeze_text_encoder"] = freeze

    if encoder_type in ("clip_both",):
        cfg["vision_encoder"] = pretrained_model
        cfg["text_encoder"] = text_pretrained_model or pretrained_model
    elif encoder_type in ("clip_text", "dino_clip_text"):
        cfg["vision_encoder"] = pretrained_model if encoder_type.startswith("dino") else None
        cfg["text_encoder"] = text_pretrained_model or pretrained_model
    elif encoder_type in ("resnet_and_text", "dino_text"):
        cfg["vision_encoder"] = pretrained_model if encoder_type.startswith("dino") else None
        cfg["text_encoder"] = "text"
    elif encoder_type in ("dino_only",):
        cfg["vision_encoder"] = pretrained_model
        cfg["text_encoder"] = None
    else:
        cfg["vision_encoder"] = None
        cfg["text_encoder"] = None
    return cfg


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
        vision_encoder: str | None = None,
        text_encoder: str | None = None,
        vision_proj_dim: int | None = 512,
        lang_proj_dim: int = 256,
        freeze_vision_encoder: bool = False,
        freeze_text_encoder: bool = False,
        **kwargs,
    ):
        """
        Args:
            vision_encoder: Key into PRETRAINED_VISION_MODELS for the vision
                backbone (e.g. "siglip2-base-patch16-224", "dinov2-base").
                None uses ResNet-18.
            text_encoder: Key into PRETRAINED_VISION_MODELS for a clip/siglip
                text encoder, "text" for the standalone sentence-transformer,
                or None to disable language conditioning.
            vision_proj_dim: Output dimension of the trainable vision
                projection MLP. Only applies to pretrained vision encoders
                (not ResNet-18). None disables projection. Defaults to 512
                to match ResNet-18 output.
            lang_proj_dim: Output dimension of the language projection MLP.
            freeze_vision_encoder: If True, freeze pretrained vision encoder
                weights so only the UNet trains on visual features.
            freeze_text_encoder: If True, freeze pretrained text encoder
                weights so only the projection MLP trains.
        """
        super().__init__()

        # Store config for checkpoint serialization
        self.model_config: dict = {
            "action_dim": action_dim,
            "state_obs_dim": state_obs_dim,
            "obs_horizon": obs_horizon,
            "diff_step_dim": diff_step_dim,
            "down_dims": down_dims,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "vision_encoder": vision_encoder,
            "text_encoder": text_encoder,
            "vision_proj_dim": vision_proj_dim,
            "lang_proj_dim": lang_proj_dim,
            "freeze_vision_encoder": freeze_vision_encoder,
            "freeze_text_encoder": freeze_text_encoder,
        }

        # --- Vision encoder ---
        shared_text_model = None
        if vision_encoder is not None:
            vis_config = PRETRAINED_VISION_MODELS[vision_encoder]
            vis_family: str = vis_config["family"]

            if vis_family == "dino":
                self.visual_encoder: Encoder = DINOv2VisualEncoder(
                    model_key=vision_encoder,
                    proj_dim=vision_proj_dim,
                )
            elif (
                vis_family in ("clip", "siglip")
                and text_encoder == vision_encoder
            ):
                # Same model for vision and text — share weights
                if vis_family == "clip":
                    from transformers import CLIPModel
                    full_model = CLIPModel.from_pretrained(vis_config["hf_name"])
                else:
                    from transformers import SiglipModel
                    full_model = SiglipModel.from_pretrained(vis_config["hf_name"])

                self.visual_encoder = PretrainedVisualEncoder(
                    model_key=vision_encoder,
                    vision_model=full_model.vision_model,
                    visual_projection=getattr(full_model, "visual_projection", None),
                    proj_dim=vision_proj_dim,
                )
                shared_text_model = full_model.text_model
            else:
                self.visual_encoder = PretrainedVisualEncoder(
                    model_key=vision_encoder,
                    proj_dim=vision_proj_dim,
                )

            visual_obs_dim: int = self.visual_encoder.output_dim
        else:
            self.visual_encoder = get_visual_encoder()
            visual_obs_dim = self.visual_encoder.output_dim

        if freeze_vision_encoder:
            self.visual_encoder.freeze_backbone()

        # Diffusion step encoder:
        self.diffusion_step_encoder = DiffusionStepEncoder(diff_step_dim)

        cond_dim = (visual_obs_dim + state_obs_dim) * obs_horizon + diff_step_dim

        # --- Language encoder (optional) ---
        self.lang_encoder: LanguageEncoder | None = None
        if text_encoder is not None:
            lang_backend: str = "text" if text_encoder == "text" else "clip"
            self.lang_encoder = LanguageEncoder(
                encoder_type=lang_backend,
                proj_dim=lang_proj_dim,
                freeze=freeze_text_encoder,
                pretrained_model=text_encoder,
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

        print(f"Vision encoder: {vision_encoder or 'resnet18'}, "
              f"Text encoder: {text_encoder or 'none'}")
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
