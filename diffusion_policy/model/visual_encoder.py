# ---
# Generated: 2026-04-06 | claude-opus-4-6
# Prompt: Resnet18 visual encoder, copied from Diffusion Policy Colab's Notebook.
# Modifications:
#   2026-04-14 | Prompt: Add CLIP vision encoder option | Added CLIPVisualEncoder
#               that wraps CLIP ViT-B/32 vision model.  Resizes and normalizes
#               inputs to CLIP's expected format, outputs 512-dim features matching
#               the ResNet-18 interface.
#   2026-04-15 | Prompt: Add DINOv2 vision encoder | Added dinov2-small/base/large
#               entries to PRETRAINED_VISION_MODELS and DINOv2VisualEncoder class
#               that wraps HuggingFace Dinov2Model, using CLS token output with
#               ImageNet normalization and auto-resize.
#   2026-04-15 | Prompt: Add vision projection layer | Added optional proj_dim
#               parameter to PretrainedVisualEncoder and DINOv2VisualEncoder.
#               When set, a trainable 2-layer MLP (Linear→Mish→Linear) projects
#               encoder output to proj_dim, updating output_dim accordingly.
#               None (default) preserves original behavior with no projection.
# ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Callable


def get_visual_encoder() -> nn.Module:
    resnet = get_resnet("resnet18")
    resnet_with_gn = replace_bn_with_gn(resnet)
    return resnet_with_gn


def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, features_per_group: int = 16
) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    _replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group, num_channels=x.num_features
        ),
    )
    return root_module


PRETRAINED_VISION_MODELS: dict[str, dict] = {
    "clip-vit-b-32": {
        "hf_name": "openai/clip-vit-base-patch32",
        "family": "clip",
        "output_dim": 512,
        "image_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
    "clip-vit-b-16": {
        "hf_name": "openai/clip-vit-base-patch16",
        "family": "clip",
        "output_dim": 512,
        "image_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
    "siglip-base-patch16-224": {
        "hf_name": "google/siglip-base-patch16-224",
        "family": "siglip",
        "output_dim": 768,
        "image_size": 224,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
    "siglip2-base-patch16-224": {
        "hf_name": "google/siglip2-base-patch16-224",
        "family": "siglip",
        "output_dim": 768,
        "image_size": 224,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
    "dinov2-base": {
        "hf_name": "facebook/dinov2-base",
        "family": "dino",
        "output_dim": 768,
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "dinov2-small": {
        "hf_name": "facebook/dinov2-small",
        "family": "dino",
        "output_dim": 384,
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "dinov2-large": {
        "hf_name": "facebook/dinov2-large",
        "family": "dino",
        "output_dim": 1024,
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


class DINOv2VisualEncoder(nn.Module):
    """Pretrained DINOv2 vision encoder.

    Uses the CLS token output from DINOv2's ViT backbone. Handles resizing
    and ImageNet normalization internally, so the caller can pass images in
    the same [0,1] (H,W) format used by other encoders.
    """

    def __init__(
        self,
        model_key: str = "dinov2-base",
        proj_dim: int | None = None,
    ) -> None:
        super().__init__()

        if model_key not in PRETRAINED_VISION_MODELS:
            raise ValueError(f"Unknown model_key '{model_key}'")

        config = PRETRAINED_VISION_MODELS[model_key]
        if config["family"] != "dino":
            raise ValueError(f"'{model_key}' is not a DINOv2 model")

        from transformers import Dinov2Model

        self.encoder: nn.Module = Dinov2Model.from_pretrained(config["hf_name"])
        encoder_dim: int = config["output_dim"]
        self._image_size: int = config["image_size"]

        # Trainable projection MLP (encoder_dim → proj_dim)
        if proj_dim is not None:
            self.proj: nn.Module | None = nn.Sequential(
                nn.Linear(encoder_dim, proj_dim),
                nn.Mish(),
                nn.Linear(proj_dim, proj_dim),
            )
            self.output_dim: int = proj_dim
        else:
            self.proj = None
            self.output_dim = encoder_dim

        self.register_buffer(
            "_mean", torch.tensor(config["mean"]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor(config["std"]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) images in [0, 1] range, any spatial size.
        Returns:
            (B, output_dim) CLS token feature vectors.
        """
        sz = self._image_size
        if x.shape[-2] != sz or x.shape[-1] != sz:
            x = F.interpolate(
                x, size=(sz, sz), mode="bilinear", align_corners=False,
            )

        x = (x - self._mean) / self._std

        output = self.encoder(pixel_values=x)
        cls_token: torch.Tensor = output.last_hidden_state[:, 0]

        if self.proj is not None:
            cls_token = self.proj(cls_token)

        return cls_token


class PretrainedVisualEncoder(nn.Module):
    """Pretrained vision encoder (CLIP or SigLIP family).

    Handles resizing and model-specific normalization internally, so the
    caller can pass images in the same [0,1] (H,W) format used by ResNet-18.
    """

    def __init__(
        self,
        model_key: str = "clip-vit-b-32",
        vision_model: nn.Module | None = None,
        visual_projection: nn.Module | None = None,
        proj_dim: int | None = None,
    ) -> None:
        """
        Args:
            model_key: Key into PRETRAINED_VISION_MODELS for config and
                       (if vision_model is None) for loading weights.
            vision_model: Pre-loaded vision backbone. When provided together
                          with visual_projection, skips loading from HuggingFace.
            visual_projection: Pre-loaded projection layer.
            proj_dim: If set, adds a trainable projection MLP mapping the
                      pretrained output to this dimension.
        """
        super().__init__()

        if model_key not in PRETRAINED_VISION_MODELS:
            available = ", ".join(PRETRAINED_VISION_MODELS.keys())
            raise ValueError(
                f"Unknown model_key '{model_key}'. Available: {available}"
            )

        config = PRETRAINED_VISION_MODELS[model_key]
        encoder_dim: int = config["output_dim"]
        self._image_size: int = config["image_size"]

        if vision_model is not None and visual_projection is not None:
            self.vision_model: nn.Module = vision_model
            self.visual_projection: nn.Module = visual_projection
        else:
            hf_name: str = config["hf_name"]
            family: str = config["family"]

            if family == "clip":
                from transformers import CLIPModel

                model = CLIPModel.from_pretrained(hf_name)
            elif family == "siglip":
                from transformers import SiglipModel

                model = SiglipModel.from_pretrained(hf_name)

            self.vision_model = model.vision_model
            self.visual_projection = model.visual_projection

        # Trainable projection MLP (encoder_dim → proj_dim)
        if proj_dim is not None:
            self.proj: nn.Module | None = nn.Sequential(
                nn.Linear(encoder_dim, proj_dim),
                nn.Mish(),
                nn.Linear(proj_dim, proj_dim),
            )
            self.output_dim: int = proj_dim
        else:
            self.proj = None
            self.output_dim = encoder_dim

        # Normalization stats the model was pretrained with (move with .to(device))
        self.register_buffer(
            "_mean", torch.tensor(config["mean"]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor(config["std"]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) images in [0, 1] range, any spatial size.
        Returns:
            (B, output_dim) feature vectors.
        """
        sz = self._image_size
        if x.shape[-2] != sz or x.shape[-1] != sz:
            x = F.interpolate(
                x, size=(sz, sz), mode="bilinear", align_corners=False,
            )

        x = (x - self._mean) / self._std

        vision_out = self.vision_model(pixel_values=x).pooler_output
        projected: torch.Tensor = self.visual_projection(vision_out)

        if self.proj is not None:
            projected = self.proj(projected)

        return projected
