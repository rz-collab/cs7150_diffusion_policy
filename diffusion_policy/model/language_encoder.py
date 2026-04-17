# ---
# Generated: 2026-04-14 | claude-opus-4-6
# Prompt: Create a language encoder module supporting both a pure text encoder
#         and a CLIP encoder, with configurable projection dimension and freeze
#         setting, for use as conditioning in the diffusion policy.
# Modifications:
#   2026-04-14 | Prompt: Make pretrained model configurable | Replaced hardcoded
#               CLIP model with a pretrained_model param that looks up the model
#               from PRETRAINED_VISION_MODELS (shared with visual_encoder.py).
#               Supports CLIP and SigLIP family text encoders alongside the
#               standalone "text" backend.
# ---

import contextlib

import torch
import torch.nn as nn

from diffusion_policy.model.visual_encoder import PRETRAINED_VISION_MODELS

# Standalone text encoder (not tied to a vision-language model)
_TEXT_ENCODER_CONFIG: dict = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "output_dim": 384,
}


class LanguageEncoder(nn.Module):
    """Pretrained language encoder with a trainable projection head.

    Supports two backends:
      - "clip": Text encoder from a vision-language model (CLIP or SigLIP),
                selected via pretrained_model.
      - "text": Standalone sentence-transformers encoder (384-dim), lightweight.

    The pretrained encoder can be frozen (default) so only the projection MLP
    trains, or left unfrozen for end-to-end fine-tuning.
    """

    def __init__(
        self,
        encoder_type: str = "clip",
        proj_dim: int = 256,
        freeze: bool = True,
        pretrained_model: str = "clip-vit-b-32",
        text_model: nn.Module | None = None,
    ) -> None:
        """
        Args:
            encoder_type: "clip" for a vision-language model's text encoder,
                          "text" for the standalone sentence-transformer.
            proj_dim: Output dimension of the trainable projection MLP.
            freeze: Freeze pretrained encoder weights.
            pretrained_model: Key into PRETRAINED_VISION_MODELS (used when
                              encoder_type == "clip" and text_model is None).
            text_model: Pre-loaded text encoder module.  When provided, skips
                        loading from HuggingFace (used to share weights with
                        the vision encoder loaded by DiffusionPolicy).
        """
        super().__init__()

        self.encoder_type = encoder_type
        self.freeze = freeze
        self.proj_dim = proj_dim

        if encoder_type == "clip":
            if pretrained_model not in PRETRAINED_VISION_MODELS:
                available = ", ".join(PRETRAINED_VISION_MODELS.keys())
                raise ValueError(
                    f"Unknown pretrained_model '{pretrained_model}'. "
                    f"Available: {available}"
                )
            config = PRETRAINED_VISION_MODELS[pretrained_model]
            hf_name: str = config["hf_name"]
            family: str = config["family"]
            encoder_dim: int = config["output_dim"]
            self._family = family

            if text_model is not None:
                # Use pre-loaded text model (shared with vision encoder)
                self.encoder: nn.Module = text_model
            elif family == "clip":
                from transformers import CLIPTextModel

                self.encoder = CLIPTextModel.from_pretrained(hf_name)
            elif family == "siglip":
                from transformers import SiglipTextModel

                self.encoder = SiglipTextModel.from_pretrained(hf_name)
            else:
                raise ValueError("No encoder was specified for ")

            # Load tokenizer (always needed regardless of text_model source)
            if family == "clip":
                from transformers import CLIPTokenizer

                self._tokenizer = CLIPTokenizer.from_pretrained(hf_name)
            elif family == "siglip":
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(hf_name)

        elif encoder_type == "text":
            from transformers import AutoModel, AutoTokenizer

            model_name = _TEXT_ENCODER_CONFIG["model_name"]
            encoder_dim = _TEXT_ENCODER_CONFIG["output_dim"]
            self._family = "text"
            self.encoder = AutoModel.from_pretrained(model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(
                f"Unknown encoder_type '{encoder_type}'. Use 'clip' or 'text'."
            )

        # Freeze encoder parameters if requested
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Trainable projection MLP: maps raw encoder dim → proj_dim
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, proj_dim),
            nn.Mish(),
            nn.Linear(proj_dim, proj_dim),
        )

    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Tokenize and encode raw text strings into raw encoder embeddings.

        Returns the pretrained encoder output **before** the projection head.
        Respects the freeze setting: when frozen, runs under torch.no_grad();
        when unfrozen, gradients flow through the encoder.

        Args:
            texts:  List of task description strings.
            device: Device to place tensors on.

        Returns:
            Raw encoder embeddings of shape (len(texts), encoder_dim).
        """
        tokens = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with ctx:
            if self._family in ("clip", "siglip"):
                raw: torch.Tensor = self.encoder(**tokens).pooler_output
            elif self._family == "text":
                output = self.encoder(**tokens)
                # Mean-pool over token dimension
                raw = output.last_hidden_state.mean(dim=1)

        return raw

    def forward(self, lang_embedding: torch.Tensor) -> torch.Tensor:
        """Project a raw encoder embedding to the conditioning dimension.

        Args:
            lang_embedding: (B, encoder_dim) raw output from encode_text().

        Returns:
            (B, proj_dim) projected language conditioning vector.
        """
        return self.proj(lang_embedding)
