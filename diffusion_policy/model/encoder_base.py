from abc import ABC, abstractmethod

import torch.nn as nn


class Encoder(ABC, nn.Module):
    """Abstract base class for all encoders (visual and language).

    Subclasses must implement freeze_backbone() to define which parameters
    belong to the pretrained backbone (frozen) vs. the trainable projection.
    """

    output_dim: int

    @abstractmethod
    def freeze_backbone(self) -> None:
        """Freeze the pretrained backbone weights, leaving any trainable
        projection head unfrozen."""
        ...
