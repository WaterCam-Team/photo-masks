"""Model registry.

Every architecture is wrapped so the training engine can stay architecture-
agnostic:

    m = build("segformer-b0", in_channels=5, num_classes=2)
    scores = m.scores(x)              # (N,K,H,W) at input resolution
    loss   = m.train_loss(x, y, fn)   # per-architecture training objective

The one asymmetry worth knowing about: SegFormer is a per-pixel classifier, so
any of `losses.SegLoss`'s components apply to it. Mask2Former is a *mask
classifier* trained with Hungarian matching between predicted and target masks;
its objective is not interchangeable with pixel-wise CE, so it computes its own
loss internally and `--loss` does not apply. `supports_pixel_loss` says which
is which, and the engine records it in the run metadata rather than silently
ignoring the flag.
"""
from __future__ import annotations

from .base import SegModel
from .mask2former import Mask2FormerWrapper
from .segformer import SegformerWrapper

#: arch name -> (wrapper class, default pretrained init)
REGISTRY: dict[str, tuple[type[SegModel], str]] = {
    "segformer-b0": (SegformerWrapper, "nvidia/mit-b0"),
    "segformer-b1": (SegformerWrapper, "nvidia/mit-b1"),
    "segformer-b2": (SegformerWrapper, "nvidia/mit-b2"),
    "segformer-b3": (SegformerWrapper, "nvidia/mit-b3"),
    "segformer-b4": (SegformerWrapper, "nvidia/mit-b4"),
    "segformer-b5": (SegformerWrapper, "nvidia/mit-b5"),
    "mask2former-tiny": (Mask2FormerWrapper, "facebook/mask2former-swin-tiny-ade-semantic"),
    "mask2former-small": (Mask2FormerWrapper, "facebook/mask2former-swin-small-ade-semantic"),
    "mask2former-base": (Mask2FormerWrapper, "facebook/mask2former-swin-base-ade-semantic"),
}

ARCHS = tuple(REGISTRY)


def build(arch: str, in_channels: int, num_classes: int = 2, init: str | None = None,
          emit=None) -> SegModel:
    if arch not in REGISTRY:
        raise KeyError(f"unknown arch {arch!r}; choose from {', '.join(ARCHS)}")
    cls, default_init = REGISTRY[arch]
    return cls(arch=arch, in_channels=in_channels, num_classes=num_classes,
               init=init or default_init, emit=emit)


def load(arch: str, path, in_channels: int, num_classes: int = 2, emit=None) -> SegModel:
    """Rebuild a saved run's model from its checkpoint directory."""
    cls, _ = REGISTRY[arch]
    return cls(arch=arch, in_channels=in_channels, num_classes=num_classes,
               init=str(path), emit=emit, from_checkpoint=True)
