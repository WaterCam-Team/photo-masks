"""Segmentation losses, all `ignore_index`-aware.

`annotator/trainer.py` used bare `F.cross_entropy`. Cross-entropy optimises
per-pixel likelihood, not the IoU everything downstream is actually judged on,
and the audit in `segformer_5band/PERFORMANCE.md` flagged unweighted CE on an
imbalanced problem as its second real bug. So this module offers:

    ce              weighted cross-entropy (weights from measured class balance)
    dice            soft Dice on the water class
    lovasz          Lovasz-softmax — a convex surrogate for IoU itself
    ce+dice         the usual robust pairing
    ce+lovasz       what the prior B2 modality configs used (CE 0.5 / Lovasz 0.5)

Water covers ~46% of the labelled frames here, so class weighting is close to a
no-op for `fiveband` — it earns its keep on scenes where only a puddle is in
view, and on the single-band modalities.

Lovasz-softmax follows Berman, Rannen Triki & Blaschko, "The Lovasz-Softmax
loss" (CVPR 2018); `lovasz_grad` and the per-class flat form are the reference
formulation from the authors' released code.
"""
from __future__ import annotations

IGNORE = 255


def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension of the Jaccard loss."""
    import torch
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return torch.clamp(jaccard, min=0.0)


def lovasz_softmax(logits, target, ignore_index: int = IGNORE, classes="present"):
    """Multiclass Lovasz-softmax over the flattened batch."""
    import torch
    import torch.nn.functional as F

    C = logits.shape[1]
    probas = F.softmax(logits, dim=1)
    probas = probas.permute(0, 2, 3, 1).reshape(-1, C)
    labels = target.reshape(-1)
    keep = labels != ignore_index
    if keep.sum() == 0:
        return logits.sum() * 0.0
    probas, labels = probas[keep], labels[keep]

    losses = []
    for c in range(C):
        fg = (labels == c).to(probas.dtype)
        if classes == "present" and fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def dice_loss(logits, target, ignore_index: int = IGNORE, eps: float = 1.0,
              water_class: int = 1):
    """Soft Dice on the water class."""
    import torch
    import torch.nn.functional as F

    prob = F.softmax(logits, dim=1)[:, water_class]
    keep = target != ignore_index
    if keep.sum() == 0:
        return logits.sum() * 0.0
    p = prob[keep]
    t = (target[keep] == water_class).to(p.dtype)
    inter = torch.sum(p * t)
    return 1.0 - (2 * inter + eps) / (torch.sum(p) + torch.sum(t) + eps)


class SegLoss:
    """Weighted sum of the components named in `spec` (e.g. "ce+lovasz")."""

    SPECS = ("ce", "dice", "lovasz", "ce+dice", "ce+lovasz", "ce+dice+lovasz")

    def __init__(self, spec: str = "ce+lovasz", class_weights=None,
                 ignore_index: int = IGNORE, ce_w: float = 0.5, aux_w: float = 0.5):
        spec = (spec or "ce").lower()
        if spec not in self.SPECS:
            raise ValueError(f"loss must be one of {self.SPECS}, got {spec!r}")
        self.parts = spec.split("+")
        self.spec = spec
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.ce_w = ce_w
        self.aux_w = aux_w
        self._w = None

    def __call__(self, logits, target):
        import torch
        import torch.nn.functional as F

        if self._w is None and self.class_weights:
            self._w = torch.tensor(self.class_weights, dtype=torch.float32,
                                   device=logits.device)
        aux = [p for p in self.parts if p != "ce"]
        share = self.aux_w / len(aux) if aux else 0.0
        total = None
        for part in self.parts:
            if part == "ce":
                v = F.cross_entropy(logits, target, weight=self._w,
                                    ignore_index=self.ignore_index)
                w = self.ce_w if aux else 1.0
            elif part == "dice":
                v, w = dice_loss(logits, target, self.ignore_index), share
            else:
                v, w = lovasz_softmax(logits, target, self.ignore_index), share
            total = v * w if total is None else total + v * w
        return total
