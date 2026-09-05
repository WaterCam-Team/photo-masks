"""Mask2Former (HF `transformers`) for semantic water segmentation.

Mask2Former predicts a set of binary masks plus a class per mask, and trains by
Hungarian-matching them against the ground-truth masks. Two consequences the
rest of the pipeline has to respect:

* **It brings its own loss.** `--loss ce+lovasz` and friends are pixel-wise
  objectives and cannot be substituted for the matching loss, so `train_loss`
  ignores `loss_fn` here and `supports_pixel_loss` is False. The engine records
  that in the run metadata instead of pretending the flag took effect.
* **Per-pixel scores are derived, not native.** `scores()` combines the mask
  logits with the per-query class probabilities exactly as
  `post_process_semantic_segmentation` does, so evaluation and metrics share
  one code path with SegFormer.

Targets are built straight from the semantic mask (one binary mask per class
present, ignore pixels excluded) rather than going through
`Mask2FormerImageProcessor`, which would re-do the resizing and normalisation
this pipeline has already done under its own geometry rules.
"""
from __future__ import annotations

from .base import SegModel

IGNORE = 255


class Mask2FormerWrapper(SegModel):
    supports_pixel_loss = False

    def _build(self, from_checkpoint: bool):
        from transformers import Mask2FormerForUniversalSegmentation

        if from_checkpoint:
            return Mask2FormerForUniversalSegmentation.from_pretrained(self.init)

        # ADE-pretrained heads carry 150 classes; ours has 2, and the Swin stem
        # needs `in_channels` inputs. Both are shape mismatches, so they are
        # re-initialised and then warm-started where that is meaningful.
        m = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.init, num_labels=self.num_classes, ignore_mismatched_sizes=True)
        self._emit(event="info", msg=f"initialised {self.arch} from {self.init}")
        if self.in_channels != 3:
            self._adapt_backbone_stem(m)
        return m

    def _adapt_backbone_stem(self, m) -> None:
        """Give the Swin backbone an n-channel patch-embedding stem."""
        self._emit(event="info", msg=self.adapt_input_channels(m))
        cfg = getattr(m.config, "backbone_config", None)
        if cfg is not None and hasattr(cfg, "num_channels"):
            cfg.num_channels = self.in_channels     # so save/load round-trips

    # -- inference / training --------------------------------------------
    def scores(self, x):
        import torch
        import torch.nn.functional as F

        out = self.module(pixel_values=x)
        masks = F.interpolate(out.masks_queries_logits, size=x.shape[-2:],
                              mode="bilinear", align_corners=False).sigmoid()
        cls = out.class_queries_logits.softmax(dim=-1)[..., :-1]   # drop "no object"
        return torch.einsum("bqc,bqhw->bchw", cls, masks)

    def _targets(self, y):
        """Semantic (N,H,W) -> per-sample binary masks + class labels."""
        import torch
        mask_labels, class_labels = [], []
        for yi in y:
            classes = [c for c in torch.unique(yi).tolist() if c != IGNORE]
            if not classes:                       # all-ignore tile: one empty target
                mask_labels.append(torch.zeros((0, *yi.shape), dtype=torch.float32,
                                               device=yi.device))
                class_labels.append(torch.zeros((0,), dtype=torch.long, device=yi.device))
                continue
            mask_labels.append(torch.stack([(yi == c).to(torch.float32) for c in classes]))
            class_labels.append(torch.tensor(classes, dtype=torch.long, device=yi.device))
        return mask_labels, class_labels

    def train_loss(self, x, y, loss_fn):
        """Hungarian-matching loss. `loss_fn` does not apply (see module docstring)."""
        mask_labels, class_labels = self._targets(y)
        out = self.module(pixel_values=x, mask_labels=mask_labels,
                          class_labels=class_labels)
        return out.loss
