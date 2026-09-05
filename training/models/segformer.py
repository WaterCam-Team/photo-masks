"""SegFormer (HF `transformers`) for arbitrary input channel counts.

Weights come from the MiT ImageNet encoders (`nvidia/mit-b0` .. `mit-b5`). The
patch-embedding stem is the only shape-incompatible tensor when the input is
not 3-channel, so it is re-created and then warm-started from the RGB filters
(see `base.SegModel.adapt_stem`).
"""
from __future__ import annotations

from pathlib import Path

from .base import SegModel


class SegformerWrapper(SegModel):
    supports_pixel_loss = True

    def _build(self, from_checkpoint: bool):
        import torch
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if from_checkpoint:
            m = SegformerForSemanticSegmentation.from_pretrained(self.init)
            got = m.config.num_channels
            if got != self.in_channels:
                raise ValueError(
                    f"checkpoint {self.init} expects {got}-channel input but this "
                    f"modality has {self.in_channels} channels")
            return m

        try:
            # Load the plain 3-channel encoder, then rebuild the stem. Passing
            # num_channels here would instead create a *randomly initialised*
            # n-channel stem, which is what has to be warm-started anyway.
            m = SegformerForSemanticSegmentation.from_pretrained(
                self.init, num_labels=self.num_classes, ignore_mismatched_sizes=True)
            self._emit(event="info", msg=f"initialised {self.arch} from {self.init}")
        except Exception as e:                              # noqa: BLE001
            self._emit(event="warn",
                       msg=f"pretrained init failed ({e}); training from random init")
            cfg = SegformerConfig(num_labels=self.num_classes,
                                  num_channels=self.in_channels)
            return SegformerForSemanticSegmentation(cfg)

        self._emit(event="info", msg=self.adapt_input_channels(m))
        m.config.num_channels = self.in_channels        # so save/load round-trips
        return m

    def scores(self, x):
        import torch.nn.functional as F
        lo = self.module(pixel_values=x).logits
        return F.interpolate(lo, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def train_loss(self, x, y, loss_fn):
        return loss_fn(self.scores(x), y)

    def export_onnx(self, onnx_out: Path, size: int = 512, opset: int = 17) -> Path:
        """Export for the camera nodes: fixed CHW, dynamic batch, bilinear-upsampled.

        Opset 17, not 13: SegFormer attention in `transformers` 5.x lowers to
        `aten::scaled_dot_product_attention`, which the exporter only supports
        from opset 14. At 13 the export fails outright with
        "Exporting the operator ... is not supported".
        """
        import torch

        onnx_out = Path(onnx_out)
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        self.module.eval()

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                lo = self.m(pixel_values=x).logits
                return torch.nn.functional.interpolate(
                    lo, size=(x.shape[-2], x.shape[-1]), mode="bilinear",
                    align_corners=False)

        dummy = torch.zeros(1, self.in_channels, size, size)
        torch.onnx.export(Wrap(self.module), dummy, str(onnx_out),
                          input_names=["input"], output_names=["logits"],
                          opset_version=opset, dynamo=False,
                          dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}})
        return onnx_out
