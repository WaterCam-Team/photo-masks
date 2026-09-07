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

    def export_onnx(self, onnx_out: Path, size: int = 512, opset: int = 17,
                    stats=None, embed_norm: bool = True) -> Path:
        """Export for the camera nodes: fixed CHW, dynamic batch, bilinear-upsampled.

        Opset 17, not 13: SegFormer attention in `transformers` 5.x lowers to
        `aten::scaled_dot_product_attention`, which the exporter only supports
        from opset 14. At 13 the export fails outright with
        "Exporting the operator ... is not supported".

        With `stats` and `embed_norm`, the band normalisation is compiled into
        the graph as a Sub/Div (or clip for percentile) on the input, and the
        deployed model takes **raw resized bands in 0-255 units**. That removes
        the one failure a sidecar file cannot prevent: a runtime normalising
        the input differently from how the model was trained, silently and
        without error. Either way the graph is stamped with metadata saying
        what it expects, so a runtime can check rather than assume.

        Per-image `minmax` cannot be embedded — it depends on the image, not on
        the model — so that one stays external and is declared as such.
        """
        import torch

        onnx_out = Path(onnx_out)
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        self.module.eval()

        method = getattr(stats, "method", None) if stats is not None else None
        embed = bool(stats is not None and embed_norm and method in ("meanstd", "percentile"))

        class Wrap(torch.nn.Module):
            def __init__(self, m, lo=None, scale=None, clip=False):
                super().__init__()
                self.m = m
                self.clip = clip
                if lo is not None:
                    self.register_buffer("lo", lo)
                    self.register_buffer("scale", scale)
                self.embed = lo is not None

            def forward(self, x):
                if self.embed:
                    x = (x - self.lo) / self.scale
                    if self.clip:
                        x = torch.clamp(x, 0.0, 1.0)
                lo = self.m(pixel_values=x).logits
                return torch.nn.functional.interpolate(
                    lo, size=(x.shape[-2], x.shape[-1]), mode="bilinear",
                    align_corners=False)

        if embed:
            t = lambda v: torch.tensor(v, dtype=torch.float32).view(1, -1, 1, 1)  # noqa: E731
            if method == "meanstd":
                wrap = Wrap(self.module, t(stats.mean),
                            torch.clamp(t(stats.std), min=1e-6))
            else:
                lo, hi = t(stats.p_lo), t(stats.p_hi)
                wrap = Wrap(self.module, lo, torch.clamp(hi - lo, min=1e-6), clip=True)
        else:
            wrap = Wrap(self.module)

        dummy = torch.zeros(1, self.in_channels, size, size)
        torch.onnx.export(wrap, dummy, str(onnx_out),
                          input_names=["input"], output_names=["logits"],
                          opset_version=opset, dynamo=False,
                          dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}})
        self._stamp(onnx_out, stats, embed)
        return onnx_out

    def _stamp(self, onnx_out: Path, stats, embed: bool) -> None:
        """Record in the graph what preprocessing it expects.

        `onnxruntime` exposes these as
        `session.get_modelmeta().custom_metadata_map`, so a deployment can read
        them without the `onnx` package installed.
        """
        import json as _json

        import onnx

        m = onnx.load(str(onnx_out))
        meta = {
            "arch": self.arch,
            "in_channels": str(self.in_channels),
            "num_classes": str(self.num_classes),
            "input_layout": "NCHW",
            "input_range": "raw_0_255" if embed else "normalised",
            "normalization": ("embedded:" + stats.method) if embed
            else ("external:" + (getattr(stats, "method", None) or "minmax")),
        }
        if stats is not None:
            meta["modality"] = stats.modality
            meta["bands"] = ",".join(stats.band_names)
            meta["norm_mean"] = _json.dumps(stats.mean)
            meta["norm_std"] = _json.dumps(stats.std)
            meta["norm_p_lo"] = _json.dumps(stats.p_lo)
            meta["norm_p_hi"] = _json.dumps(stats.p_hi)
        del m.metadata_props[:]
        for k, v in meta.items():
            e = m.metadata_props.add()
            e.key, e.value = k, str(v)
        onnx.save(m, str(onnx_out))
