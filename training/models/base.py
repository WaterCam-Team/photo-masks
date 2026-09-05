"""Common wrapper behaviour: n-channel stems, saving, parameter counts."""
from __future__ import annotations

from pathlib import Path


class SegModel:
    """Base wrapper. Subclasses implement `_build`, `scores` and `train_loss`."""

    supports_pixel_loss = True

    def __init__(self, arch: str, in_channels: int, num_classes: int = 2,
                 init: str | None = None, emit=None, from_checkpoint: bool = False):
        self.arch = arch
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init = init
        self._emit = emit or (lambda **kw: None)
        self.module = self._build(from_checkpoint)

    # -- to implement -----------------------------------------------------
    def _build(self, from_checkpoint: bool):
        raise NotImplementedError

    def scores(self, x):
        """(N,C,H,W) input -> (N,K,H,W) class scores at the input resolution."""
        raise NotImplementedError

    def train_loss(self, x, y, loss_fn):
        raise NotImplementedError

    # -- shared -----------------------------------------------------------
    def to(self, device):
        self.module.to(device)
        return self

    def train(self):
        self.module.train()
        return self

    def eval(self):
        self.module.eval()
        return self

    def parameters(self):
        return self.module.parameters()

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.module.save_pretrained(out_dir)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd, strict: bool = True):
        return self.module.load_state_dict(sd, strict=strict)

    # -- n-channel stem ---------------------------------------------------
    @staticmethod
    def first_conv(module):
        """(path, conv) of the input stem — the first Conv2d in module order.

        Deliberately structural rather than a hard-coded attribute path.
        `transformers` moved SegFormer's stem from
        `segformer.encoder.patch_embeddings[0].proj` to
        `segformer.stages.0.patch_embeddings.proj` between 4.x and 5.x, which
        silently turned the old warm-start into a no-op (it was wrapped in a
        try/except). Verified in both SegFormer and Mask2Former: the stem is
        the first Conv2d `named_modules()` yields.
        """
        import torch
        for name, mod in module.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                return name, mod
        return None, None

    def _set_submodule(self, root, path: str, new) -> None:
        parts = path.split(".")
        obj = root
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], new)

    def adapt_input_channels(self, model) -> str:
        """Replace the stem with an `in_channels` one, warm-started from RGB.

        Raises if the stem cannot be found: a silently-skipped warm-start means
        training a randomly-initialised stem while the log says "pretrained",
        which is exactly the failure this replaces.
        """
        import torch

        path, conv = self.first_conv(model)
        if conv is None:
            raise RuntimeError(f"{self.arch}: no Conv2d stem found to adapt")
        if conv.in_channels == self.in_channels:
            return f"stem already takes {self.in_channels} channel(s)"
        new = torch.nn.Conv2d(
            self.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
            stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
            bias=conv.bias is not None)
        note = self.adapt_stem(new.weight.data, conv.weight.data)
        if conv.bias is not None:
            with torch.no_grad():
                new.bias.data.copy_(conv.bias.data)
        new.to(conv.weight.device, conv.weight.dtype)
        self._set_submodule(model, path, new)
        return f"{note} (at {path})"

    def adapt_stem(self, new_w, old_w) -> str:
        """Copy RGB stem weights into an n-channel stem, in place.

        (out, in, kh, kw) both. RGB channels are copied straight across; every
        extra channel starts as the mean of the RGB filters, which is a far
        better start than random for a band spatially correlated with
        luminance. For a 1-channel modality (thermal, NIR) the mean *is* the
        whole stem — the standard grayscale adaptation.

        `segformer_5band/MODALITY_COMPARISON.md` notes the prior LWIR config
        gave up here and trained from random init, naming this averaging as the
        fix it never applied. This is that fix.
        """
        import torch
        n_new, n_old = new_w.shape[1], old_w.shape[1]
        with torch.no_grad():
            if n_new == n_old:
                new_w.copy_(old_w)
                return f"copied {n_old}-channel stem verbatim"
            mean = old_w.mean(dim=1, keepdim=True)
            if n_new < n_old:                       # e.g. 1-channel thermal
                new_w.copy_(mean.repeat(1, n_new, 1, 1))
                return (f"{n_new}-channel stem warm-started from the mean of "
                        f"{n_old} RGB filters")
            new_w[:, :n_old].copy_(old_w)
            new_w[:, n_old:].copy_(mean.repeat(1, n_new - n_old, 1, 1))
            return (f"{n_new}-channel stem warm-started: RGB copied, "
                    f"{n_new - n_old} extra band(s) from the RGB mean")
