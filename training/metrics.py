"""Evaluation metrics.

The confusion matrix is accumulated over whole frames at native resolution, so
mIoU here means what it means in a paper — not "mIoU on 512x512 squashed
thumbnails".

Boundary F1 is imported from `agreement.py` rather than reimplemented, which is
the point: the waterline metric scoring the model is the *same function* used
for inter-annotator agreement, so "is the model inside the range two humans
disagree by?" is a question the numbers can answer directly.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from agreement import boundary_f1                    # noqa: E402  (one definition)

IGNORE = 255


@dataclass
class Metrics:
    """Confusion-matrix accumulator for `num_classes` classes."""

    num_classes: int = 2
    water_class: int = 1
    cm: np.ndarray = field(default=None)
    boundary: list[float] = field(default_factory=list)

    def __post_init__(self):
        if self.cm is None:
            self.cm = np.zeros((self.num_classes, self.num_classes), np.int64)

    def add(self, pred: np.ndarray, true: np.ndarray, boundary_tol: int | None = None):
        p, t = np.asarray(pred).ravel(), np.asarray(true).ravel()
        k = t != IGNORE
        p, t = p[k], t[k]
        n = self.num_classes
        self.cm += np.bincount(n * t.astype(np.int64) + p.astype(np.int64),
                               minlength=n * n).reshape(n, n)
        if boundary_tol:
            # boundary_f1 is a 2-D operation, and averaging it per frame is the
            # honest form anyway: a big frame must not dominate the waterline
            # score of a small one.
            pf, tf = np.asarray(pred), np.asarray(true)
            if pf.ndim == 2:
                pf, tf = pf[None], tf[None]
            for pi, ti in zip(pf.reshape(-1, *pf.shape[-2:]),
                              tf.reshape(-1, *tf.shape[-2:])):
                v = boundary_f1(pi == self.water_class, ti == self.water_class,
                                boundary_tol)
                if v is not None:
                    self.boundary.append(float(v))

    # -- derived numbers ------------------------------------------------
    @property
    def iou(self) -> list[float | None]:
        cm = self.cm
        denom = cm.sum(1) + cm.sum(0) - np.diag(cm)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(denom == 0, np.nan, np.diag(cm) / denom)
        return [None if np.isnan(x) else float(x) for x in v]

    @property
    def miou(self) -> float:
        """Mean over classes *present* in the data — an absent class is nan,
        not 0, so a frame with no water can't quietly halve the score."""
        v = np.array([np.nan if x is None else x for x in self.iou])
        return float(np.nanmean(v)) if np.isfinite(v).any() else 0.0

    def prf(self, c: int | None = None) -> tuple[float, float, float]:
        """precision, recall, F1 for one class (default: water)."""
        c = self.water_class if c is None else c
        tp = float(self.cm[c, c])
        fp = float(self.cm[:, c].sum() - tp)
        fn = float(self.cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return prec, rec, f1

    @property
    def accuracy(self) -> float:
        n = self.cm.sum()
        return float(np.diag(self.cm).sum() / n) if n else 0.0

    def summary(self) -> dict:
        iou = self.iou
        prec, rec, f1 = self.prf()
        out = {
            "miou": round(self.miou, 4),
            "iou_bg": None if iou[0] is None else round(iou[0], 4),
            "iou_water": None if iou[self.water_class] is None
            else round(iou[self.water_class], 4),
            "precision_water": round(prec, 4),
            "recall_water": round(rec, 4),
            "f1_water": round(f1, 4),
            "accuracy": round(self.accuracy, 4),
            "n_pixels": int(self.cm.sum()),
        }
        if self.boundary:
            out["boundary_f1"] = round(float(np.mean(self.boundary)), 4)
        return out
