"""Model metadata, schema 1: what an exported ONNX model says about itself.

The same text lives in `docs/MODEL_METADATA.md` here and in photo_processing;
keep the two identical. The node daemon (SU-WaterCam
`tools/segformer_preprocess.py`) reads these keys and refuses a model whose
channels or normalization it cannot honour.

Stdlib only, except `stamp_onnx`/`read_onnx` which import `onnx` when called.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date

SCHEMA = 1
BAND_NAMES = ("red", "green", "blue", "thermal", "nir")
METHODS = ("minmax", "meanstd", "percentile")
INTERPOLATIONS = ("area", "bilinear")
SIDECAR_NAME = "model.json"

#: keys whose value is a comma-separated list / a JSON list / an int
_CSV_KEYS = ("bands", "classes")
_JSON_KEYS = ("norm_mean", "norm_std", "norm_p_lo", "norm_p_hi")
_INT_KEYS = ("schema", "in_channels", "num_classes")


def _hw(v) -> str:
    return "dynamic" if v == "dynamic" else f"{int(v[0])},{int(v[1])}"


def build_metadata(*, model_name: str, model_version: str | None = None,
                   bands, classes, normalization: str, norm_stats: dict | None = None,
                   input_hw="dynamic", trained_size=(512, 512),
                   resize_interpolation: str = "area", arch: str | None = None,
                   input_range: str | None = None) -> dict:
    """The `metadata_props` for one model, as {str: str}.

    `normalization` is a method ("minmax", "meanstd", "percentile": the node
    applies it) or "embedded:<method>" (the graph applies it; feed raw 0-255).
    `norm_stats` holds norm_mean/norm_std (meanstd) or norm_p_lo/norm_p_hi
    (percentile), as lists of one number per band.
    """
    bands, classes = list(bands), list(classes)
    props = {
        "schema": SCHEMA,
        "model_name": model_name,
        "model_version": model_version or date.today().isoformat(),
        "bands": ",".join(bands),
        "classes": ",".join(classes),
        "in_channels": len(bands),
        "num_classes": len(classes),
        "normalization": normalization,
        "input_layout": "NCHW",
        "input_hw": _hw(input_hw),
        "trained_size": _hw(trained_size),
        "input_range": input_range or ("raw_0_255" if normalization.startswith("embedded:")
                                       else "normalised"),
        "resize_interpolation": resize_interpolation,
    }
    if arch:
        props["arch"] = arch
    for k, v in (norm_stats or {}).items():
        props[k] = json.dumps([float(x) for x in v])
    props = {k: str(v) for k, v in props.items()}
    problems = validate(props)
    if problems:
        raise ValueError("inconsistent model metadata: " + "; ".join(problems))
    return props


def validate(props: dict) -> list:
    """Producer-side consistency check; returns a list of problems (empty = ok)."""
    out = []
    bands = [b for b in props.get("bands", "").split(",") if b]
    classes = [c for c in props.get("classes", "").split(",") if c]
    norm = props.get("normalization", "")
    method = norm.split(":", 1)[-1]
    if props.get("schema") != str(SCHEMA):
        out.append(f"schema must be {SCHEMA}")
    if not bands:
        out.append("no bands")
    if props.get("in_channels") != str(len(bands)):
        out.append("in_channels does not match the number of bands")
    if props.get("num_classes") != str(len(classes)):
        out.append("num_classes does not match the number of classes")
    if method not in METHODS:
        out.append(f"unknown normalization {norm!r}")
    if props.get("resize_interpolation") not in INTERPOLATIONS:
        out.append("resize_interpolation must be one of " + "/".join(INTERPOLATIONS))
    keys = {"meanstd": ("norm_mean", "norm_std"),
            "percentile": ("norm_p_lo", "norm_p_hi")}.get(method, ())
    for k in keys:
        try:
            if len(json.loads(props[k])) != len(bands):
                out.append(f"{k} needs one value per band")
        except (KeyError, ValueError, TypeError):
            out.append(f"{k} missing or not a JSON list")
    return out


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar(props: dict, onnx_path) -> dict:
    """The `model.json` content: the same keys, typed, plus the file's sha256."""
    out = {}
    for k, v in props.items():
        if k in _CSV_KEYS:
            out[k] = [x for x in v.split(",") if x]
        elif k in _JSON_KEYS:
            out[k] = json.loads(v)
        elif k in _INT_KEYS:
            out[k] = int(v)
        else:
            out[k] = v
    out["file"] = os.path.basename(str(onnx_path))
    out["sha256"] = sha256_file(onnx_path)
    return out


def write_sidecar(onnx_path, props: dict, path=None) -> str:
    """Write `model.json` next to the model (call after the .onnx is final)."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(str(onnx_path))), SIDECAR_NAME)
    with open(path, "w") as f:
        json.dump(sidecar(props, onnx_path), f, indent=2)
        f.write("\n")
    return str(path)


def stamp_onnx(onnx_path, props: dict) -> None:
    """Replace the model's `metadata_props` with `props` (needs `onnx`)."""
    import onnx

    m = onnx.load(str(onnx_path))
    del m.metadata_props[:]
    for k, v in props.items():
        e = m.metadata_props.add()
        e.key, e.value = k, str(v)
    onnx.save(m, str(onnx_path))


def read_onnx(onnx_path) -> dict:
    """The model's `metadata_props` as a dict (needs `onnx`)."""
    import onnx

    return {p.key: p.value for p in onnx.load(str(onnx_path)).metadata_props}
