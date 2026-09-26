"""Build the artifact the camera nodes actually run.

`export_onnx` produces an fp32 graph, which is the benchmark baseline, not the
deployment target. On a Pi 4B `SU-WaterCam/docs/SEGFORMER_OPTIMIZATION.md` puts
B0 at ~10-18 s per frame in fp32 and ~4-7 s in INT8, so shipping fp32 costs
roughly 2.5x. This module does the quantization here, next to the training
data, instead of requiring a second pass with a different script on the Pi.

Output, named as `SU-WaterCam/tools/segformer_daemon.py` expects to find it:

    segformer_5band_fp32.onnx    benchmark baseline
    segformer_5band_int8.onnx    what the node runs
    segformer_5band_prep.onnx    quantization intermediate (kept for debugging)
    deploy.json                  what these are and how they were made

Static quantization needs representative input, and "representative" means the
exact preprocessing the graph will see in the field. Since normalisation is
compiled into the graph (see `models/segformer.py`), that is raw resized bands
in 0-255 units -- calibrating on normalised [0,1] data instead would pick
quantization ranges for a distribution that never arrives.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import modalities as M

FP32_NAME = "segformer_5band_fp32.onnx"
INT8_NAME = "segformer_5band_int8.onnx"
PREP_NAME = "segformer_5band_prep.onnx"

#: Pi 4B per-frame estimates at 512x512 from SU-WaterCam/docs/SEGFORMER_OPTIMIZATION.md
PI4B_SECONDS = {
    "segformer-b0": (10, 18, 4, 7),
    "segformer-b1": (20, 30, 8, 12),
    "segformer-b2": (40, 60, 15, 25),
    "segformer-b5": (90, 120, 30, 40),
}


def raw_input(tiff_path: Path, scene_dir: Path, modality: str, size: int) -> np.ndarray:
    """(1,C,size,size) float32 in raw 0-255 units — the graph's input contract."""
    import cv2
    m = M.get(modality)
    a = M.read_modality(m, Path(tiff_path), Path(scene_dir))
    r = np.stack([cv2.resize(a[i], (size, size), interpolation=cv2.INTER_AREA)
                  for i in range(a.shape[0])])
    return r.astype(np.float32)[None]


class _CalibrationReader:
    """Streams one calibration sample at a time; never holds the set in RAM."""

    def __init__(self, rows: list[dict], modality: str, input_name: str,
                 size: int, emit=None):
        self.rows = rows
        self.modality = modality
        self.input_name = input_name
        self.size = size
        self.i = 0
        self._emit = emit or (lambda **kw: None)

    def get_next(self):
        while self.i < len(self.rows):
            r = self.rows[self.i]
            self.i += 1
            try:
                return {self.input_name: raw_input(
                    r["tiff_path"], r.get("scene_dir") or Path(r["tiff_path"]).parent,
                    self.modality, self.size)}
            except Exception as e:                    # noqa: BLE001 - skip a bad scene
                self._emit(event="warn", msg=f"calibration skipped {r.get('stem')}: {e}")
        return None

    def rewind(self):
        self.i = 0


def build(hf_dir: Path, out_dir: Path, stats, calib_rows: list[dict],
          arch: str = "segformer-b0", size: int = 512, int8: bool = True,
          emit=None, dynamic_hw: bool = True) -> dict:
    """Export fp32 (+INT8) for the camera nodes and describe what was made."""
    import onnxruntime as ort

    from . import models

    emit = emit or (lambda **kw: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / FP32_NAME
    int8_path = out_dir / INT8_NAME

    m = models.load(arch, str(hf_dir), in_channels=stats.channels)
    t0 = time.time()
    m.export_onnx(fp32, size=size, stats=stats, embed_norm=True,
                  dynamic_hw=dynamic_hw)
    emit(event="info", msg=f"fp32 graph -> {fp32.name} "
                           f"({fp32.stat().st_size / 1e6:.1f} MB, {time.time()-t0:.1f}s)")

    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "arch": arch,
        "modality": stats.modality,
        "bands": stats.band_names,
        "in_channels": stats.channels,
        "input": {"layout": "NCHW",
                  "size": "dynamic" if dynamic_hw else [size, size],
                  "trained_size": [size, size], "units": "raw_0_255",
                  "normalization": f"embedded:{stats.method}"},
        "files": {"fp32": fp32.name},
        "sizes_mb": {"fp32": round(fp32.stat().st_size / 1e6, 2)},
        "deploy_to": "/home/pi/segformer_5band/",
        "runs_with": "SU-WaterCam/tools/segformer_daemon.py",
    }
    lo32, hi32, lo8, hi8 = PI4B_SECONDS.get(arch, (None,) * 4)
    if lo32:
        info["pi4b_estimate_s"] = {"fp32": [lo32, hi32], "int8": [lo8, hi8]}

    if not int8:
        info["int8"] = "skipped"
        (out_dir / "deploy.json").write_text(json.dumps(info, indent=2))
        return info

    try:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.preprocess import quant_pre_process
    except Exception as e:                            # noqa: BLE001
        emit(event="warn", msg=f"INT8 skipped — onnxruntime.quantization unavailable: {e}")
        info["int8"] = f"unavailable: {e}"
        (out_dir / "deploy.json").write_text(json.dumps(info, indent=2))
        return info

    prep = out_dir / PREP_NAME
    quant_pre_process(str(fp32), str(prep), skip_optimization=False)
    sess = ort.InferenceSession(str(prep), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    if not calib_rows:
        emit(event="warn", msg="no calibration scenes — INT8 ranges will be guessed "
                               "from random data and accuracy will suffer")
    reader = _CalibrationReader(calib_rows, stats.modality, input_name, size, emit)

    t0 = time.time()
    # `optimize_model` was removed from quantize_static (gone by onnxruntime
    # 1.29); quant_pre_process above already does that pass. Pass it only where
    # it still exists, so this works across the versions in play here and on
    # the Pi.
    import inspect
    kw = {}
    if "optimize_model" in inspect.signature(quantize_static).parameters:
        kw["optimize_model"] = True
    quantize_static(model_input=str(prep), model_output=str(int8_path),
                    calibration_data_reader=reader, quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, **kw)
    emit(event="info", msg=f"INT8 graph -> {int8_path.name} "
                           f"({int8_path.stat().st_size / 1e6:.1f} MB, "
                           f"{time.time()-t0:.0f}s, {len(calib_rows)} calibration scenes)")

    info["files"]["int8"] = int8_path.name
    info["files"]["prep"] = prep.name
    info["sizes_mb"]["int8"] = round(int8_path.stat().st_size / 1e6, 2)
    info["calibration_scenes"] = len(calib_rows)
    info.update(verify(fp32, int8_path, calib_rows, stats.modality, size, emit))
    if dynamic_hw:
        info["accepts_node_shape"] = {
            k: _accepts_node_shape(v, stats.channels, emit)
            for k, v in (("fp32", fp32), ("int8", int8_path))}
    (out_dir / "deploy.json").write_text(json.dumps(info, indent=2))
    return info


#: What `SU-WaterCam/tools/segformer_daemon.py` actually feeds a dynamic graph:
#: a 972x1296 capture rescaled keeping its aspect ratio to img_scale (1024,512),
#: giving 512x683, then padded up to the next multiple of 32.
NODE_HW = (512, 704)


def _accepts_node_shape(onnx_path: Path, channels: int, emit=None) -> bool:
    """Does this graph run at the non-square shape the node feeds it?

    A graph frozen at size x size loads and infers happily — on square input.
    The failure only appears on the node, where the frame is 4:3, so it is
    worth one forward pass here rather than a support call from the field.
    """
    import onnxruntime as ort

    emit = emit or (lambda **kw: None)
    h, w = NODE_HW
    try:
        s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        y = s.run(None, {s.get_inputs()[0].name:
                         np.zeros((1, channels, h, w), np.float32)})[0]
    except Exception as e:                            # noqa: BLE001
        emit(event="warn", msg=f"{onnx_path.name} rejects the node's {h}x{w} input "
                               f"— the daemon will squash the 4:3 frame: {e}")
        return False
    if y.shape[-2:] != (h, w):
        emit(event="warn", msg=f"{onnx_path.name} returned {y.shape[-2:]} for a "
                               f"{h}x{w} input")
        return False
    emit(event="info", msg=f"{onnx_path.name} accepts the node's {h}x{w} input")
    return True


def verify(fp32: Path, int8_path: Path, rows: list[dict], modality: str,
           size: int, emit=None) -> dict:
    """Does the quantized graph still agree with the one it came from?

    Quantization is lossy and silent: an INT8 model always loads and always
    returns a mask. Comparing the two on real scenes is the only way to see
    whether the loss matters, so the number lands in deploy.json rather than
    being assumed acceptable.
    """
    import onnxruntime as ort

    emit = emit or (lambda **kw: None)
    if not rows:
        return {}
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4                       # what the Pi daemon uses
    a = ort.InferenceSession(str(fp32), so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(int8_path), so, providers=["CPUExecutionProvider"])
    na, nb = a.get_inputs()[0].name, b.get_inputs()[0].name

    agree, ious, t_fp32, t_int8 = [], [], [], []
    for r in rows[:8]:
        try:
            x = raw_input(r["tiff_path"], r.get("scene_dir") or Path(r["tiff_path"]).parent,
                          modality, size)
        except Exception:                             # noqa: BLE001
            continue
        t = time.perf_counter(); pa = a.run(None, {na: x})[0][0].argmax(0); t_fp32.append(time.perf_counter()-t)
        t = time.perf_counter(); pb = b.run(None, {nb: x})[0][0].argmax(0); t_int8.append(time.perf_counter()-t)
        agree.append(float((pa == pb).mean()))
        u = np.logical_or(pa == 1, pb == 1).sum()
        ious.append(float(np.logical_and(pa == 1, pb == 1).sum() / u) if u else 1.0)
    if not agree:
        return {}
    out = {"int8_vs_fp32": {
        "scenes": len(agree),
        "pixel_agreement": round(float(np.mean(agree)), 4),
        "water_iou": round(float(np.mean(ious)), 4),
        "host_ms_fp32": round(float(np.mean(t_fp32)) * 1000),
        "host_ms_int8": round(float(np.mean(t_int8)) * 1000)}}
    emit(event="info", msg=f"INT8 vs fp32 on {len(agree)} scene(s): "
                           f"{out['int8_vs_fp32']['pixel_agreement']*100:.2f}% pixels, "
                           f"water IoU {out['int8_vs_fp32']['water_iou']:.4f}")
    return out
