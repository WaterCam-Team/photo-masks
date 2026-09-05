"""Auto-segmentation backend adapters for the UFONet water annotator.

Every backend maps a co-registered 5-band scene to a binary water mask
(uint8 HxW, 0 = background, 255 = water) at the TIFF's native resolution,
behind one interface so the annotator UI can switch between them freely.

Backends degrade gracefully: if a model / weights / env is missing,
``available()`` returns ``(False, reason)`` and the UI disables the option
instead of crashing.

Band order in ``color_preserved_5_band.tiff``:
    1 = Red   2 = Green   3 = Blue   4 = Thermal (LWIR)   5 = NIR difference
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ANN_DIR = Path(__file__).resolve().parent            # .../photo_processing/annotator
REPO_ROOT = ANN_DIR.parent                           # .../photo_processing

B_R, B_G, B_B, B_THERMAL, B_NIR = 0, 1, 2, 3, 4      # 0-based into read() stack

_DEVICE_CACHE: dict[str, str] = {}


_DEVICE_ALIASES = {"amd": "cuda", "rocm": "cuda", "hip": "cuda", "gpu": "cuda",
                   "nvidia": "cuda", "intel": "xpu"}


def resolve_device(spec: str | None = "auto") -> str:
    """Map a device spec -> a concrete torch device string.

    Accepts 'auto' | 'cpu' | 'cuda[:N]' | 'mps' | 'xpu', plus the aliases
    amd/rocm/hip/nvidia/gpu -> 'cuda' (AMD ROCm PyTorch also uses the 'cuda'
    device string) and intel -> 'xpu'. 'auto' -> cuda, else Intel xpu, else
    Apple mps, else cpu. Result cached.
    """
    spec = _DEVICE_ALIASES.get((spec or "auto").lower(), (spec or "auto").lower())
    if spec != "auto":
        return spec
    if "auto" in _DEVICE_CACHE:
        return _DEVICE_CACHE["auto"]
    dev = "cpu"
    try:
        import torch
        if torch.cuda.is_available():                # NVIDIA CUDA or AMD ROCm
            dev = "cuda"
        elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            dev = "xpu"                              # Intel GPU (IPEX)
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            dev = "mps"
    except Exception:                                # noqa: BLE001
        pass
    _DEVICE_CACHE["auto"] = dev
    return dev


def device_label(spec: str | None = "auto") -> str:
    dev = resolve_device(spec)
    try:
        import torch
        if dev.startswith("cuda"):
            name = torch.cuda.get_device_name(0)
            hip = getattr(torch.version, "hip", None)
            return f"{dev} ({name}{' · ROCm ' + hip if hip else ''})"
        if dev == "xpu":
            return f"xpu ({torch.xpu.get_device_name(0)})"
    except Exception:                                # noqa: BLE001
        pass
    return dev


def accel_hint() -> str | None:
    """One-line hint when a GPU is physically present but torch can't use it.

    Returns None when a device is usable, or when there's nothing to say.
    """
    if resolve_device("auto") != "cpu":
        return None
    try:
        import torch
        cuda_build = bool(getattr(torch.version, "cuda", None))
        hip_build = bool(getattr(torch.version, "hip", None))
    except Exception:                                # noqa: BLE001
        return None
    amd = Path("/dev/kfd").exists()                  # AMD KFD (ROCm) device node
    nvidia = Path("/dev/nvidiactl").exists() or Path("/dev/nvidia0").exists()
    if amd and cuda_build:
        return ("AMD GPU present (/dev/kfd) but torch is a CUDA build — install a ROCm "
                "torch to use it (README: 'Integrated AMD Radeon'). Running on CPU.")
    if amd and not hip_build:
        return ("AMD GPU present but this torch has no ROCm support — install a ROCm "
                "torch build. Running on CPU.")
    if nvidia and cuda_build:
        return ("NVIDIA GPU present but torch.cuda is unavailable — check the driver / "
                "CUDA runtime. Running on CPU.")
    return None


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


def _read_bands(tiff_path: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(tiff_path) as src:
        arr = src.read().astype(np.float32)          # (bands, H, W)
    if arr.shape[0] < 5:
        raise ValueError(f"expected >=5 bands, got {arr.shape[0]}: {tiff_path}")
    return arr


def _minmax_per_image(bands: np.ndarray) -> np.ndarray:
    """Legacy preprocessing: per-band min-max to [0,1], measured per image.

    Kept only for checkpoints trained before band statistics existed. It
    erases absolute radiometry (water's low NIR reflectance is exactly that
    signature), which is why trained models now carry a norm.json instead —
    see training/stats.py.
    """
    out = np.asarray(bands, np.float32).copy()
    for i in range(out.shape[0]):
        lo, hi = float(out[i].min()), float(out[i].max())
        out[i] = (out[i] - lo) / (hi - lo) if hi > lo else 0.0
    return np.clip(out, 0.0, 1.0)


def _tiff_size(tiff_path: Path) -> tuple[int, int]:
    import rasterio
    with rasterio.open(tiff_path) as src:
        return src.height, src.width


SAM_DEFAULT_WEIGHTS = {"fastsam": "FastSAM-s.pt", "sam": "sam_b.pt", "sam2": "sam2.1_l.pt"}


def resolve_sam_weights(weights_cfg: str | None, allow_download: bool = False,
                        model_type: str = "fastsam") -> tuple[str | None, str]:
    """Locate a SAM/SAM2/FastSAM checkpoint. Returns (path_or_name, model_type).

    A relative path is tried as given, then against the cwd, then against the
    repo root. The checkpoint's filename also settles `model_type`, so pointing
    `sam.weights` at a sam2 file does the right thing instead of loading it
    with the wrong class.
    """
    if weights_cfg:
        cand = Path(os.path.expanduser(str(weights_cfg)))
        tries = [cand, Path.cwd() / cand, REPO_ROOT / cand] if not cand.is_absolute() else [cand]
        wp = next((t.resolve() for t in tries if t.exists()), None)
        if wp is not None:
            n = wp.name.lower()
            if "sam2" in n:
                model_type = "sam2"
            elif "fastsam" in n:
                model_type = "fastsam"
            elif n.startswith("sam"):
                model_type = "sam"
            return str(wp), model_type
    if allow_download:
        return (weights_cfg or SAM_DEFAULT_WEIGHTS.get(model_type, "FastSAM-s.pt")), model_type
    return None, model_type


def sam_prompt_image(arr: np.ndarray, on: str = "rgb") -> np.ndarray:
    """The 3-channel image handed to SAM, in **BGR** order.

    ultralytics documents numpy inputs as OpenCV-order BGR and its preprocess
    does `im[..., ::-1]` to recover RGB, so handing it an RGB array silently
    swaps red and blue before the model ever sees it.

    on="rgb"   -> true colour
    on="false" -> colour-IR (NIR->R, R->G, G->B), where water is very dark
    """
    if on == "false":
        def st(x):
            lo, hi = np.percentile(x, [2, 98])
            return np.clip((x - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
        return np.dstack([st(arr[B_G]), st(arr[B_R]), st(arr[B_NIR])])      # B, G, R
    return np.dstack([arr[B_B], arr[B_G], arr[B_R]]).clip(0, 255).astype(np.uint8)


def _clean(mask_bool: np.ndarray, open_iter: int = 2, close_iter: int = 1,
           fill: bool = True) -> np.ndarray:
    """Morphological tidy-up shared by the spectral backends."""
    import cv2
    from scipy.ndimage import binary_fill_holes

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = mask_bool.astype(np.uint8)
    if open_iter:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=open_iter)
    if fill:
        m = binary_fill_holes(m).astype(np.uint8)
    if close_iter:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iter)
    return m > 0


@dataclass
class BackendResult:
    mask: np.ndarray | None                          # uint8 HxW {0, 255}
    elapsed_s: float = 0.0
    meta: dict = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------

class Backend:
    name = "base"
    label = "Base"
    params_schema: list[dict] = []

    def available(self) -> tuple[bool, str]:
        return True, ""

    def run(self, scene_dir: Path, tiff_path: Path, params: dict) -> BackendResult:
        raise NotImplementedError

    def info(self) -> dict:
        ok, reason = self.available()
        return {
            "name": self.name,
            "label": self.label,
            "params": self.params_schema,
            "available": ok,
            "reason": reason,
        }


# ---------------------------------------------------------------------------
# 1. NIR threshold  (in-process, no extra deps beyond rasterio/scipy/cv2)
#    the classic single-cue baseline: water absorbs NIR, so it reads dark
# ---------------------------------------------------------------------------

class NirThresholdBackend(Backend):
    name = "nir"
    label = "NIR threshold"
    params_schema = [
        {"name": "threshold", "type": "float", "default": 0.25,
         "min": 0.02, "max": 0.90, "step": 0.01,
         "help": "Water = NIR below this (per-image normalised). Lower = less water."},
        {"name": "open_iter", "type": "int", "default": 2, "min": 0, "max": 6, "step": 1,
         "help": "Speckle-removal passes."},
    ]

    def run(self, scene_dir, tiff_path, params):
        t0 = time.time()
        try:
            arr = _read_bands(tiff_path)
        except Exception as e:                        # noqa: BLE001
            return BackendResult(None, error=str(e))

        nir = arr[B_NIR]
        lo, hi = float(nir.min()), float(nir.max())
        if hi - lo < 1e-6:
            return BackendResult(None, error="NIR band has no dynamic range")

        nn = (nir - lo) / (hi - lo)
        thr = float(params.get("threshold", 0.25))
        water = _clean(nn < thr, open_iter=int(params.get("open_iter", 2)))
        mask = water.astype(np.uint8) * 255
        return BackendResult(mask, time.time() - t0,
                             {"threshold": thr,
                              "water_pct": round(100 * float(water.mean()), 2)})


# ---------------------------------------------------------------------------
# 2. Thermal + NIR  (Otsu, in-process)  — the night-time / low-light fallback
# ---------------------------------------------------------------------------

class ThermalOtsuBackend(Backend):
    name = "thermal"
    label = "Thermal + NIR (Otsu)"
    params_schema = []

    def run(self, scene_dir, tiff_path, params):
        import cv2
        t0 = time.time()
        try:
            arr = _read_bands(tiff_path)
        except Exception as e:                        # noqa: BLE001
            return BackendResult(None, error=str(e))

        thermal, nir = arr[B_THERMAL], arr[B_NIR]

        def otsu(b):
            b8 = np.clip(b, 0, 255).astype(np.uint8)
            t, _ = cv2.threshold(b8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return float(t)

        t_hot, t_veg = otsu(thermal), otsu(nir)
        water = _clean((thermal < t_hot) & (nir < t_veg), open_iter=2, close_iter=2)
        mask = water.astype(np.uint8) * 255
        return BackendResult(mask, time.time() - t0,
                             {"thermal_cut": t_hot, "nir_cut": t_veg,
                              "water_pct": round(100 * float(water.mean()), 2)})


# ---------------------------------------------------------------------------
# 3. TinySAM  (subprocess into an env that has `tinysam` + torch)
#    Entirely external: it drives a checkout of the TinySAM repo plus a driver
#    script, neither of which ships here. Both paths must be configured
#    (backends.tinysam.{tinysam_path,script}, or the TINYSAM_PATH env var);
#    unconfigured, the backend reports itself unavailable and is greyed out.
# ---------------------------------------------------------------------------

class TinySamBackend(Backend):
    name = "tinysam"
    label = "TinySAM (auto points)"
    params_schema = [
        {"name": "n_points", "type": "int", "default": 5, "min": 1, "max": 20, "step": 1,
         "help": "Number of auto-selected water prompt points."},
    ]

    def __init__(self, python_exe: str | None = None, tinysam_path: str | None = None,
                 script: str | None = None, weights: str | None = None):
        self.python_exe = python_exe or sys.executable
        env_path = os.environ.get("TINYSAM_PATH")
        raw_path = tinysam_path or env_path
        self.tinysam_path = _expand(raw_path) if raw_path else None
        self.script = _expand(script) if script else None
        if weights:
            self.weights = _expand(weights)
        elif self.tinysam_path:
            self.weights = self.tinysam_path / "weights" / "tinysam.pth"
        else:
            self.weights = None
        # `import tinysam` (pulls in torch) can take 10-30 s, so probe it in a
        # background thread instead of blocking startup / the first /api/config.
        self._probe: tuple[bool, str] | None = None
        self._probe_started = False

    def _env(self) -> dict:
        """os.environ + the TinySAM checkout on PYTHONPATH (when configured)."""
        env = dict(os.environ)
        parts = [p for p in (str(self.tinysam_path) if self.tinysam_path else "",
                             env.get("PYTHONPATH", "")) if p]
        if parts:
            env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def _start_probe(self):
        if self._probe_started:
            return
        self._probe_started = True

        def work():
            try:
                env = self._env()
                pr = subprocess.run([self.python_exe, "-c", "import tinysam"],
                                    capture_output=True, text=True, timeout=60, env=env)
                if pr.returncode == 0:
                    self._probe = (True, "")
                else:
                    last = (pr.stderr.strip().splitlines() or ["import failed"])[-1]
                    self._probe = (False, last)
            except Exception as e:                    # noqa: BLE001
                self._probe = (False, f"probe failed: {e}")

        threading.Thread(target=work, name="tinysam-probe", daemon=True).start()

    def available(self):
        if self.script is None:
            return False, ("not configured — set backends.tinysam.script to the "
                           "TinySAM driver script (this repo does not ship one)")
        if self.weights is None:
            return False, ("not configured — set backends.tinysam.tinysam_path "
                           "(or TINYSAM_PATH), or backends.tinysam.weights")
        if not self.script.exists():
            return False, f"script not found: {self.script}"
        if not self.weights.exists():
            return False, f"weights not found: {self.weights}"
        self._start_probe()
        if self._probe is None:
            return False, "checking `import tinysam` in a background thread — reload in a few seconds"
        ok, why = self._probe
        if not ok:
            return False, (f"`import tinysam` fails in {self.python_exe}: {why} "
                           f"— set backends.tinysam.python in the config")
        return True, ""

    def run(self, scene_dir, tiff_path, params):
        import cv2
        t0 = time.time()
        n = int(params.get("n_points", 5))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.png"
            cmd = [self.python_exe, str(self.script), str(scene_dir),
                   "--multi-point", str(n), "--output", str(out)]
            try:
                proc = subprocess.run(cmd, cwd=str(self.script.parent), env=self._env(),
                                      capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                return BackendResult(None, error="TinySAM timed out (300 s)")
            if not out.exists():
                tail = (proc.stderr or proc.stdout or "")[-800:]
                return BackendResult(None, error=f"TinySAM produced no mask.\n{tail}")
            m = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)

        if m is None:
            return BackendResult(None, error="could not read TinySAM output PNG")
        H, W = _tiff_size(tiff_path)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (m > 127).astype(np.uint8) * 255
        return BackendResult(mask, time.time() - t0,
                             {"n_points": n,
                              "water_pct": round(100 * float((mask > 0).mean()), 2)})


# ---------------------------------------------------------------------------
# 4. SegFormer  in-process, from a .onnx (onnxruntime) or an HF checkpoint dir.
#    Preprocessing is dictated by the model, not by this file: a checkpoint
#    trained by the `training/` package ships a `norm.json` naming its modality
#    (which bands it consumes) and the band statistics it was normalised with.
#    This backend reads that file and applies exactly those, which is what
#    keeps training and inference from drifting apart. A model with no
#    norm.json is treated as legacy 5-band per-image min-max, the behaviour
#    every checkpoint had before the statistics were introduced.
# ---------------------------------------------------------------------------


class SegformerOnnxBackend(Backend):
    name = "segformer"
    label = "SegFormer (5-band)"
    params_schema = []

    def __init__(self, onnx_path: str | None = None,
                 size: tuple[int, int] | None = None, water_class: int = 1,
                 device: str | None = "auto"):
        self.model_path = Path(onnx_path) if onnx_path else self._autofind()
        self.size = size                             # (H, W) or None -> from model
        self.water_class = water_class
        self.device = resolve_device(device)
        self._sess = None                            # onnxruntime session
        self._torch = None                           # HF model (moved to self.device)
        self._pre = None                             # resolved preprocessing, cached
        self._lock = threading.Lock()                # serialise inference across request threads
        self._is_hf = bool(self.model_path
                           and Path(self.model_path).is_dir()
                           and (Path(self.model_path) / "config.json").exists())

    @staticmethod
    def _autofind() -> Path | None:
        """Newest local fine-tune, else any .onnx under annotator/weights/.

        Deliberately searches only inside this repo — an auto-search that
        wandered into the user's home directory would silently pick up a model
        nobody asked for. Anything elsewhere goes in backends.segformer.onnx.
        """
        runs = sorted((ANN_DIR / "work" / "runs").glob("*/best_hf"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            return runs[0]
        weights = ANN_DIR / "weights"
        cands = sorted(weights.rglob("*.onnx")) if weights.exists() else []
        cands.sort(key=lambda p: ("5band" not in p.name.lower(),
                                  "int8" not in p.name.lower()))
        return cands[0] if cands else None

    def available(self):
        if self.model_path is None or not Path(self.model_path).exists():
            return False, ("no SegFormer model — fine-tune one in the Train panel, "
                           "or set backends.segformer.onnx to a .onnx / HF checkpoint dir")
        if self._is_hf:
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
            except Exception:                         # noqa: BLE001
                return False, ("transformers/torch are not installed — run `uv sync` "
                               "(or `uv sync --group train`)")
            return True, ""
        try:
            import onnxruntime  # noqa: F401
        except Exception:                             # noqa: BLE001
            return False, ("onnxruntime is not installed — run `uv sync` "
                           "(or `uv sync --group onnx`)")
        return True, ""

    def _size_hw(self, native_hw: tuple[int, int] | None = None,
                 pre: dict | None = None) -> tuple[int, int]:
        """Input size, matched to how the model was trained.

        A checkpoint from the `training/` package was validated on whole frames
        at native resolution (its training crops came from scale-jittered but
        never aspect-squashed frames), so it is served the same way: native,
        padded up to a multiple of 32. Squashing a 4:3 frame into 512x512 would
        stretch every shoreline by a third relative to what it learned.

        Legacy checkpoints stay on 512x512 because that is the geometry they
        were trained with, and a .onnx graph keeps whatever size it was
        exported at. `backends.segformer.size` overrides either way.
        """
        if self.size:
            return self.size
        if not self._is_hf:
            shp = self._session().get_inputs()[0].shape
            h = shp[2] if isinstance(shp[2], int) else 512
            w = shp[3] if isinstance(shp[3], int) else 512
            return h, w
        if native_hw and pre is not None and not pre["legacy"]:
            h, w = native_hw
            return h + (-h) % 32, w + (-w) % 32
        return 512, 512

    # ONNX Runtime GPU execution providers, in preference order. Picked up
    # automatically from whatever ORT build is installed:
    #   onnxruntime-gpu       -> CUDAExecutionProvider (also AMD via ROCm builds)
    #   onnxruntime-rocm      -> ROCMExecutionProvider / MIGraphXExecutionProvider (AMD)
    #   onnxruntime-directml  -> DmlExecutionProvider  (AMD / Intel / NVIDIA on Windows)
    #   onnxruntime-openvino  -> OpenVINOExecutionProvider (Intel)
    _GPU_EPS = ("CUDAExecutionProvider", "ROCMExecutionProvider",
                "MIGraphXExecutionProvider", "DmlExecutionProvider",
                "OpenVINOExecutionProvider", "CoreMLExecutionProvider")

    def _session(self):
        if self._sess is None:
            import onnxruntime as ort
            avail = set(ort.get_available_providers())
            providers = []
            if self.device != "cpu":
                providers = [ep for ep in self._GPU_EPS if ep in avail]
            providers.append("CPUExecutionProvider")
            self._sess = ort.InferenceSession(str(self.model_path), providers=providers)
        return self._sess

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """x: (1,5,H,W) float32 -> logits (C,H,W) float32."""
        if self._is_hf:
            import torch
            import torch.nn.functional as F
            from transformers import SegformerForSemanticSegmentation
            if self._torch is None:
                m = SegformerForSemanticSegmentation.from_pretrained(str(self.model_path))
                m.eval()
                try:
                    m = m.to(self.device)
                except Exception:                    # noqa: BLE001 - fall back to cpu
                    self.device = "cpu"
                self._torch = m
            with torch.no_grad():
                xt = torch.from_numpy(x).to(self.device)
                lo = self._torch(pixel_values=xt).logits
                lo = F.interpolate(lo, size=x.shape[-2:], mode="bilinear",
                                   align_corners=False)
            return lo[0].cpu().numpy()
        sess = self._session()
        out = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0])
        return out[0] if out.ndim == 4 else out

    def _norm_json(self) -> Path | None:
        """Where a checkpoint's band statistics live, if it has any."""
        p = Path(self.model_path)
        cands = [p / "norm.json"] if p.is_dir() else \
            [p.with_suffix(""), p.parent / (p.stem + ".norm.json")]
        for c in cands:
            c = c if c.name.endswith("norm.json") else c.parent / (c.name + ".norm.json")
            if c.exists():
                return c
        return None

    def _preprocessing(self) -> dict:
        """Resolve {bands, norm, read} from the model's norm.json, once.

        Falls back to legacy 5-band per-image min-max when there is no
        norm.json, so checkpoints predating the statistics keep working
        unchanged. `meta["norm"]` reports which path was taken, because a
        silent switch between the two would be an accuracy bug nobody sees.
        """
        if self._pre is not None:
            return self._pre
        legacy = {"bands": (0, 1, 2, 3, 4), "read": None, "modality": "fiveband",
                  "label": "legacy per-image min-max", "norm": _minmax_per_image,
                  "legacy": True}
        nj = self._norm_json()
        if nj is None:
            self._pre = legacy
            return self._pre
        try:
            import sys
            if str(ANN_DIR.parent) not in sys.path:
                sys.path.insert(0, str(ANN_DIR.parent))
            from training import modalities as _M
            from training.stats import BandStats, Normalizer

            st = BandStats.from_json(nj)
            m = _M.get(st.modality)
            if m.channels != st.channels:
                raise ValueError(f"norm.json says {st.channels} channels but modality "
                                 f"{st.modality!r} has {m.channels}")
            norm = Normalizer(st)
            read = None
            if m.extra_glob:                         # e.g. rgb_nofilt's NIR-ON frame
                def read(scene_dir, tiff_path, _m=m):
                    return _M.read_modality(_m, tiff_path, scene_dir)
            self._pre = {"bands": tuple(m.bands), "read": read, "modality": st.modality,
                         "label": f"{st.modality}/{norm.method} from {nj.name}",
                         "norm": norm, "legacy": False}
        except Exception as e:                        # noqa: BLE001 - never break inference
            self._pre = dict(legacy, label=f"legacy min-max ({nj.name} unusable: {e})")
        return self._pre

    def run(self, scene_dir, tiff_path, params):
        with self._lock:                             # one forward() at a time
            return self._run(scene_dir, tiff_path, params)

    def _run(self, scene_dir, tiff_path, params):
        import cv2
        t0 = time.time()
        try:
            arr = _read_bands(tiff_path)
        except Exception as e:                        # noqa: BLE001
            return BackendResult(None, error=str(e))
        H0, W0 = arr.shape[1], arr.shape[2]
        pre = self._preprocessing()
        H, W = self._size_hw((H0, W0), pre)

        if pre["read"] is not None:                  # modality needs a sibling file
            bands = pre["read"](Path(scene_dir), Path(tiff_path)).astype(np.float32)
        else:
            bands = arr[list(pre["bands"])].copy()
        if bands.shape[1:] != (H, W):
            bands = np.stack(
                [cv2.resize(bands[i], (W, H), interpolation=cv2.INTER_AREA)
                 for i in range(bands.shape[0])], axis=0)
        x = pre["norm"](bands)[None].astype(np.float32)

        try:
            out = np.asarray(self._predict(x))
        except Exception as e:                        # noqa: BLE001
            return BackendResult(None, error=f"SegFormer inference failed: {e}")

        meta: dict = {"preprocessing": pre["label"], "modality": pre["modality"],
                      "input_hw": [int(H), int(W)]}
        if out.ndim == 3:                            # (C, h, w) logits
            ex = np.exp(out - out.max(axis=0, keepdims=True))
            p = ex / ex.sum(axis=0, keepdims=True)
            ent = -(p * np.log(p + 1e-9)).sum(axis=0)
            pred = p.argmax(axis=0)
            meta["mean_entropy"] = round(float(ent.mean()), 4)
            meta["low_conf_frac"] = round(float((p.max(axis=0) < 0.6).mean()), 4)
        else:                                        # (h, w) class idx
            pred = out

        water = (pred == self.water_class).astype(np.uint8)
        if water.shape != (H0, W0):
            water = cv2.resize(water, (W0, H0), interpolation=cv2.INTER_NEAREST)
        mask = water * 255
        meta["water_pct"] = round(100 * float(water.mean()), 2)
        meta["model"] = Path(self.model_path).name
        meta["kind"] = "hf" if self._is_hf else "onnx"
        return BackendResult(mask, time.time() - t0, meta)


# ---------------------------------------------------------------------------
# 5. Spectral fusion  (in-process; the recommended cheap prior)
#    NIR-dark gate  x  (NDWI + low-texture + thermal-smoothness) + snow guard
# ---------------------------------------------------------------------------

class SpectralBackend(Backend):
    name = "spectral"
    label = "Spectral fusion (NIR/NDWI/thermal/texture)"
    params_schema = [
        {"name": "refine", "type": "select", "default": "guided",
         "options": ["guided", "rw", "none"],
         "help": "Boundary snap: guided filter (fast), random-walker (slow), or none."},
        {"name": "thresh", "type": "float", "default": 0.5, "min": 0.2, "max": 0.9, "step": 0.02,
         "help": "Water-probability cutoff. Higher = less water, higher precision."},
        {"name": "snow_guard", "type": "bool", "default": True,
         "help": "Suppress bright achromatic regions (snow, blown-out sky, white siding)."},
        {"name": "horizon", "type": "bool", "default": True,
         "help": "Treat the detected sky region as hard non-water."},
        {"name": "specular", "type": "bool", "default": False,
         "help": "Add bright smooth NIR-dark patches next to water as reflected sky."},
    ]

    def run(self, scene_dir, tiff_path, params):
        import autolabel
        t0 = time.time()
        try:
            arr = _read_bands(tiff_path)
        except Exception as e:                        # noqa: BLE001
            return BackendResult(None, error=str(e))
        mask, meta, _prob = autolabel.spectral_water(arr, params or {})
        return BackendResult(mask, time.time() - t0, meta)


# ---------------------------------------------------------------------------
# 6. SAM (spectral-prompted)  via ultralytics FastSAM / SAM2.
#    Spectral prior -> positive/negative points + box -> promptable segmenter.
# ---------------------------------------------------------------------------

class SamPromptedBackend(Backend):
    name = "sam"
    label = "SAM (spectral-prompted)"
    params_schema = [
        {"name": "on", "type": "select", "default": "rgb", "options": ["rgb", "false"],
         "help": "Image given to SAM: true-colour RGB, or NIR/R/G false-colour."},
        {"name": "n_pos", "type": "int", "default": 8, "min": 1, "max": 24, "step": 1,
         "help": "Positive (water) prompt points sampled from the spectral prior."},
        {"name": "n_neg", "type": "int", "default": 8, "min": 0, "max": 24, "step": 1,
         "help": "Negative (non-water) prompt points."},
        {"name": "use_box", "type": "bool", "default": True,
         "help": "Also pass the prior's bounding box as a prompt."},
    ]
    _DEFAULT_NAME = SAM_DEFAULT_WEIGHTS

    def __init__(self, weights: str | None = None, model_type: str = "fastsam",
                 allow_download: bool = False, name: str | None = None,
                 label: str | None = None, device: str | None = "auto"):
        if name:
            self.name = name
        if label:
            self.label = label
        self.model_type = (model_type or "fastsam").lower()
        self.allow_download = bool(allow_download)
        self.device = resolve_device(device)
        self._weights_cfg = weights
        self._model = None
        self._resolved: str | None = None
        self._lock = threading.Lock()                # serialise inference across request threads
        self._resolve()

    def _resolve(self):
        self._resolved, self.model_type = resolve_sam_weights(
            self._weights_cfg, self.allow_download, self.model_type)

    def available(self):
        try:
            import ultralytics  # noqa: F401
        except Exception:                             # noqa: BLE001
            return False, ("ultralytics is not installed — run `uv sync` "
                           "(or `uv sync --group sam`, or `pip install ultralytics`)")
        if not self._resolved:
            return False, (f"no weights — set backends.{self.name}.weights to a FastSAM/SAM/SAM2 "
                           f".pt, or backends.{self.name}.allow_download=true to fetch one")
        return True, ""

    def _load(self):
        if self._model is None:
            from ultralytics import FastSAM, SAM
            cls = FastSAM if self.model_type == "fastsam" else SAM
            self._model = cls(self._resolved)
        return self._model

    def run(self, scene_dir, tiff_path, params):
        with self._lock:                             # one ultralytics predict at a time
            return self._run(scene_dir, tiff_path, params)

    def _run(self, scene_dir, tiff_path, params):
        import autolabel
        import cv2
        t0 = time.time()
        p = {"on": "rgb", "n_pos": 8, "n_neg": 8, "use_box": True}
        p.update(params or {})
        try:
            arr = _read_bands(tiff_path)
        except Exception as e:                        # noqa: BLE001
            return BackendResult(None, error=str(e))
        H, W = arr.shape[1], arr.shape[2]

        img = sam_prompt_image(arr, p["on"])

        _m, prior_meta, prob = autolabel.spectral_water(arr, {"refine": "none"})
        pr = autolabel.derive_prompts(arr, prob, int(p["n_pos"]), int(p["n_neg"]))

        # geometry fallback: fixed cameras frame water in the lower-centre, so
        # when the spectral prior is too weak/degenerate to prompt SAM well,
        # add a lower-centre box + points.
        pts, labels, box = pr["points"], pr["labels"], pr["box"]
        n_pos = int((labels == 1).sum()) if len(labels) else 0
        area = ((box[2] - box[0]) * (box[3] - box[1]) / (H * W)) if box else 0.0
        used_geom = n_pos < 3 or box is None or area < 0.04 or area > 0.92
        if used_geom:
            gpts = np.array([[W // 2, int(H * 0.75)], [int(W * 0.3), int(H * 0.85)],
                             [int(W * 0.7), int(H * 0.85)], [W // 2, int(H * 0.93)]])
            glab = np.ones(len(gpts), int)
            gbox = [int(W * 0.10), int(H * 0.42), int(W * 0.90), int(H * 0.98)]
            pts = np.vstack([pts, gpts]) if len(pts) else gpts
            labels = np.concatenate([labels, glab]) if len(labels) else glab
            box = gbox

        try:
            model = self._load()
        except Exception as e:                        # noqa: BLE001
            return BackendResult(None, error=f"failed to load SAM ({self._resolved}): {e}")

        # prompt-shape differs: FastSAM wants FLAT points/labels; SAM & SAM2
        # want ONE nested group  points=[[[x,y],...]]  labels=[[1,0,...]].
        pl, ll = pts.tolist(), labels.tolist()
        common = dict(retina_masks=True, verbose=False, device=self.device)
        if self.model_type == "fastsam":
            kw = dict(points=pl, labels=ll, **common)
        else:
            kw = dict(points=[pl], labels=[ll], **common)
        if p["use_box"] and box is not None:
            kw["bboxes"] = [box]
        try:
            res = model(img, **kw)
        except Exception as e:                        # noqa: BLE001
            return BackendResult(None, error=f"SAM inference failed: {e}")

        md = None
        if res and res[0].masks is not None:
            md = res[0].masks.data.cpu().numpy()      # (N, h, w)
        if (md is None or md.shape[0] == 0) and box is not None:
            try:                                     # retry with box only
                res = model(img, bboxes=[box], retina_masks=True, verbose=False,
                            device=self.device)
                if res and res[0].masks is not None:
                    md = res[0].masks.data.cpu().numpy()
            except Exception:                        # noqa: BLE001
                md = None
        if md is None or md.shape[0] == 0:
            return BackendResult(None, error="SAM returned no mask for these prompts "
                                            "(try more n_pos, enable use_box, or a different 'on')")
        m = (md.sum(axis=0) > 0).astype(np.uint8)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = m * 255
        meta = {
            "model": Path(str(self._resolved)).name, "prompted_on": p["on"],
            "n_pos": int((labels == 1).sum()), "n_neg": int((labels == 0).sum()),
            "used_box": bool("bboxes" in kw), "geom_fallback": used_geom,
            "prior_water_pct": prior_meta["water_pct"],
            "water_pct": round(100 * float((mask > 0).mean()), 2),
        }
        return BackendResult(mask, time.time() - t0, meta)


# ---------------------------------------------------------------------------
# Interactive click-to-segment (SAM2)
#
# NOT a Backend: backends map a scene to a mask in one shot, this one holds
# conversational state across many requests.
#
# The whole design follows from one measurement: encoding a 1296x972 scene with
# SAM2.1-L costs ~20 s on CPU, while decoding a click against a cached
# embedding costs ~60-150 ms. ultralytics exposes exactly that split —
# `set_image()` stores `predictor.features`, and `prompt_inference()` reuses
# them instead of re-encoding. So the encoder runs once per (scene, view) and
# every click after it is interactive.
#
# Only the promptable SAM/SAM2 predictors work this way. FastSAM is a YOLO-seg
# model whose "prompting" is post-hoc selection among masks it already
# produced; there is no embedding to cache, so it is rejected here.
# ---------------------------------------------------------------------------

class InteractiveSam:
    """One cached SAM2 image embedding + fast prompt decoding."""

    def __init__(self, weights: str | None = None, model_type: str = "sam2",
                 allow_download: bool = False, device: str | None = "auto"):
        self.device = resolve_device(device)
        self._resolved, self.model_type = resolve_sam_weights(
            weights, allow_download, model_type or "sam2")
        self._pred = None
        self.key: tuple | None = None          # (scene_id, on) currently encoded
        # One embedding is cached at a time, and torch is not reentrant, so all
        # encode/decode traffic serialises here. Two people clicking different
        # scenes at once will thrash the cache — acceptable for a localhost tool.
        self.lock = threading.RLock()

    def available(self) -> tuple[bool, str]:
        try:
            import ultralytics  # noqa: F401
        except Exception:                                 # noqa: BLE001
            return False, ("ultralytics is not installed — run `uv sync` "
                           "(or `uv sync --group sam`)")
        if self.model_type == "fastsam":
            return False, ("click-to-segment needs a promptable SAM/SAM2 checkpoint — "
                           "FastSAM has no reusable image embedding. Point "
                           "backends.sam2.weights at a sam2*.pt")
        if not self._resolved:
            return False, ("no SAM2 weights — set backends.sam2.weights to a sam2*.pt, "
                           "or backends.sam2.allow_download=true to fetch one")
        return True, ""

    def info(self) -> dict:
        ok, reason = self.available()
        return {"available": ok, "reason": reason,
                "model": Path(str(self._resolved)).name if self._resolved else None,
                "device": self.device}

    def _predictor(self):
        if self._pred is None:
            from ultralytics.models.sam import Predictor, SAM2Predictor
            cls = SAM2Predictor if self.model_type == "sam2" else Predictor
            self._pred = cls(overrides=dict(
                conf=0.25, task="segment", mode="predict", imgsz=1024,
                model=self._resolved, verbose=False, save=False, device=self.device))
        return self._pred

    def prepare(self, key: tuple, image_bgr: np.ndarray) -> dict:
        """Encode `image_bgr` unless `key` is already the cached embedding."""
        with self.lock:
            if self.key == key and self._pred is not None and self._pred.features is not None:
                return {"ok": True, "cached": True, "elapsed_s": 0.0}
            t0 = time.time()
            pred = self._predictor()
            try:
                pred.reset_image()
                pred.set_image(image_bgr)
            except Exception as e:                        # noqa: BLE001
                self.key = None
                return {"ok": False, "error": f"SAM2 encode failed: {e}"}
            self.key = key
            return {"ok": True, "cached": False, "elapsed_s": round(time.time() - t0, 1)}

    def ready(self, key: tuple) -> bool:
        return (self.key == key and self._pred is not None
                and getattr(self._pred, "features", None) is not None)

    def predict(self, key: tuple, points: list, labels: list,
                box: list | None = None, multimask: bool = False) -> tuple[np.ndarray | None, list, str | None]:
        """Decode one prompt. Returns (masks (N,H,W) bool, scores, error).

        Points and labels go in as ONE nested group so that positive and
        negative clicks refine a single object, per the ultralytics prompt-shape
        rule in CLAUDE.md; passing them flat makes each point its own object.
        """
        with self.lock:
            if not self.ready(key):
                return None, [], "scene not prepared — call prepare first"
            kw: dict = {}
            if points:
                kw["points"] = [list(points)]
                kw["labels"] = [list(labels)]
            if box:
                kw["bboxes"] = [list(box)]
            if not kw:
                return None, [], "no prompt given"
            try:
                res = self._pred(multimask_output=bool(multimask), **kw)
            except Exception as e:                        # noqa: BLE001
                return None, [], f"SAM2 inference failed: {e}"
            if not res or res[0].masks is None:
                return None, [], "SAM2 returned no mask for that prompt"
            md = res[0].masks.data.cpu().numpy() > 0.5
            # `.masks` can be present but empty — a box with no point prompt
            # comes back as a (0, H, W) stack rather than None.
            if md.shape[0] == 0:
                return None, [], ("SAM2 found nothing for that prompt — add a point "
                                  "inside the water")
            scores = []
            conf = getattr(getattr(res[0], "boxes", None), "conf", None)
            if conf is not None:
                try:
                    scores = [round(float(c), 4) for c in conf.cpu().numpy()]
                except Exception:                         # noqa: BLE001
                    scores = []
            return md, scores, None

    def release(self) -> None:
        """Drop the cached embedding (frees a few hundred MB)."""
        with self.lock:
            if self._pred is not None:
                try:
                    self._pred.reset_image()
                except Exception:                         # noqa: BLE001
                    pass
            self.key = None


def build_interactive(bcfg: dict | None = None, device: str = "auto") -> InteractiveSam:
    """Click-to-segment session, configured from the `sam2` backend's settings."""
    bcfg = bcfg or {}
    cfg = bcfg.get("sam2", {})
    return InteractiveSam(weights=cfg.get("weights"),
                          model_type=cfg.get("model", "sam2"),
                          allow_download=bool(cfg.get("allow_download", False)),
                          device=cfg.get("device", device))


# ---------------------------------------------------------------------------
# 7. Change vs. dry background  (fixed-camera advantage)
#    median background from sibling captures under the same parent directory.
# ---------------------------------------------------------------------------

class ChangeBackend(Backend):
    name = "change"
    label = "Change vs. dry background"
    params_schema = [
        {"name": "k_refs", "type": "int", "default": 6, "min": 2, "max": 20, "step": 1,
         "help": "How many sibling captures to build the background from."},
        {"name": "min_refs", "type": "int", "default": 3, "min": 2, "max": 10, "step": 1,
         "help": "Minimum siblings required, else the backend errors for this scene."},
    ]

    def run(self, scene_dir, tiff_path, params):
        import cv2
        import rasterio
        from scipy.ndimage import binary_fill_holes

        from autolabel import _grad_mag, _norm
        t0 = time.time()
        p = {"k_refs": 6, "min_refs": 3}
        p.update(params or {})

        sibs = []
        for d in sorted(scene_dir.parent.iterdir()):
            if not d.is_dir() or d == scene_dir:
                continue
            for nm in ("color_preserved_5_band.tiff", "final_5_band.tiff"):
                if (d / nm).exists():
                    sibs.append(d / nm)
                    break
        if len(sibs) < int(p["min_refs"]):
            return BackendResult(None, error=(
                f"only {len(sibs)} sibling capture(s) under {scene_dir.parent.name}/; "
                f"need >= {p['min_refs']} of the same view at other times"))

        try:
            arr = _read_bands(tiff_path)
        except Exception as e:                        # noqa: BLE001
            return BackendResult(None, error=str(e))
        H, W = arr.shape[1], arr.shape[2]
        cur_nir, cur_g = arr[B_NIR], arr[B_G]
        cur_gray = (0.299 * arr[B_R] + 0.587 * arr[B_G] + 0.114 * arr[B_B]).astype(np.float32)

        stack_nir, stack_g = [], []
        for sp in sibs[: int(p["k_refs"])]:
            try:
                with rasterio.open(sp) as s:
                    ref = s.read().astype(np.float32)
            except Exception:                         # noqa: BLE001
                continue
            rn, rg = ref[B_NIR], ref[B_G]
            if rn.shape != (H, W):
                rn = cv2.resize(rn, (W, H)); rg = cv2.resize(rg, (W, H))
            stack_nir.append(rn); stack_g.append(rg)
        if len(stack_nir) < int(p["min_refs"]):
            return BackendResult(None, error="sibling TIFFs unreadable")

        bg_nir = np.median(np.stack(stack_nir), axis=0)
        bg_g = np.median(np.stack(stack_g), axis=0)

        nir_drop = _norm(np.clip(bg_nir - cur_nir, 0, None))       # NIR fell -> water appeared
        ndwi_cur = (cur_g - cur_nir) / (cur_g + cur_nir + 1e-6)
        ndwi_bg = (bg_g - bg_nir) / (bg_g + bg_nir + 1e-6)
        ndwi_rise = _norm(np.clip(ndwi_cur - ndwi_bg, 0, None))
        tex = _norm(cv2.GaussianBlur(_grad_mag(cur_gray), (0, 0), 2))

        score = 0.5 * nir_drop + 0.35 * ndwi_rise + 0.15 * (1.0 - tex)
        mask = (score > 0.55).astype(np.uint8)
        mask = _clean(mask.astype(bool), open_iter=1, close_iter=1)
        mask = binary_fill_holes(mask).astype(np.uint8)
        return BackendResult(mask * 255, time.time() - t0, {
            "n_refs": len(stack_nir),
            "water_pct": round(100 * float(mask.mean()), 2),
        })


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def build_registry(bcfg: dict | None = None, device: str = "auto") -> dict[str, Backend]:
    """bcfg is config['backends'] — a dict of {backend_name: {enabled, ...settings}}.

    ``device`` ('auto' | 'cpu' | 'cuda' | 'cuda:0' | 'mps') is the default for the
    torch/ultralytics backends; a backend may override it with backends.<n>.device.
    A backend with ``enabled: false`` is left out of the registry entirely.
    """
    bcfg = bcfg or {}
    gdev = resolve_device(device)

    def enabled(name: str) -> bool:
        return bcfg.get(name, {}).get("enabled", True)

    def opt(name: str, key: str, default=None):
        return bcfg.get(name, {}).get(key, default)

    def dev(name: str) -> str:
        return resolve_device(opt(name, "device", gdev))

    reg: dict[str, Backend] = {}
    # SAM backends first: when weights are present they're the strongest, so the
    # UI dropdown (which defaults to the first *available* one) lands there.
    # `sam2` = the heavy high-quality pass (SAM2.1); `sam` = the fast pass.
    if enabled("sam2"):
        reg["sam2"] = SamPromptedBackend(
            name="sam2", label="SAM2.1 (spectral-prompted, high quality)",
            weights=opt("sam2", "weights"),
            model_type=opt("sam2", "model", "sam2"),
            allow_download=bool(opt("sam2", "allow_download", False)),
            device=dev("sam2"))
    if enabled("sam"):
        reg["sam"] = SamPromptedBackend(
            name="sam", label="SAM (spectral-prompted, fast)",
            weights=opt("sam", "weights"),
            model_type=opt("sam", "model", "fastsam"),
            allow_download=bool(opt("sam", "allow_download", False)),
            device=dev("sam"))
    if enabled("spectral"):
        reg["spectral"] = SpectralBackend()
    if enabled("nir"):
        reg["nir"] = NirThresholdBackend()
    if enabled("thermal"):
        reg["thermal"] = ThermalOtsuBackend()
    if enabled("change"):
        reg["change"] = ChangeBackend()
    if enabled("tinysam"):
        reg["tinysam"] = TinySamBackend(
            python_exe=opt("tinysam", "python"),
            tinysam_path=opt("tinysam", "tinysam_path"),
            script=opt("tinysam", "script"),
            weights=opt("tinysam", "weights"))
    if enabled("segformer"):
        size = opt("segformer", "size")
        if isinstance(size, str) and "x" in size.lower():
            h, w = size.lower().split("x")
            size = (int(h), int(w))
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            size = (int(size[0]), int(size[1]))
        else:
            size = None
        reg["segformer"] = SegformerOnnxBackend(
            onnx_path=opt("segformer", "onnx"),
            size=size,
            water_class=int(opt("segformer", "water_class", 1)),
            device=dev("segformer"))
    return reg
