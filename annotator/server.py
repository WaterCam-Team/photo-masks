#!/usr/bin/env python
"""Local web app for labeling UFONet 5-band water segmentation training data.

    manual painting  +  switchable auto-segmentation backends  +  RL decision log

Runs on the standard library only (http.server); needs a Python with
rasterio, numpy, opencv, scipy, Pillow (see requirements.txt / pyproject.toml).

    uv run server.py                       # uses annotator_config.json if present
    uv run server.py --config my.json      # explicit config file
    uv run server.py --root /data/captures --disable segformer
    uv run server.py --print-config        # show the resolved config and exit
    # then open http://localhost:8000   (tunnel the port if running over SSH)

Input scenes are 5-band TIFFs that a capture rig has ALREADY co-registered;
co-registration is not part of this repo.

Configuration (photo roots + which backends are available) comes from, in
increasing precedence: built-in defaults  <  a JSON config file  <  CLI flags.
Config file is --config PATH, else ./annotator_config.json, else
<this dir>/annotator_config.json.  See annotator_config.example.json.

Outputs, per scene, next to the 5-band TIFF:
    water_mask.png              accepted training mask (0 / 255)   <- export_dataset.py reads this
    _annotator_auto_<b>.png     last auto mask from backend <b>    (transient seed)

Work dir (default: ./work):
    state.json                  per-scene review status
    label_log.csv               one row per labeling decision (the RL dataset)
    features/<scene>.json       cached scene features
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import backends as backends_mod
import rl_features as rlf

HERE = Path(__file__).resolve().parent
FRONTEND = HERE / "frontend"
REPO_ROOT = HERE.parent                           # .../photo_processing

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
# Precedence (low -> high):  DEFAULT_CONFIG  <  config file  <  CLI flags
# Config file search order:  --config PATH  >  ./annotator_config.json (cwd)
#                            >  <this dir>/annotator_config.json
# Relative roots resolve against the repo root (.../photo_processing), so the
# bundled example_data/ works from a fresh clone with no configuration at all.

DEFAULT_CONFIG = {
    "roots": ["example_data"],
    "tiff_names": ["color_preserved_5_band.tiff", "final_5_band.tiff"],
    "exclude_dirs": [".git", ".venv", "venv", "node_modules",
                     "site-packages", "__pycache__"],
    "max_depth": 6,
    "host": "127.0.0.1",
    "port": 8000,
    "annotator": "anon",
    "work_dir": str(HERE / "work"),
    "device": "auto",              # auto | cpu | cuda | cuda:0 | mps  (torch/ultralytics backends + trainer)
    "backends": {
        "spectral": {"enabled": True},
        "nir": {"enabled": True},
        "thermal": {"enabled": True},
        "change": {"enabled": True},
        "sam": {"enabled": True, "weights": None, "model": "fastsam",
                "allow_download": False},
        "sam2": {"enabled": True, "weights": None, "model": "sam2",
                 "allow_download": False},
        "tinysam": {"enabled": True, "python": None, "tinysam_path": None,
                    "script": None, "weights": None},
        "segformer": {"enabled": True, "onnx": None, "size": None, "water_class": 1},
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def find_config(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise SystemExit(f"--config file not found: {p}")
        return p
    for cand in (Path.cwd() / "annotator_config.json",
                 HERE / "annotator_config.json"):
        if cand.exists():
            return cand
    return None


def load_config(explicit: str | None) -> tuple[dict, Path | None]:
    path = find_config(explicit)
    cfg = _deep_merge(DEFAULT_CONFIG, {})
    if path:
        raw = json.loads(path.read_text())
        for k in [k for k in raw if k.startswith("_help")]:
            raw.pop(k, None)
        cfg = _deep_merge(cfg, raw)
    return cfg, path


def resolve_roots(roots: list[str]) -> list[Path]:
    out = []
    for r in roots:
        p = Path(r).expanduser()
        p = p if p.is_absolute() else (REPO_ROOT / p)
        out.append(p.resolve())                       # match the per-scene 'root' key
    return out


MANIFEST_FIELDS = [
    "scene_dir", "tiff_path", "autolabel_status",
    "mask_path", "water_pct", "review_status", "updated",
]

# Disagreement-map palette (R, G, B). Colour-vision-deficiency safe: teal /
# amber / magenta, deliberately avoiding a red-green pair. Mirrored by the
# legend swatches in frontend/style.css — change both together.
#
# Hue says *which way* the vote went; ALPHA says *how contested* it is, scaled
# by closeness to a tie. Without that, a single over-segmenting backend (e.g.
# `nir` calling the sky water) floods the whole frame and hides the real signal.
AGREE_RGB = (0, 190, 200)                # every backend agrees: water
MAJORITY_RGB = (245, 180, 0)             # most say water
CONTESTED_RGB = (230, 70, 200)           # only a minority say water
AGREE_ALPHA = 90                         # settled — present but quiet
CONTESTED_ALPHA_MIN, CONTESTED_ALPHA_MAX = 18, 215

# on-disk masks the UI offers to load as a starting point
EXISTING_MASKS = [
    ("water_mask.png", "manual (saved)"),
    ("water_mask_auto.png", "NIR auto"),
    ("water_mask_thermal.png", "thermal auto"),
    ("tinysam_water_mask.png", "TinySAM"),
]


# ===========================================================================
# scene discovery
# ===========================================================================

def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")


def clean_annotator(name, fallback: str) -> str:
    """Sanitise a client-supplied labeler name for the CSV log.

    Several people can share one running server, each with their own name set
    in the UI, so this value is untrusted. Strip control characters, collapse
    whitespace, cap the length, and fall back to the server's --annotator when
    the field is blank.
    """
    if not isinstance(name, str):
        return fallback
    name = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", "", name)).strip()
    return name[:40] if name else fallback


def _scene_id(scene_dir: Path) -> str:
    """Stable, readable id: <parent>__<name>-<6 hex of the resolved path>.

    Stable across restarts / data reorganisation because it's derived only
    from the scene directory's own path, not from discovery order.
    """
    h = hashlib.sha1(str(scene_dir).encode()).hexdigest()[:6]
    return f"{_slug(f'{scene_dir.parent.name}__{scene_dir.name}')}-{h}"


def discover_scenes(roots: list[Path], tiff_names: list[str],
                    exclude_dirs: list[str], max_depth: int = 6) -> list[dict]:
    import os as _os

    skip = set(exclude_dirs)
    # scene_dir -> (tiff_path, manifest_path|None, root)  ; prefer the first-listed name
    found: dict[Path, tuple[Path, Path | None, Path]] = {}
    order: list[Path] = []
    seen_real: set[str] = set()                      # realpath guard against symlink loops

    for root in roots:
        if not root.exists():
            continue
        root = root.resolve()
        root_manifest = root / "manifest.csv"
        root_manifest = root_manifest if root_manifest.exists() else None
        base_depth = len(root.parts)

        # os.walk with followlinks=True so a symlinked data dir under a root is scanned
        for dirpath, dirnames, filenames in _os.walk(root, followlinks=True):
            d = Path(dirpath)
            rp = _os.path.realpath(dirpath)
            if rp in seen_real:
                dirnames[:] = []
                continue
            seen_real.add(rp)
            dirnames[:] = sorted(x for x in dirnames if x not in skip)
            if len(d.parts) - base_depth >= max_depth:
                dirnames[:] = []
            if d in found:
                continue
            tiff = next((d / n for n in tiff_names if n in filenames), None)
            if tiff is None:
                continue
            man = root_manifest
            if (d / "manifest.csv").exists():
                man = d / "manifest.csv"
            found[d] = (tiff, man, root)
            order.append(d)

    scenes: list[dict] = []
    for sd in order:
        tiff, manifest, root = found[sd]
        scenes.append({
            "id": _scene_id(sd),
            "name": f"{sd.parent.name}/{sd.name}",
            "dir": str(sd),
            "tiff": str(tiff),
            "manifest": str(manifest) if manifest else None,
            "root": str(root),
        })
    return scenes


# ===========================================================================
# rendering helpers
# ===========================================================================

MAX_BODY_BYTES = 64 * 1024 * 1024


def _read5(tiff_path: str) -> np.ndarray:
    import rasterio
    with rasterio.open(tiff_path) as src:
        if src.count < 5:
            raise ValueError(f"{Path(tiff_path).name}: {src.count} band(s), need 5")
        return src.read().astype(np.float32)


def _tiff_hw(tiff_path: str) -> tuple[int, int]:
    """(height, width) from the TIFF header — no pixel decode.

    Routes that only need the mask geometry use this instead of _read5(); a
    full 5-band decode of a 1296x972 scene is ~25 MB of float32 per request.
    """
    import rasterio
    with rasterio.open(tiff_path) as src:
        if src.count < 5:
            raise ValueError(f"{Path(tiff_path).name}: {src.count} band(s), need 5")
        return int(src.height), int(src.width)


def _stretch(x: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    a, b = np.percentile(x, [lo, hi])
    if b <= a:
        return np.zeros_like(x, dtype=np.uint8)
    return np.clip((x - a) / (b - a) * 255.0, 0, 255).astype(np.uint8)


def render_view(arr: np.ndarray, layer: str, raw: bool = False) -> np.ndarray:
    import cv2
    r, g, b, th, nir = arr[0], arr[1], arr[2], arr[3], arr[4]

    if layer == "rgb":
        return np.dstack([r, g, b]).clip(0, 255).astype(np.uint8)

    if layer == "false":                 # colour-IR: NIR->R  R->G  G->B  (water very dark)
        return np.dstack([_stretch(nir), _stretch(r), _stretch(g)])

    if layer == "nir":
        gimg = _stretch(nir)
        return gimg if raw else np.dstack([gimg, gimg, gimg])

    if layer == "thermal":
        gimg = _stretch(th)
        if raw:
            return gimg
        c = cv2.applyColorMap(gimg, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

    if layer == "ndwi":
        nd = (g - nir) / (g + nir + 1e-6)
        gimg = np.clip((nd + 1) / 2 * 255, 0, 255).astype(np.uint8)
        if raw:
            return gimg
        cmap = getattr(cv2, "COLORMAP_BWR", cv2.COLORMAP_JET)
        c = cv2.applyColorMap(gimg, cmap)
        return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

    return np.dstack([r, g, b]).clip(0, 255).astype(np.uint8)


def png_bytes(img: np.ndarray) -> bytes:
    from PIL import Image
    if img.ndim == 2:
        im = Image.fromarray(img, mode="L")
    else:
        im = Image.fromarray(img, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def read_mask_file(path: Path, size_hw: tuple[int, int] | None = None) -> np.ndarray | None:
    import cv2
    if not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if size_hw and m.shape != size_hw:
        m = cv2.resize(m, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8) * 255


def decode_mask_png(b64: str, size_hw: tuple[int, int]) -> np.ndarray:
    import cv2
    from PIL import Image
    raw = base64.b64decode(b64.split(",", 1)[-1])
    im = Image.open(io.BytesIO(raw)).convert("L")
    m = np.array(im)
    if m.shape != size_hw:
        m = cv2.resize(m, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8) * 255


# ===========================================================================
# app state
# ===========================================================================

# ===========================================================================
# fine-tune orchestration
# ===========================================================================

class TrainManager:
    """One SegFormer fine-tune run at a time: export gold -> train -> use / onnx."""

    def __init__(self, work_dir: Path, device: str = "auto"):
        self.work_dir = work_dir
        self.device = device
        self.runs_dir = work_dir / "runs"
        self.dataset_dir = work_dir / "dataset"
        self.proc: subprocess.Popen | None = None
        self.run_dir: Path | None = None
        self.phase = "idle"                          # idle|training|done|error|stopped
        self.cfg: dict = {}
        self.lock = threading.Lock()
        # remember the newest finished run across restarts so Use / Export-ONNX work
        prev = sorted((p for p in self.runs_dir.glob("*") if (p / "best_hf").exists()),
                      key=lambda p: p.stat().st_mtime) if self.runs_dir.exists() else []
        if prev:
            self.run_dir = prev[-1]
            self.phase = "done"
            try:
                self.cfg = json.loads((self.run_dir / "metrics.json").read_text()).get("args", {})
            except Exception:                        # noqa: BLE001
                pass

    # -- gold dataset ------------------------------------------------
    def export_gold(self, label_log: Path) -> dict:
        script = (HERE.parent / "export_dataset.py").resolve()
        cmd = [sys.executable, str(script), "--gold-only", str(self.dataset_dir),
               "--label-log", str(label_log), "--val-split", "0.2"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

        def count(split):
            d = self.dataset_dir / "img_dir" / split
            return len(list(d.glob("*.tif*"))) if d.exists() else 0

        prov = self.dataset_dir / "dataset_provenance.csv"
        routes: dict[str, int] = {}
        if prov.exists():
            with prov.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    routes[row["route"]] = routes.get(row["route"], 0) + 1
        return {"ok": r.returncode == 0, "train": count("train"), "val": count("val"),
                "routes": routes, "log": (r.stdout + r.stderr)[-2000:]}

    # -- training ---------------------------------------------------
    def start(self, params: dict) -> dict:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return {"ok": False, "error": "a run is already in progress"}
            if not (self.dataset_dir / "img_dir" / "train").exists():
                return {"ok": False, "error": "export the gold dataset first"}
            # fail here with a clear message rather than in the subprocess
            import importlib.util as _iu
            missing = [m for m in ("torch", "transformers") if _iu.find_spec(m) is None]
            if missing:
                return {"ok": False, "error":
                        f"training needs {' and '.join(missing)} — run `uv sync` "
                        f"(or `uv sync --group train`)"}
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_dir = self.runs_dir / ts
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.cfg = {
                "epochs": int(params.get("epochs", 40)),
                "lr": float(params.get("lr", 6e-5)),
                "batch": int(params.get("batch", 2)),
                "img_size": int(params.get("img_size", 512)),
                "init": params.get("init", "nvidia/mit-b0"),
                "device": params.get("device") or self.device,
            }
            cmd = [sys.executable, str(HERE / "trainer.py"),
                   "--data-root", str(self.dataset_dir), "--out", str(self.run_dir),
                   "--epochs", str(self.cfg["epochs"]), "--lr", str(self.cfg["lr"]),
                   "--batch", str(self.cfg["batch"]), "--img-size", str(self.cfg["img_size"]),
                   "--init", str(self.cfg["init"]), "--device", str(self.cfg["device"])]
            with (self.run_dir / "train.log").open("w") as log:
                # child dups the fd; parent can close its copy immediately
                self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                             cwd=str(HERE))
            self.phase = "training"
            return {"ok": True, "run": ts, "cfg": self.cfg}

    def stop(self) -> dict:
        if not (self.proc and self.proc.poll() is None):
            return {"ok": False, "error": "no active run"}
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.phase = "stopped"
        return {"ok": True}

    def _events(self) -> list[dict]:
        out = []
        if self.run_dir and (self.run_dir / "train.log").exists():
            for ln in (self.run_dir / "train.log").read_text().splitlines()[-4000:]:
                ln = ln.strip()
                if ln.startswith("{"):
                    try:
                        out.append(json.loads(ln))
                    except Exception:                # noqa: BLE001
                        pass
        return out

    def status(self) -> dict:
        running = bool(self.proc and self.proc.poll() is None)
        ev = self._events()
        steps = [e for e in ev if e.get("event") == "step"]
        evals = [e for e in ev if e.get("event") == "eval"]
        errs = [e for e in ev if e.get("event") == "error"]
        done = [e for e in ev if e.get("event") == "done"]
        starts = [e for e in ev if e.get("event") == "start"]
        phase = self.phase
        if not running and phase == "training":
            phase = "error" if errs else ("done" if done else "stopped")
            self.phase = phase
        st = starts[-1] if starts else {}
        return {
            "phase": phase, "running": running,
            "run": self.run_dir.name if self.run_dir else None,
            "cfg": self.cfg,
            "device": st.get("device"), "gpu": st.get("gpu"), "amp": st.get("amp"),
            "steps": [{"step": e["step"], "loss": e["loss"], "lr": e.get("lr")}
                      for e in steps[-60:]],
            "evals": [{"epoch": e["epoch"], "miou": e["miou"],
                       "iou_water": e.get("iou_water"), "best": e.get("best")}
                      for e in evals],
            "best_miou": (done[0]["best_miou"] if done
                          else max((e["miou"] for e in evals), default=None)),
            "has_ckpt": bool(self.run_dir and (self.run_dir / "best_hf").exists()),
            "error": errs[-1].get("msg") if errs else None,
            "info": [e.get("msg") for e in ev if e.get("event") in ("info", "warn")][-5:],
        }

    def export_onnx(self, onnx_out: Path) -> dict:
        if not self.run_dir or not (self.run_dir / "best_hf").exists():
            return {"ok": False, "error": "no finished run with a checkpoint"}
        cmd = [sys.executable, str(HERE / "trainer.py"), "--export-onnx",
               str(self.run_dir / "best_hf"), "--onnx-out", str(onnx_out),
               "--img-size", str(self.cfg.get("img_size", 512))]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        return {"ok": r.returncode == 0,
                "path": str(onnx_out) if r.returncode == 0 and onnx_out.exists() else None,
                "log": (r.stdout + r.stderr)[-1500:]}


class App:
    def __init__(self, config: dict, work_dir: Path, config_path: Path | None):
        self.config = config
        self.config_path = config_path
        self.roots = resolve_roots(config["roots"])
        self.work_dir = work_dir
        self.annotator = config["annotator"]
        self.feat_dir = work_dir / "features"
        self.feat_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = work_dir / "state.json"
        self.log_path = work_dir / "label_log.csv"
        self.lock = threading.Lock()

        self.scenes = discover_scenes(
            self.roots, config["tiff_names"], config["exclude_dirs"],
            int(config["max_depth"]))
        self.by_id = {s["id"]: s for s in self.scenes}
        self.device = backends_mod.resolve_device(config.get("device", "auto"))
        self.registry = backends_mod.build_registry(config["backends"], device=self.device)
        self.trainer = TrainManager(work_dir, device=config.get("device", "auto"))
        self.state = self._load_state()

    def scenes_per_root(self) -> dict:
        counts = {str(r): 0 for r in self.roots}
        for s in self.scenes:
            rt = s.get("root")
            if rt in counts:
                counts[rt] += 1
        return counts

    # -- transient (auto/ensemble) masks: work/cache/<id>/, not next to the TIFF
    def cache_dir(self, scene_id: str) -> Path:
        d = self.work_dir / "cache" / scene_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- persistence -------------------------------------------------------
    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except Exception:                         # noqa: BLE001
                pass
        return {}

    def _save_state(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def features(self, scene: dict) -> dict:
        fp = self.feat_dir / f"{scene['id']}.json"
        if fp.exists():
            try:
                return json.loads(fp.read_text())
            except Exception:                         # noqa: BLE001
                pass
        feats = rlf.scene_features(Path(scene["tiff"]))
        self.feat_dir.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(feats))
        return feats

    def sync_manifest(self, scene: dict, review_status: str,
                      mask_path: str, water_pct: float) -> None:
        mpath = scene.get("manifest")
        if not mpath:
            return
        mpath = Path(mpath)
        rows: dict[str, dict] = {}
        if mpath.exists():
            with mpath.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    rows[row.get("scene_dir", "")] = row
        key = scene["dir"]
        row = rows.get(key, {k: "" for k in MANIFEST_FIELDS})
        row["scene_dir"] = key
        row["tiff_path"] = scene["tiff"]
        row["autolabel_status"] = row.get("autolabel_status") or "ok"
        row["mask_path"] = mask_path
        row["water_pct"] = f"{water_pct:.1f}"
        row["review_status"] = review_status
        row["updated"] = datetime.now().strftime("%Y%m%d-%H%M%S")
        rows[key] = row
        with mpath.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows.values())

    # -- scene list ------------------------------------------------------
    def scene_list(self) -> list[dict]:
        out = []
        for s in self.scenes:
            st = self.state.get(s["id"], {})
            out.append({
                "id": s["id"], "name": s["name"],
                "review_status": st.get("review_status", "pending"),
                "water_pct": st.get("final_water_pct"),
                "route": st.get("route"),
                "updated": st.get("updated"),
                "has_manual": (Path(s["dir"]) / "water_mask.png").exists(),
            })
        return out


# ===========================================================================
# HTTP handler
# ===========================================================================

class Handler(BaseHTTPRequestHandler):
    app: App | None = None                            # set in main()
    allowed_hosts: set[str] | None = None             # set in main(); None = unrestricted
    server_version = "UFONetAnnotator/1.0"

    # -- origin checks ---------------------------------------------------
    # There is no authentication (see the README callout), so the only thing
    # standing between a *local* server and a malicious web page the operator
    # happens to visit is the browser. Two cheap checks close that gap:
    #
    #   Host   — a DNS-rebinding attack reaches us with an attacker-controlled
    #            hostname in Host, so pinning it to what we bound rejects the
    #            request before any scene data is read.
    #   Origin — a cross-site fetch/form POST carries an Origin that isn't ours.
    #            Same-origin XHR sends either no Origin or our own.
    #
    # Neither is authentication; both are free, and without them "localhost
    # only" is not actually a boundary.
    def _origin_ok(self) -> bool:
        allowed = self.allowed_hosts
        if allowed is None:               # bound to a wildcard address: no name to pin
            return True
        host = (self.headers.get("Host") or "").split(",")[0].strip()
        hostname = host.rsplit(":", 1)[0].strip("[]").lower() if host else ""
        if hostname and hostname not in allowed:
            return False
        origin = self.headers.get("Origin")
        if origin:
            o = (urlparse(origin).hostname or "").lower()
            if o not in allowed:
                return False
        return True

    # -- low-level replies ---------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _png(self, img: np.ndarray) -> None:
        self._send(200, png_bytes(img), "image/png")

    def _err(self, code: int, msg: str) -> None:
        self._json({"ok": False, "error": msg}, code)

    def log_message(self, fmt, *args):               # quieter console
        if "api/scene" in (self.path or "") and "view" in (self.path or ""):
            return
        super().log_message(fmt, *args)

    def _body_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n > MAX_BODY_BYTES:
            self.close_connection = True
            raise ValueError(f"request body too large ({n} bytes, max {MAX_BODY_BYTES})")
        if not n:
            return {}
        return json.loads(self.rfile.read(n) or b"{}")

    # -- routing ------------------------------------------------------
    def do_GET(self):
        self._dispatch(self._route_get)

    def do_POST(self):
        self._dispatch(self._route_post)

    def _dispatch(self, fn):
        try:
            if not self._origin_ok():
                return self._err(403, "rejected: unexpected Host/Origin header — "
                                      "reach the annotator at the address it "
                                      "printed on startup")
            fn()
        except BrokenPipeError:
            pass
        except ValueError as e:                       # bad input (band count, body size, bad param)
            self._err(422, str(e))
        except Exception as e:                        # noqa: BLE001
            traceback.print_exc()
            self._err(500, str(e))

    # -- GET routes -------------------------------------------------
    def _route_get(self):
        u = urlparse(self.path)
        p = u.path
        q = parse_qs(u.query)
        app = self.app

        if p in ("/", "/index.html"):
            return self._send(200, (FRONTEND / "index.html").read_bytes(), "text/html")
        if p == "/static/app.js":
            return self._send(200, (FRONTEND / "app.js").read_bytes(),
                              "application/javascript")
        if p == "/static/style.css":
            return self._send(200, (FRONTEND / "style.css").read_bytes(), "text/css")
        if p == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if p == "/api/config":
            return self._json({
                "annotator": app.annotator,
                "backends": [b.info() for b in app.registry.values()],
                "layers": ["rgb", "false", "nir", "thermal", "ndwi"],
                "n_scenes": len(app.scenes),
                "config_file": str(app.config_path) if app.config_path else None,
                "roots": [str(r) for r in app.roots],
                "scenes_per_root": app.scenes_per_root(),
                "tiff_names": app.config["tiff_names"],
                "device": backends_mod.device_label(app.config.get("device", "auto")),
                "device_hint": backends_mod.accel_hint(),
            })

        if p == "/api/scenes":
            return self._json({"scenes": app.scene_list()})

        if p == "/api/train/status":
            return self._json(app.trainer.status())

        m = re.match(r"^/api/scene/([^/]+)/view$", p)
        if m:
            scene = app.by_id.get(m.group(1))
            if not scene:
                return self._err(404, "no such scene")
            layer = q.get("layer", ["false"])[0]
            raw = q.get("raw", ["0"])[0] == "1"
            arr = _read5(scene["tiff"])
            return self._png(render_view(arr, layer, raw=raw))

        m = re.match(r"^/api/scene/([^/]+)/maskfile$", p)
        if m:
            scene = app.by_id.get(m.group(1))
            if not scene:
                return self._err(404, "no such scene")
            name = q.get("name", [""])[0]
            if not re.fullmatch(r"[A-Za-z0-9._-]+\.png", name or ""):
                return self._err(400, "bad name")
            base = app.cache_dir(scene["id"]) if name.startswith("_annotator_") \
                else Path(scene["dir"])
            if q.get("fmt", ["raw"])[0] == "asis":     # already-coloured RGBA overlay
                p_img = base / name
                if not p_img.exists():
                    return self._err(404, "overlay not found")
                return self._send(200, p_img.read_bytes(), "image/png")
            hw = _tiff_hw(scene["tiff"])
            mask = read_mask_file(base / name, hw)
            if mask is None:
                return self._err(404, "mask not found")
            if q.get("fmt", ["raw"])[0] == "rgba":
                rgba = np.zeros((*hw, 4), np.uint8)
                rgba[mask > 0] = (0, 200, 255, 170)
                from PIL import Image
                buf = io.BytesIO()
                Image.fromarray(rgba, "RGBA").save(buf, "PNG")
                return self._send(200, buf.getvalue(), "image/png")
            return self._png(mask)

        m = re.match(r"^/api/scene/([^/]+)$", p)
        if m:
            scene = app.by_id.get(m.group(1))
            if not scene:
                return self._err(404, "no such scene")
            hw = list(_tiff_hw(scene["tiff"]))
            existing = [{"name": n, "label": lbl}
                        for n, lbl in EXISTING_MASKS
                        if (Path(scene["dir"]) / n).exists()]
            for f in sorted(app.cache_dir(scene["id"]).glob("_annotator_auto_*.png")):
                existing.append({"name": f.name,
                                 "label": f"last {f.stem.replace('_annotator_auto_', '')}"})
            st = app.state.get(scene["id"], {})
            return self._json({
                "id": scene["id"], "name": scene["name"], "size": hw,
                "review_status": st.get("review_status", "pending"),
                "existing_masks": existing,
            })

        return self._err(404, "not found")

    # -- POST routes ----------------------------------------------
    def _route_post(self):
        u = urlparse(self.path)
        p = u.path
        app = self.app

        m = re.match(r"^/api/scene/([^/]+)/autoseg$", p)
        if m:
            scene = app.by_id.get(m.group(1))
            if not scene:
                return self._err(404, "no such scene")
            data = self._body_json()
            bname = data.get("backend")
            backend = app.registry.get(bname)
            if not backend:
                return self._err(400, f"unknown backend {bname!r}")
            ok, reason = backend.available()
            if not ok:
                return self._err(400, reason)
            res = backend.run(Path(scene["dir"]), Path(scene["tiff"]),
                              data.get("params", {}))
            if res.error or res.mask is None:
                return self._err(500, res.error or "backend returned no mask")
            cache = app.cache_dir(scene["id"]) / f"_annotator_auto_{bname}.png"
            from PIL import Image
            Image.fromarray(res.mask, "L").save(cache, "PNG")
            return self._json({
                "ok": True, "backend": bname,
                "elapsed_s": round(res.elapsed_s, 2),
                "meta": res.meta,
                "water_pct": res.meta.get("water_pct"),
                "mask_url": f"/api/scene/{scene['id']}/maskfile?name={cache.name}&t={int(time.time())}",
            })

        m = re.match(r"^/api/scene/([^/]+)/ensemble$", p)
        if m:
            scene = app.by_id.get(m.group(1))
            if not scene:
                return self._err(404, "no such scene")
            return self._handle_ensemble(scene, self._body_json())

        m = re.match(r"^/api/scene/([^/]+)/save$", p)
        if m:
            scene = app.by_id.get(m.group(1))
            if not scene:
                return self._err(404, "no such scene")
            return self._handle_save(scene, self._body_json())

        m = re.match(r"^/api/scene/([^/]+)/review$", p)
        if m:
            scene = app.by_id.get(m.group(1))
            if not scene:
                return self._err(404, "no such scene")
            return self._handle_review(scene, self._body_json())

        if p == "/api/train/export":
            return self._json(app.trainer.export_gold(app.log_path))
        if p == "/api/train/start":
            return self._json(app.trainer.start(self._body_json()))
        if p == "/api/train/stop":
            return self._json(app.trainer.stop())
        if p == "/api/train/use":
            tr = app.trainer
            if not (tr.run_dir and (tr.run_dir / "best_hf").exists()):
                return self._err(400, "no finished run with a checkpoint")
            app.registry["segformer"] = backends_mod.SegformerOnnxBackend(
                onnx_path=str(tr.run_dir / "best_hf"),
                water_class=int(app.config["backends"].get("segformer", {}).get("water_class", 1)))
            return self._json({"ok": True, "model": str(tr.run_dir / "best_hf"),
                               "backend": app.registry["segformer"].info()})
        if p == "/api/train/export-onnx":
            return self._json(app.trainer.export_onnx(HERE / "weights" / "segformer_5band.onnx"))

        return self._err(404, "not found")

    # -- ensemble (run all backends, measure agreement) ----------
    def _handle_ensemble(self, scene: dict, data: dict):
        app = self.app
        from PIL import Image

        names = data.get("backends") or [
            n for n, b in app.registry.items() if b.available()[0]]
        params = data.get("params", {})
        hw = _tiff_hw(scene["tiff"])

        cdir = app.cache_dir(scene["id"])
        results, masks = [], {}
        for n in names:
            backend = app.registry.get(n)
            if backend is None:
                results.append({"backend": n, "error": "unknown backend"})
                continue
            ok, reason = backend.available()
            if not ok:
                results.append({"backend": n, "error": reason})
                continue
            res = backend.run(Path(scene["dir"]), Path(scene["tiff"]), params.get(n, {}))
            if res.error or res.mask is None:
                results.append({"backend": n, "error": res.error or "no mask"})
                continue
            m = (res.mask > 127)
            masks[n] = m
            cache = cdir / f"_annotator_auto_{n}.png"
            Image.fromarray((m.astype(np.uint8) * 255), "L").save(cache, "PNG")
            results.append({
                "backend": n, "elapsed_s": round(res.elapsed_s, 2),
                "water_pct": round(100 * float(m.mean()), 2), "meta": res.meta,
                "mask_url": f"/api/scene/{scene['id']}/maskfile?name={cache.name}&t={int(time.time())}",
            })

        ok_names = list(masks)
        matrix, ious = {}, []
        for i, a in enumerate(ok_names):
            for bcol in ok_names[i + 1:]:
                v = rlf.mask_iou(masks[a].astype(np.uint8), masks[bcol].astype(np.uint8))
                matrix[f"{a}|{bcol}"] = round(v, 3)
                ious.append(v)

        consensus_url = votes_url = None
        consensus_pct = disagreement_frac = None
        if ok_names:
            n = len(ok_names)
            stack = np.stack([masks[nm] for nm in ok_names]).astype(np.int16)
            votes = stack.sum(axis=0)
            half = np.ceil(n / 2)
            consensus = (votes >= half).astype(np.uint8)
            disagreement_frac = round(float(((votes > 0) & (votes < n)).mean()), 4)
            consensus_pct = round(100 * float(consensus.mean()), 2)
            cpath = cdir / "_annotator_ensemble.png"
            Image.fromarray(consensus * 255, "L").save(cpath, "PNG")
            consensus_url = (f"/api/scene/{scene['id']}/maskfile?"
                             f"name={cpath.name}&t={int(time.time())}")

            # Disagreement map, pre-coloured RGBA so the client just draws it.
            # Hue = which way the vote went, alpha = how close to a tie (so one
            # over-segmenting backend can't flood the frame). See the palette
            # constants and the legend in the ensemble panel.
            heat = np.zeros((*hw, 4), np.uint8)
            frac = votes / max(n, 1)
            tie = 1.0 - np.abs(2.0 * frac - 1.0)          # 0 = unanimous, 1 = dead tie
            alpha = (CONTESTED_ALPHA_MIN
                     + (CONTESTED_ALPHA_MAX - CONTESTED_ALPHA_MIN) * tie ** 2)

            unanimous = votes == n
            leaning_water = (votes >= half) & ~unanimous
            leaning_dry = (votes > 0) & (votes < half)
            heat[unanimous] = (*AGREE_RGB, AGREE_ALPHA)
            heat[leaning_water] = (*MAJORITY_RGB, 0)
            heat[leaning_dry] = (*CONTESTED_RGB, 0)
            contested = leaning_water | leaning_dry
            heat[..., 3] = np.where(contested, alpha.astype(np.uint8), heat[..., 3])
            vpath = cdir / "_annotator_votes.png"
            Image.fromarray(heat, "RGBA").save(vpath, "PNG")
            votes_url = (f"/api/scene/{scene['id']}/maskfile?"
                         f"name={vpath.name}&fmt=asis&t={int(time.time())}")

        agreement = round(float(np.mean(ious)), 3) if ious else None
        max_pct = max((r.get("water_pct", 0) for r in results if "water_pct" in r), default=0)
        return self._json({
            "ok": True,
            "results": results,
            "iou_matrix": matrix,
            "agreement": agreement,                # mean pairwise IoU
            "disagreement_frac": disagreement_frac,
            "consensus_url": consensus_url,
            "consensus_water_pct": consensus_pct,
            "votes_url": votes_url,
            "n_ran": len(ok_names),
            "water_present": bool((consensus_pct or 0) > 0.5 or max_pct > 2.0),
        })

    # -- save (accept / edit / from-scratch) ---------------------
    def _handle_save(self, scene: dict, data: dict):
        app = self.app
        hw = _tiff_hw(scene["tiff"])

        if not data.get("mask_png_b64"):
            return self._err(400, "missing mask_png_b64")
        final = decode_mask_png(data["mask_png_b64"], hw)

        sess = data.get("session", {})
        seed_backend = sess.get("seed_backend")
        seed = None
        if seed_backend and re.fullmatch(r"[A-Za-z0-9._-]+", str(seed_backend)):
            cdir = app.cache_dir(scene["id"])
            for cand in (cdir / f"_annotator_auto_{seed_backend}.png",
                         cdir / str(seed_backend),          # e.g. _annotator_ensemble.png
                         Path(scene["dir"]) / str(seed_backend)):  # on-disk mask
                seed = read_mask_file(cand, hw)
                if seed is not None:
                    break

        who = clean_annotator(sess.get("annotator"), app.annotator)
        n_strokes = int(sess.get("n_strokes", 0))
        n_undos = int(sess.get("n_undos", 0))
        active_s = float(sess.get("active_seconds", 0))
        iou = rlf.mask_iou(seed, final)
        edited = rlf.edited_fraction(seed, final)
        final_pct = round(100 * float((final > 0).mean()), 3)
        auto_pct = round(100 * float((seed > 0).mean()), 3) if seed is not None else None
        route = rlf.classify_route(seed is not None, n_strokes, iou, (final > 0).any())

        with app.lock:
            from PIL import Image
            Image.fromarray(final, "L").save(Path(scene["dir"]) / "water_mask.png", "PNG")
            feats = app.features(scene)
            rlf.append_log(app.log_path, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "annotator": who,
                "scene_id": scene["id"], "scene_dir": scene["dir"],
                "route": route,
                "seed_backend": seed_backend or "",
                "backend_params": json.dumps(sess.get("backend_params", {})),
                "auto_water_pct": auto_pct if auto_pct is not None else "",
                "final_water_pct": final_pct,
                "auto_vs_final_iou": round(iou, 4) if iou is not None else "",
                "edited_pixel_frac": round(edited, 5) if edited is not None else "",
                "active_seconds": round(active_s, 1),
                "n_strokes": n_strokes, "n_undos": n_undos,
                "seg_mean_entropy": sess.get("seg_mean_entropy", ""),
                "seg_low_conf_frac": sess.get("seg_low_conf_frac", ""),
                "ensemble_agreement": sess.get("ensemble_agreement", ""),
                "ensemble_disagreement_frac": sess.get("ensemble_disagreement_frac", ""),
                "n_backends_run": sess.get("n_backends_run", ""),
                "features_json": json.dumps(feats),
            })
            app.state[scene["id"]] = {
                "review_status": "approved", "route": route,
                "final_water_pct": final_pct,
                "updated": datetime.now().strftime("%Y%m%d-%H%M%S"),
            }
            app._save_state()
            app.sync_manifest(scene, "approved", "water_mask.png", final_pct)

        return self._json({
            "ok": True, "route": route, "annotator": who,
            "auto_vs_final_iou": iou, "edited_pixel_frac": edited,
            "final_water_pct": final_pct,
            "next": self._next_scene_id(scene["id"]),
        })

    # -- review (reject / skip) ---------------------------------
    def _handle_review(self, scene: dict, data: dict):
        app = self.app
        status = data.get("status", "")
        if status not in ("rejected", "skip"):
            return self._err(400, "status must be 'rejected' or 'skip'")

        if status == "rejected":
            sess = data.get("session", {})
            with app.lock:
                feats = app.features(scene)
                rlf.append_log(app.log_path, {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "annotator": clean_annotator(sess.get("annotator"), app.annotator),
                    "scene_id": scene["id"], "scene_dir": scene["dir"],
                    "route": "rejected",
                    "seed_backend": sess.get("seed_backend", "") or "",
                    "backend_params": json.dumps(sess.get("backend_params", {})),
                    "auto_water_pct": "", "final_water_pct": "",
                    "auto_vs_final_iou": "", "edited_pixel_frac": "",
                    "active_seconds": round(float(sess.get("active_seconds", 0)), 1),
                    "n_strokes": int(sess.get("n_strokes", 0)),
                    "n_undos": int(sess.get("n_undos", 0)),
                    "seg_mean_entropy": sess.get("seg_mean_entropy", ""),
                    "seg_low_conf_frac": sess.get("seg_low_conf_frac", ""),
                    "features_json": json.dumps(feats),
                })
                app.state[scene["id"]] = {
                    "review_status": "rejected", "route": "rejected",
                    "updated": datetime.now().strftime("%Y%m%d-%H%M%S"),
                }
                app._save_state()
                app.sync_manifest(scene, "rejected", "", 0.0)

        return self._json({"ok": True, "next": self._next_scene_id(scene["id"])})

    def _next_scene_id(self, current: str) -> str | None:
        ids = [s["id"] for s in self.app.scenes]
        try:
            start = ids.index(current)
        except ValueError:
            start = -1
        for sid in ids[start + 1:] + ids[:start + 1]:
            if sid == current:
                continue
            if self.app.state.get(sid, {}).get("review_status", "pending") == "pending":
                return sid
        return None


# ===========================================================================
# main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every setting also lives in annotator_config.json; CLI flags win. "
               "See annotator_config.example.json.")
    ap.add_argument("--config", default=None,
                    help="Path to a JSON config file (default: ./annotator_config.json "
                         "or <this dir>/annotator_config.json if present)")
    ap.add_argument("--root", action="append", default=None, metavar="DIR",
                    help="Directory to scan for the 5-band TIFFs (repeatable). "
                         "Overrides config 'roots'. Relative paths resolve against the repo root.")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--annotator", default=None, help="Name recorded in label_log.csv")
    ap.add_argument("--device", default=None,
                    help="Compute device for torch/ultralytics backends + trainer: "
                         "auto (default) | cpu | cuda | cuda:0 | mps")
    ap.add_argument("--tinysam-python", default=None,
                    help="Python executable for the TinySAM subprocess (needs tinysam+timm+torch)")
    ap.add_argument("--segformer-onnx", default=None, help="Path to a 5-band SegFormer .onnx")
    ap.add_argument("--segformer-size", default=None, metavar="HxW",
                    help="Model input size, e.g. 512x512 (default: from the ONNX graph)")
    ap.add_argument("--segformer-water-class", type=int, default=None)
    ap.add_argument("--disable", action="append", default=[], metavar="BACKEND",
                    help="Turn a backend off entirely (repeatable): nir thermal tinysam segformer")
    ap.add_argument("--print-config", action="store_true",
                    help="Print the resolved config as JSON and exit")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                                # noqa: BLE001
        pass

    config, config_path = load_config(args.config)

    # --- CLI overrides (only when the flag was actually given) ---
    if args.root:
        config["roots"] = args.root
    for key, val in (("work_dir", args.work_dir), ("host", args.host),
                     ("annotator", args.annotator), ("device", args.device)):
        if val is not None:
            config[key] = val
    if args.port is not None:
        config["port"] = args.port
    bk = config["backends"]
    if args.tinysam_python is not None:
        bk["tinysam"]["python"] = args.tinysam_python
    if args.segformer_onnx is not None:
        bk["segformer"]["onnx"] = args.segformer_onnx
    if args.segformer_size is not None:
        bk["segformer"]["size"] = args.segformer_size
    if args.segformer_water_class is not None:
        bk["segformer"]["water_class"] = args.segformer_water_class
    for name in args.disable:
        if name in bk:
            bk[name]["enabled"] = False

    if args.print_config:
        print(json.dumps(config, indent=2))
        return

    work_dir = Path(config["work_dir"]).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)
    app = App(config, work_dir, config_path)
    Handler.app = app

    # Host/Origin pinning (see Handler._origin_ok). A wildcard bind has no
    # single name to pin, so it can only be left open — which is one more
    # reason the documented deployment is localhost + an SSH tunnel.
    bind = str(config["host"]).strip()
    if bind in ("", "0.0.0.0", "::", "*"):
        Handler.allowed_hosts = None
    else:
        Handler.allowed_hosts = {bind.lower(), "localhost", "127.0.0.1", "::1"}

    print(f"config file : {config_path or '(built-in defaults)'}")
    print(f"device      : {backends_mod.device_label(config.get('device', 'auto'))}")
    _hint = backends_mod.accel_hint()
    if _hint:
        print(f"              ! {_hint}")
    print(f"found {len(app.scenes)} scene(s) across {len(app.roots)} root(s):")
    for root, count in app.scenes_per_root().items():
        exists = "" if Path(root).exists() else "  (missing)"
        print(f"  {count:4d}  {root}{exists}")
    for b in app.registry.values():
        ok, reason = b.available()
        print(f"  backend {b.name:10s} {'ok' if ok else 'DISABLED: ' + reason}")
    disabled = [n for n in ("nir", "thermal", "tinysam", "segformer")
                if n not in app.registry]
    if disabled:
        print(f"  backends off by config: {', '.join(disabled)}")
    if Handler.allowed_hosts is None:
        print("\n  ! bound to a wildcard address — there is NO authentication and "
              "no Host pinning.\n    Anyone who can reach this port can read your "
              "imagery, overwrite masks and\n    start training jobs. Prefer "
              "--host 127.0.0.1 plus an SSH tunnel.")
    print(f"\n  http://{config['host']}:{config['port']}\n")

    ThreadingHTTPServer((config["host"], config["port"]), Handler).serve_forever()


if __name__ == "__main__":
    main()
