# UFONet Water Annotator — System Reference

**Version:** 1.0
**Last updated:** 2026-09-03
**Purpose:** Comprehensive reference for AI assistants and developers working on
`photo_processing/annotator/`. Read this before changing anything here.

---

## Table of Contents

1. [What this is](#1-what-this-is)
2. [Status](#2-status)
3. [Architecture](#3-architecture)
4. [Data flow](#4-data-flow)
5. [Configuration](#5-configuration)
6. [Invariants — don't break these](#6-invariants--dont-break-these)
7. [Gotchas discovered the hard way](#7-gotchas-discovered-the-hard-way)
8. [Already evaluated and rejected](#8-already-evaluated-and-rejected)
9. [How to test safely](#9-how-to-test-safely)
10. [Open items](#10-open-items)

---

## 1. What this is

A local web tool for producing 5-band water-segmentation training masks, with a
switchable set of auto-segmentation backends, an in-tool SegFormer fine-tune
loop, and a decision log that doubles as the dataset for a reinforcement-learning
**labeling-route policy** (accept auto mask / edit it / hand-label / reject).

It sits between two sibling scripts in this repo and changes neither:

```
pipeline.py → [ annotator ] → export_dataset.py → SegFormer
```

Co-registration happens upstream on the capture rig and is **not part of this
repo** — every tool here starts from an already-co-registered 5-band TIFF.

**Two audiences, one tool:** the human gets a labeler; the RL project gets
`work/label_log.csv`, where each row pairs a decision with its realised cost
(`active_seconds` = `h_i`) and quality (`auto_vs_final_iou` = `Q_i`).

Input scenes are co-registered 5-band GeoTIFFs, uint8, typically 1296×972:

| band (0-based) | content |
|---|---|
| 0,1,2 | R, G, B (optical) |
| 3 | Thermal / LWIR (FLIR Lepton, normalised 0–255) |
| 4 | NIR difference (NIR-ON minus NIR-OFF) |

---

## 2. Status

**Code-complete and tested. Not yet used for real** — as of this writing
`work/label_log.csv` does not exist; zero scenes have been labeled. The next
action is labeling, not coding. See [§10](#10-open-items).

Dependencies are declared in `pyproject.toml` (uv) and `requirements.txt` (pip),
core vs. optional extras. Use `uv sync && uv run server.py`; the examples below
assume that. Note the core dep is **`opencv-contrib-python-headless`**, not
plain opencv — `cv2.ximgproc.guidedFilter` (the `spectral` backend's default
boundary snap) exists only in the contrib build.

---

## 3. Architecture

| file | role |
|---|---|
| `server.py` | stdlib `ThreadingHTTPServer` app: config, scene discovery, band rendering, save/ensemble/train routes, `TrainManager` |
| `backends.py` | 8 auto-seg adapters behind one `Backend` interface + `build_registry()` + device resolution + `InteractiveSam` (click-to-segment) |
| `autolabel.py` | `spectral_water()` fusion engine and `derive_prompts()` — pure numpy/cv2, no torch |
| `trainer.py` | thin adapter over `../training/` keeping the Train panel's CLI + JSONL stable; ONNX export |
| `../training/` | the real training pipeline: modalities, band stats, augmentation, losses, metrics, resume, TTA, modality×model comparison. See `training/README.md` |
| `rl_features.py` | `scene_features()`, mask IoU / edit-fraction, `label_log.csv` writer |
| `../agreement.py` | offline inter-annotator agreement over `work/masks/` (IoU, kappa, boundary F1) |
| `frontend/` | `index.html` · `style.css` · `app.js` — canvas labeler + ensemble panel + train panel, zero JS dependencies |
| `LABELING_GUIDE.md` | class definitions — **read before labeling or changing class semantics** |

### Backends

Registry order matters: the UI dropdown defaults to the first *available* one.

| name | method | needs |
|---|---|---|
| `sam2` | spectral prior → points+box → ultralytics SAM 2.1-L | `sam2.1_l.pt` |
| `sam` | same prompting → FastSAM-s (fast pass) | `FastSAM-s.pt` |
| `spectral` | NIR-dark gate × (NDWI + low-texture + thermal-smoothness) + snow guard + horizon, then guided-filter/random-walker snap | nothing |
| `nir` | NIR below per-image threshold | nothing |
| `thermal` | Otsu(thermal) ∧ Otsu(NIR) | nothing |
| `change` | median dry-background from sibling captures → NIR-drop + NDWI-rise | ≥3 siblings |
| `tinysam` | subprocess to an **external** TinySAM driver script | a TinySAM checkout + driver script (neither ships here) + `tinysam`/`timm` |
| `segformer` | the model being trained; `.onnx` **or** an HF checkpoint dir | onnxruntime *or* transformers |

Every backend returns `BackendResult(mask uint8 {0,255} at native TIFF
resolution, elapsed_s, meta, error)`. A backend that can't run reports
`available() -> (False, reason)` and the UI greys it out — **it never crashes
the app.** Preserve that contract.

---

## 4. Data flow

```
scene dir/                      work/
  color_preserved_5_band.tiff     label_log.csv        one row per decision (RL dataset)
  water_mask.png   ← ONLY file    state.json           per-scene review status
                     we write     features/<id>.json   cached scene features
                                  cache/<id>/          transient auto/ensemble masks
                                  masks/<id>/<who>.png per-annotator mask (agreement)
                                  dataset/             --gold-only export target
                                  runs/<ts>/           best.pt, best_hf/, metrics.json, train.log
```

**Label → train loop:** label scenes → `/api/train/export` runs
`export_dataset.py --gold-only` (human-verified routes only, never falls back to
an unverified auto mask) → `/api/train/start` spawns `trainer.py` →
`/api/train/use` hot-swaps the `segformer` backend to the new checkpoint, so the
next scenes get seeded by the model you just trained.

---

## 5. Configuration

Three layers, increasing precedence: **built-in `DEFAULT_CONFIG` < JSON config
file < CLI flags**. Config file: `--config PATH`, else `./annotator_config.json`,
else next to `server.py`. `--print-config` dumps the resolved result.

Keys: `roots`, `tiff_names`, `exclude_dirs`, `max_depth`, `host`, `port`,
`annotator`, `work_dir`, `device`, `backends.<name>.{enabled,...}`.

`device` is `auto | cpu | cuda[:N] | mps | xpu`, plus aliases `amd`/`rocm`/`hip`
→ `cuda` and `intel` → `xpu`. It drives the torch/ultralytics backends *and* the
trainer. Per-backend override: `backends.<name>.device`.

---

## 6. Invariants — don't break these

1. **The only file the annotator writes into a scene directory is
   `water_mask.png`.** Everything transient goes to `work/cache/<scene-id>/`.
   Scene dirs are the user's real data archive, often reached through a symlink
   (see §7). (`pipeline.py`, a separate batch tool, additionally writes
   `water_mask_auto.png` / `water_mask_preview.jpg`; both are gitignored.)
2. **Scene IDs are `<parent>__<name>-<sha1[:6] of resolved path>`.** Stable
   across restarts and data moves. `state.json` is keyed by them; making IDs
   order-dependent desyncs the whole review state.
3. **A missing model/dep disables a backend, never crashes the app.**
4. **Gold ≠ silver.** `--gold-only` exports only `route ∈ {auto_accepted,
   auto_edited, interactive_clicks, manual_from_scratch}`. Never let a
   pseudo-label into that path. A new human-verified route must be added to
   `GOLD_ROUTES` or its scenes silently stop reaching training.
5. **An interactive SAM2 session posts its own seed.** There is no single
   auto-mask file to point at — the result is the union of candidates the
   human steered, composed client-side — so the client posts that composite
   as `seed_png_b64` and the save records route `interactive_clicks`. Without
   it the seed lookup fails (`:` is outside its charset, and candidates are
   written as `_annotator_click_<rank>.png`), the route degrades to
   `manual_from_scratch`, and `auto_vs_final_iou` / `edited_pixel_frac` are
   left empty — the RL dataset's whole point. `n_clicks` counts monotonically
   per scene for the same reason: `click.pts` is cleared on every tool switch
   and every "New object".
6. **`label_log.csv` is append-only** and is the RL dataset. Don't rewrite rows;
   add columns at the end of `LOG_HEADER` if you must extend it. The
   `annotator` value comes from the client (the header field, per-browser
   `localStorage`) and is therefore untrusted — always pass it through
   `clean_annotator()`, which falls back to the server's `--annotator`.
7. **Torch backends hold a `threading.Lock` around inference.** The server is
   threaded; a shared torch module is not reentrant.
8. **Click-to-segment caches ONE embedding, and the key must match.**
   `InteractiveSam` holds `(scene_id, on)`; `predict()` refuses when the key
   differs rather than decoding a click against the wrong scene's features.
   Encoding is ~15-20 s CPU and decoding ~70-150 ms — that ratio is the whole
   reason the feature is usable, so never re-encode per click.
   The Click tool is sticky across scenes: `loadScene()` re-runs
   `enterClickMode()` so the next scene starts encoding on load (the weights
   stay resident in `_pred`; only the embedding is per-scene). A prepare that
   returns after the user moved on is discarded client-side.
9. **A trained checkpoint carries its own preprocessing.** `training/` writes
   `best_hf/norm.json` (modality + per-band statistics) and the `segformer`
   backend reads it, so a served model is normalised exactly as it was
   trained. ONNX export copies it beside the `.onnx` as `<stem>.norm.json`.
   A checkpoint with no norm.json is legacy per-image min-max at 512x512;
   one with it runs at native resolution padded to /32, because that is what
   it was validated on. Never normalise in one place only — that is the bug
   `segformer_5band/PERFORMANCE.md` documented, and it is invisible in tests.
   For deployment the normalisation is compiled **into** the ONNX graph and
   the graph is stamped with metadata describing it; `SU-WaterCam`'s
   `segformer_preprocess.py` reads that rather than assuming min-max. Feeding
   a mean/std-trained model min-max input measured 99.7% of the frame as
   water, with no error raised.
10. **Splits are assigned per capture session, never per scene.** The rig
   fires repeatedly within a session — `20251229-1427`, `-14270`, `-1428`,
   `-1429` are the same view seconds apart. Splitting those individually puts
   near-duplicate frames on both sides and the val score measures
   memorisation: on the 13 labelled scenes a per-scene split scored 0.98 mIoU
   where a session-grouped split scored ~0.6. `training/manifest.py` groups by
   the scene's parent directory (`--group-by scene` opts out). Anything that
   reassigns splits must keep whole sessions together.
11. **Two mask encodings are in circulation.** The annotator writes
   `water_mask.png` as **0/255**; `export_dataset.py` converts to class
   indices **0/1**. Threshold masks at `> 0`, never `> 127` — at 127 every
   exported mask reads as all background: empty labels, no error, a model
   that predicts nothing. `training/engine.py` refuses to train on a split
   whose measured water fraction is 0.
12. **Painting repaints a dirty rect, not the frame.** `stamp()` marks the
   region it touched and maintains `S.waterCount` incrementally;
   `scheduleRender()` coalesces to one `putImageData` per animation frame over
   just that rect. Anything that replaces `S.mask` wholesale (undo, fill,
   clean, load, SAM) must call `renderMask()`, which recounts and repaints in
   full. Getting this wrong leaves stale pixels or a drifting water %.
13. **A replicate scene is served blind.** If `blind_for()` is true, the
   `/api/scene/<id>` response must not list `water_mask.png` and the sidebar
   must not flag `has_manual` — an anchored second opinion is not evidence.
14. **`#preview` is read-only.** The canvas stack is `#bg` (1) → `#preview` (2)
   → `#mask` (3). Hover-previews and the disagreement map draw to `#preview`
   and must never write `S.mask` — previewing has to stay non-destructive.
   `#preview` has `pointer-events: none` so it can't swallow brush strokes.

---

## 7. Gotchas discovered the hard way

- **ultralytics prompt shapes differ by model.** FastSAM wants *flat*
  `points=[[x,y],…]`, `labels=[1,0,…]`; SAM and SAM2 want one *nested* group
  `points=[[[x,y],…]]`, `labels=[[…]]`. Mixing them raises
  `Boolean value of Tensor with more than one value is ambiguous` or a size
  mismatch. Points-only sometimes returns **zero** masks — always pass a box too,
  and keep the box-only retry.
- **Python 3.14 uses `forkserver`**, so DataLoader workers must pickle the
  Dataset. It's a module-level class (`_SegDS`) for that reason — don't move it
  back inside a function. `num_workers=0` on CPU.
- **AMD ROCm PyTorch reports as `"cuda"`** (`torch.cuda.is_available() == True`,
  device string `"cuda"`). Detect AMD via `torch.version.hip`, not a separate
  device type. `accel_hint()` warns when a GPU exists but torch can't use it.
- **`rglob` does not follow symlinks.** Scene discovery uses
  `os.walk(followlinks=True)` with a realpath loop guard, so a `roots` entry
  (or anything under one) can be a symlink into a data archive elsewhere.
  `example_data/data` is the conventional spot for that symlink; it is
  gitignored.
- **Snow is not separable from water by the current spectral cues.** The
  NIR-difference band doesn't discriminate it and snow matches water on every
  appearance cue (bright, smooth, low-texture, cool). The `snow_guard` helps and
  is not a fix. This is physics, not a bug — treat `spectral` as a *seed*.
- **SAM2.1-L is not automatically better than FastSAM.** It's more *literal*
  about the prompt: a bad spectral prompt yields a precisely-wrong mask, where
  FastSAM's looseness sometimes rescues it. **The bottleneck is prompt quality,
  not model quality.**
- **`torch.onnx.export` needs `dynamo=False`** on torch 2.10 and the `onnx`
  package (not installed). The HF-checkpoint path needs neither — the
  `segformer` backend loads `best_hf/` directly.
- **Tracebacks from a running server can be nonsense** if you edited the file
  after it started — Python prints the *new* source against *old* line numbers.
  Only the last line is trustworthy. Restart before debugging.
- **The disagreement map needs tie-weighted alpha, not flat colours.** First
  version painted every non-unanimous pixel at full opacity; because `nir`
  over-segments (it called a whole white sky water), 99.8% of the frame lit up
  and the map was useless. Alpha now scales with `1 - |2·frac - 1|` squared, so
  a lone dissenter fades and near-even splits dominate. **If you change the
  palette, change `style.css`'s `.sw.*` legend swatches too** — they're hand-
  mirrored from the `*_RGB` constants in `server.py`.
- **ultralytics numpy input is BGR.** Its loader documents numpy arrays as
  OpenCV-order BGR and `preprocess` does `im[..., ::-1]`, so handing it an RGB
  array silently swaps red and blue before the model sees it. `sam_prompt_image()`
  is the single place that builds SAM's 3-channel input; keep it BGR.
- **A SAM2 box-only prompt returns an empty stack, not None.** `res[0].masks`
  is present with shape `(0, H, W)`, so checking `masks is None` isn't enough —
  check `shape[0] == 0` too, or you return "ok" with no mask.
- **`transformers` 5.x moved SegFormer's module paths.** The stem is now
  `segformer.stages.0.patch_embeddings.proj`, not
  `segformer.encoder.patch_embeddings[0].proj`. The old 5-band warm-start
  looked that path up inside a `try/except` and so silently degraded to a
  random stem while still logging "pretrained". `training/models/base.py`
  locates the stem structurally — the first `Conv2d` in `named_modules()`
  order, verified for both SegFormer and Mask2Former — and *raises* if it
  cannot find one.
- **ONNX export needs opset >= 14, and 13 was hard-coded.** SegFormer
  attention in `transformers` 5.x lowers to
  `aten::scaled_dot_product_attention`, unsupported at 13; the export died
  before it ever reached the missing-`onnx`-package check. Now opset 17.
- **The thermal band is warped, the optical band is only resized.** TIFF bands
  0-2 match `*-NIR-OFF.jpg` downscaled 2x to MAE 0.00, but band 3 differs from
  the raw `IMG_*.pgm` upsampled by MAE 12.4. So a thermal modality must read
  the TIFF band (the `.pgm` is in the Lepton's own 160x120 geometry and does
  not line up with the mask), while `rgb_nofilt` may legitimately downscale
  `*-NIR-ON.jpg` onto the grid. Re-check with
  `training.modalities.verify_provenance()` on new capture sessions — that
  second fact stops being true the moment co-registration starts warping the
  optical frame.
- **Mask2Former does not take a pixel loss.** It is a mask classifier trained
  by Hungarian matching and computes its own objective, so `--loss ce+lovasz`
  cannot apply to it. `supports_pixel_loss` says so and the run metadata
  records `pixel_loss_applies`, rather than accepting a flag that does nothing.
- **Always look at a rendered overlay before believing it works.** The bug
  above passed every numeric check; only compositing it over the photo showed
  the problem.

---

## 8. Already evaluated and rejected

Don't re-propose these without new information:

- **Unsloth / vLLM for fine-tuning** — LLM-only. Unsloth is LoRA for text
  transformers; vLLM is an inference server. Neither supports vision/dense
  prediction. HF `transformers` is the path.
- **An mmseg-based trainer** — `mmcv` pins hard to old torch/py versions and
  brings a second config system for no gain here. HF `transformers` covers
  SegFormer and is what the ONNX export path already expects. `../training/`
  is that path, and it ports the *ideas* from `segformer_5band` (band-statistic
  normalisation, Lovász loss, class weights, TTA, the four-modality
  comparison) without `mmcv`. The old tree still exists at
  `../../segformer_5band` for reference; it does not build against current
  torch (`build_segmentor` fails on `mit_b2: unexpected keyword 'style'`).
- **Satellite/SAR flood models** (Sen1Floods11, WorldFloods, ETCI-2021) — nadir,
  10 m GSD, wrong physics. Zero transfer to an oblique ground-level camera.
- **The Ryzen AI NPU** — Linux stack is ONNX/INT8-only with a Windows-centric
  toolchain. Not a practical path for SegFormer.
- **Raw SAM masks as training labels without human verification** — SAM is
  class-agnostic; it segments what you point at, with no concept of "water".
  See the Tier 1/2/3 scheme in `README.md`.

---

## 9. How to test safely

**Never run tests against a real capture archive.** Scene dirs are somebody's
primary data and the annotator writes `water_mask.png` into them. Point
`--root` at the bundled `example_data/`, or copy a handful of scene dirs to a
scratch directory first:

```bash
SP=/tmp/<scratch>
mkdir -p $SP/caps/s1 && cp "<a scene>/color_preserved_5_band.tiff" $SP/caps/s1/
uv run server.py --root $SP/caps --work-dir $SP/w --port 8799
```

Checks that catch most regressions:

```bash
uv run python -m py_compile *.py ../training/*.py ../training/models/*.py
node --check frontend/app.js
uv run server.py --print-config
# the training pipeline, end to end on the bundled scenes (no archive needed):
cd .. && uv run --project annotator python -c "
from pathlib import Path
from training.modalities import verify_provenance
print(verify_provenance(Path('example_data/Brooklyn')))"   # band provenance still holds
# every id in app.js exists in index.html:
python - <<'EOF'
import re; js=open('frontend/app.js').read(); html=open('frontend/index.html').read()
print([i for i in set(re.findall(r'\$\("#([\w-]+)"\)', js)) if f'id="{i}"' not in html] or "ok")
EOF
```

Afterwards: remove scratch dirs, `rm -rf __pycache__`, confirm no test server is
still listening, and verify no `_annotator_*` or stray `water_mask.png` landed in
the archive.

---

## 10. Open items

**Blocking real use:** nothing in the code. Label ~30–50 scenes plus a
hand-labelled val set of ~20–40. Confirm the `[DECIDE]` rows in
`LABELING_GUIDE.md` first.

**Optional installs:** `pip install onnx` (Pi deployment only), `pip install
timm` (the `tinysam` backend, largely superseded by `sam`/`sam2`), ROCm torch in
a separate py3.12 venv for an AMD iGPU (README → "Integrated AMD Radeon").

**Deferred improvements**, roughly by value:

1. ~~Interactive click-to-prompt for SAM~~ — **done.** `InteractiveSam` +
   the Click (SAM2) tool; prompt points draw on `#preview`.
2. ~~Dirty-rect mask rendering~~ — **done.** Dirty rect + rAF coalescing +
   incremental water count (~14,000x less per-stroke work); hover previews are
   cached as `ImageBitmap`s.
3. Undo as dirty-rect deltas — `snapshot()` still copies the whole 1.26 MB
   mask per stroke (~38 MB of history per scene). The bookkeeping now exists
   to store `{x, y, w, h, before}` instead.
4. ~~Trainer `--resume` from `last.pt`~~ — **done.** `training.train --resume`
   restores model, optimizer, scheduler, scaler, epoch, best and the
   early-stop counter from `ckpt.pt`, which is written *after* each eval so
   `best` is never one eval stale.
5. Unit tests (`_slug`, `_deep_merge`/config precedence, `classify_route`,
   `hash_split`, `spectral_water` on a synthetic array).
6. Incremental `train.log` parsing (`_events()` re-reads the whole file each poll).
7. `features()` computed outside `app.lock` (~1 s under the global lock per save).
8. `build_model` loads `nvidia/mit-b0` twice (model + stem warm-start source).
9. `--token` auth — `--host 0.0.0.0` still has none. A non-wildcard bind pins
   `Host`/`Origin` (see `Handler._origin_ok`), which blocks cross-origin drive-by
   requests and DNS rebinding but is **not** authentication. Localhost + SSH
   tunnel is the documented deployment and the README warns about this in a
   callout; if that warning ever stops being true, add real auth first.
