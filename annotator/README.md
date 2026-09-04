# UFONet Water Annotator

A local web tool for producing SegFormer training masks from co-registered
5-band scenes. Manual painting **plus** a switchable set of auto-segmentation
backends, and every labeling decision is logged as a row for the
labeling-route RL dataset.

```
   (capture rig)  ──►  pipeline.py  ──►  annotator (this)  ──►  export_dataset.py
 co-registration        auto-label          review / label         MMSeg format
   — not in this repo
```

**Co-registration is out of scope here.** These tools start from 5-band TIFFs
that a capture rig has already co-registered; nothing in this repo aligns raw
frames.

> **Before you label:** read [`LABELING_GUIDE.md`](LABELING_GUIDE.md) — the class
> definitions (is ice water? is wet pavement?) have `[DECIDE]` rows awaiting
> sign-off, and consistency there matters more than any model choice.
> **Before you change code:** read [`CLAUDE.md`](CLAUDE.md) — architecture,
> invariants, and the gotchas that cost time to find.

It writes `water_mask.png` next to each 5-band TIFF, which is exactly the file
`export_dataset.py` prefers, so it slots into the existing workflow without
changing anything upstream or downstream. Transient auto/ensemble masks live in
`work/cache/<scene-id>/` — the scene's own directory only ever gets
`water_mask.png`. (Older builds wrote `_annotator_*.png` next to the TIFF; those
are now inert and safe to delete: `find <roots> -name '_annotator_*.png' -delete`.)

Scene IDs are `<parent>__<name>-<6 hex of the resolved path>` — stable across
restarts and data-directory moves, so `state.json` (the sidebar chips) doesn't
desync when the archive is reorganised.

**Tier-1 gold export:** `python ../export_dataset.py --gold-only <dataset_dir>
[--label-log work/label_log.csv]` builds an MMSeg dataset from **only the
human-verified** scenes — `route` in `{auto_accepted, auto_edited,
manual_from_scratch}` — never falling back to an unverified auto mask.
`rejected` and unlabelled scenes are excluded. It also writes
`dataset_provenance.csv` (route / seed_backend / auto-vs-final IoU / edit
fraction / active seconds / annotator / timestamp per scene) and splits
train/val by a stable hash of the scene name so the split doesn't reshuffle
as labels accumulate.

## Install

With [uv](https://docs.astral.sh/uv/) (recommended — `pyproject.toml` is set up
for it):

```bash
cd photo_processing/annotator
uv sync                        # everything you normally want (~87 packages)
uv run server.py --annotator <your-name>
```

`uv sync` installs the `sam`, `train` and `onnx` dependency groups by default,
because the SAM backends are the recommended default in the UI and leaving them
out ships a deliberately worse tool. Variants:

```bash
uv sync --no-default-groups    # minimal (~16 pkgs): labeler + spectral/nir/thermal/change
uv sync --group export         # + ONNX export for deployment
uv sync --group tinysam        # + the TinySAM backend
```

Or with pip into any environment: `pip install -r requirements.txt`.

> **opencv-contrib, not plain opencv.** `cv2.ximgproc.guidedFilter` is the
> `spectral` backend's default boundary snap and ships *only* in the contrib
> build. With plain `opencv-python-headless` installed it degrades to no
> refinement at all — the backend reports `refine: "none (guided needs
> opencv-contrib-python-headless)"` in its metadata rather than pretending
> otherwise, but you'll get softer masks.

### First run

With no config file at all the server scans the bundled `example_data/`, so
`uv run server.py` gives you three real scenes to click through immediately.
Point it at your own captures with `--root /path/to/captures` (repeatable), or
copy `annotator_config.example.json` to `annotator_config.json` and set `roots`.

### Which torch build you get — read this before syncing

`uv sync` pulls torch from PyPI, and **on Linux the PyPI default is the CUDA
build**. A measured `uv sync` on one Linux test machine produced a **6.2 GB `.venv`**
with `torch 2.14.0+cu130` — roughly 2.5 GB of which is NVIDIA GPU kernels that
**cannot run on a non-NVIDIA machine**. It works fine (torch silently falls
back to CPU), it's just a lot of disk for nothing.

Pick the build that matches your hardware by uncommenting one
`[[tool.uv.index]]` block in `pyproject.toml` — all three are pre-written —
then re-run `uv sync`:

| your GPU | block to uncomment | notes |
|---|---|---|
| **None / AMD / Intel**, labeling only | `pytorch-cpu` | far smaller; identical results, training just runs slower |
| **NVIDIA** | `pytorch-cu124` (or keep the default) | fp16 AMP training, fast SAM2 |
| **AMD, want GPU** | `pytorch-rocm` | ROCm has **no Python 3.14 wheels yet** → `uv python pin 3.12` first. See **GPU acceleration** and the Radeon 780M recipe below. |

The repo ships with all three commented out so the default works anywhere.
If you're only labeling and never training, `pytorch-cpu` — or even
`uv sync --no-default-groups`, which skips torch entirely — is the sensible
choice.

If a backend shows as DISABLED in the startup banner, the reason names the fix
— usually a missing group, e.g. `` run `uv sync --group sam` ``.

## Run

```bash
uv run server.py --annotator <your-name>
# open http://127.0.0.1:8000
```

### Who gets credit for a label

Every `work/label_log.csv` row carries an `annotator` name, so a shared dataset
stays attributable and you can measure per-person labeling time and agreement.

The **“Labeling as” field in the header** is the one that counts — it's sent
with every save and reject, and it's stored per browser (`localStorage`), so
two people can share one running server and each be recorded correctly. Change
it when someone else takes over; no restart needed. Names are trimmed,
stripped of control characters, and capped at 40 characters.

`--annotator <name>` (or `"annotator"` in `annotator_config.json`) sets the
**default** that pre-fills that field and is used if it's left blank. It
defaults to `anon` — set one of them.

> ### ⚠️ There is no authentication
> The server is designed for **localhost only**. Anyone who can reach the port
> can read your imagery, overwrite masks, and start training subprocesses —
> there is no login, no token, no access control of any kind.
>
> Keep the default `--host 127.0.0.1`. To label on a remote machine, forward
> the port over SSH rather than exposing it:
>
> ```bash
> ssh -L 8000:127.0.0.1:8000 <host>
> ```
>
> **Do not run `--host 0.0.0.0`** on an untrusted network, or on any machine
> reachable from the internet.
>
> On a non-wildcard bind the server does pin the `Host` and `Origin` headers to
> the address it bound (anything else gets a 403). That is *not* authentication
> — it only stops a web page you happen to visit from driving your annotator
> across origins, and stops DNS rebinding from reaching it. A wildcard bind has
> no name to pin, so it disables the check and prints a warning at startup.

If a scanned root has a `manifest.csv` (written by `pipeline.py`), the tool
keeps its `review_status` / `mask_path` columns in sync.

## Creating a mask — walkthrough

A "mask" is a black-and-white picture the same size as the photo: **white where
there is water, black everywhere else.** That's the whole job. The model learns
from thousands of these, so consistency matters more than pixel perfection.

First, **read [`LABELING_GUIDE.md`](LABELING_GUIDE.md)** — it decides the calls
you'll hit constantly (is ice water? is wet pavement?). Being consistent with
it beats being careful without it.

### 1 · Pick a scene

The left sidebar lists every scene found. The chip on the right of each row is
its state: `•` pending, `appr` done, `reje` rejected. Click one to open it.
The header shows overall progress (`12 / 286 reviewed`).

Set **Labeling as** in the top-right to your name before you start — it's
recorded with every label you save.

### 2 · Look at the photo properly

**View** changes how the photo is drawn — it never changes your mask.

| view | when to use |
|---|---|
| **Colour-IR (water looks dark)** | the default; start here |
| **Normal colour** | to sanity-check what you're actually looking at |
| **Near-infrared** | water absorbs it → water is dark, snow/pavement bright |
| **Thermal (heat)** | at night, or when optical is useless |
| **Water index (NDWI)** | a second opinion on ambiguous surfaces |

**Zoom** magnifies; hold **Space** and drag to pan. Flip between views on a
confusing scene before deciding — that's usually faster than painting.

### 3 · Get a starting mask

You rarely paint from scratch. Two routes:

**Fast:** pick an auto-segmenter in **Auto-segment** and press **Run**. Its
result loads straight into the editor, ready to correct.

**Better:** press **Run all**. Every available auto-segmenter runs and you get
a comparison panel:

- a plain-language verdict — *"Water appears to be present. The auto-segmenters
  mostly agree…"*
- one row per segmenter with its water %, runtime and a colour swatch.
  **Hover a row to preview it on the photo** — this is non-destructive, it
  never touches your mask. Compare them, then click **Use this** on the best
  one to load it for editing.
- a **consensus (majority vote)** row — usually the best starting point
- **Show disagreement map** — shades the photo by how the segmenters voted:
  hue = which way it went (teal *all agree*, amber *most say water*, magenta
  *a few say water*), and **opacity = how close to a tie**. Faint areas are
  settled; **strongly shaded areas are where they genuinely disagree** — those
  are the pixels worth your attention.

If the verdict says they strongly disagree, treat every result as a rough
draft. If nothing looks close, just **Clear all** and paint by hand.

### 4 · Correct it

| tool | what it does |
|---|---|
| **Brush** (`B`) | paint water. `[` and `]` change the size |
| **Erase** (`E`) | remove water painted by mistake |
| **Fill** (`F`) | click once to flood-fill everything of similar brightness — grabs a whole pond in one go. **Fill spread** controls how far it reaches |
| **Clean up** | despeckle, close small gaps, fill interior holes — good final pass |
| **Undo** (`U`) / **Redo** (`R`) | last 30 changes |
| **Clear all** | wipe the mask and start over |

Drag **Mask opacity** down to check your edges against the photo underneath.
The **Water %** readout in the footer updates live — a wild number is a good
hint you've mis-painted something.

Don't pixel-polish the waterline. A few pixels of slop is expected and the
training pipeline can ignore a boundary ring; more labeled scenes beat one
perfect scene.

### 5 · Finish the scene

| button | when | effect |
|---|---|---|
| **Save & Next** (`Enter`) | the mask is right | writes `water_mask.png` next to the photo, logs the decision, opens the next pending scene |
| **Reject scene** | blurry, misaligned, or **you genuinely can't tell** | logged and excluded from training |
| **Skip** | come back later | nothing recorded |

**Reject beats guessing.** A wrong label corrupts both the model and the RL
reward signal; a missing one costs nothing.

That's the loop. Once you have ~30–50 scenes, open **⚙ Train**, and the model
you train there becomes another auto-segmenter — so the next scenes arrive
pre-labeled and the job gets faster.

**Keys:** `B` brush · `E` erase · `F` fill · `[` `]` size · `U` undo ·
`R` redo · hold `Space` to pan · `Enter` save.

## What each backend does

| backend | method | needs | UI params |
|---|---|---|---|
| `spectral` | **recommended prior.** NIR-dark *gate* × (NDWI + low-texture + thermal-smoothness) support, + bright-achromatic **snow guard** + horizon cut, then guided-filter / random-walker boundary snap. `autolabel.spectral_water()`. | nothing | `refine`, `thresh`, `snow_guard`, `horizon`, `specular` |
| `nir` | NIR-difference below a per-image threshold — the single-cue baseline | nothing | `threshold`, `open_iter` |
| `thermal` | Otsu(thermal) ∧ Otsu(NIR) + morphology — the night / low-light fallback | nothing | — |
| `change` | median **dry-background** from sibling captures under the same parent dir → NIR-drop + NDWI-rise. Fixed-camera change detection. | ≥3 sibling scene dirs | `k_refs`, `min_refs` |
| `sam` | fast interactive pass. spectral prior → pos/neg points + box (or a lower-centre geometry fallback when the prior is weak) → **ultralytics FastSAM-s** on RGB or NIR false-colour. ~0.5–2 s/scene CPU. | `ultralytics` (installed) + a `.pt` (`backends.sam.weights`, or `allow_download: true`) | `on`, `n_pos`, `n_neg`, `use_box` |
| `sam2` | high-quality pass, same prompting → **ultralytics SAM 2.1-L**. Crisper boundaries, but *more literal about the prompt* (a bad spectral prompt → a precisely-wrong mask, where FastSAM's looseness sometimes rescues it). ~10–25 s/scene CPU. Best when a human places the prompt, or for an offline batch pass. | `ultralytics` + `sam2.1_l.pt` (`backends.sam2.weights`, or `allow_download: true`) | `on`, `n_pos`, `n_neg`, `use_box` |
| `tinysam` | auto-picked points → TinySAM, as a subprocess | **external**: a TinySAM checkout + a driver script (neither ships here) + a Python with `tinysam`+`timm`; unconfigured it just shows as unavailable | `n_points` |
| `segformer` | 5-band SegFormer inference from a **`.onnx`** (onnxruntime) **or an HF checkpoint dir** (`best_hf/`, via transformers); logs mean predictive entropy + low-confidence fraction for model-in-the-loop active learning. Fine-tune one from the ⚙ Train panel. | `onnxruntime` for .onnx, or `transformers`+`torch` for an HF dir | — |

Setup notes for the model-based backends are in **Configuration → Which
segmentation backends are available** below.

**Known limitation of the pure-spectral backends** (`spectral`, `nir`,
`thermal`, `change`): in **snow scenes** the NIR-difference band does *not*
separate snow from water, and snow matches water on every appearance cue.
`spectral`'s snow guard helps but won't fully solve it — treat these as a
seed, lean on `sam` + human correction, and let `segformer` take over once
it has ~30–50 labels. The **ensemble disagreement** score exists precisely
to flag these scenes.

## Fine-tune SegFormer (⚙ Train panel)

The **⚙ Train** button opens a panel that runs the whole loop:

1. **Export gold dataset** → `export_dataset.py --gold-only` into `work/dataset/`.
2. Set epochs / lr / img-size / batch, **Start fine-tune** → `trainer.py`
   runs as a subprocess (in the server's own Python, i.e. whatever
   interpreter you launched `server.py` with), streaming
   JSONL. The panel shows live step/loss/lr and per-epoch mIoU + water-IoU; the
   run keeps going if you close the panel.
3. When done: **Use as the `segformer` backend** hot-swaps the auto-seg
   `segformer` backend to the fine-tuned checkpoint (closing the
   model-in-the-loop active-learning loop), or **Export ONNX** for Pi deployment.
   After a server restart the panel re-attaches to the newest `work/runs/*`, so
   Use / Export-ONNX still work on the last run.

`trainer.py` builds a HuggingFace `SegformerForSemanticSegmentation` with
`num_channels=5`, warm-starts the 5-band stem conv from the RGB `nvidia/mit-b0`
weights (RGB→ch 0-2, mean(RGB)→ch 3-4), trains with per-band min-max
normalisation matching `segformer_preprocess`. Outputs per run in
`work/runs/<ts>/`: `best.pt` / `last.pt`, `best_hf/` (HF dir), `metrics.json`,
`train.log`. The `segformer` backend loads `best_hf/` **directly via
transformers** — no ONNX needed for the in-tool loop; `_autofind` also picks
the newest `work/runs/*/best_hf` on startup.

**Caveats:** on CPU, training is slow (~1 min/epoch on a small set); on a CUDA
GPU it uses fp16 AMP and is many times faster (see **GPU acceleration**).
ONNX export needs `pip install onnx` (the HF-dir path doesn't). The only
trustworthy mIoU is on a **hand-labelled** val set — the auto-split val masks
are as noisy as the train masks. LLM tools (Unsloth, vLLM) do **not** apply —
they're for text models / LLM serving, not vision segmentation.

## Configuration

Everything is configurable, from three layers — **built-in defaults  <  a JSON
config file  <  CLI flags**.

**Config file** — `--config PATH`, else `./annotator_config.json`, else
`annotator_config.json` next to `server.py`. Copy
`annotator_config.example.json` and edit. Any key may be omitted.
`uv run server.py --print-config` shows the fully resolved config
and exits. The startup banner and the `/api/config` endpoint (and the header
tooltip in the UI) report which file was used, the resolved roots, and the
scene count per root.

### Where photos load from

`discover_scenes()` walks each entry of **`roots`** looking for files named in
**`tiff_names`** (first name wins when a directory has more than one), skipping
any path component in **`exclude_dirs`**, no deeper than **`max_depth`**
levels, one scene per directory.

| key | default | meaning |
|---|---|---|
| `roots` | `["example_data"]` | dirs to scan; relative paths resolve against the repo root (`photo_processing/`), `~` expands. The default is the bundled sample data, so a fresh clone works unconfigured. CLI: `--root DIR` (repeatable) replaces the list. **Symlinked directories inside a root are followed** (`os.walk(followlinks=True)`, with a realpath loop guard) — so a `roots` entry can be a symlink to a data archive elsewhere. |
| `tiff_names` | `["color_preserved_5_band.tiff", "final_5_band.tiff"]` | filenames that mark a scene |
| `exclude_dirs` | `.git .venv venv node_modules site-packages __pycache__` | path components to skip |
| `max_depth` | `6` | max directory depth below a root |

### GPU acceleration (NVIDIA, AMD, Intel, Apple)

**`device`** (default `"auto"`; CLI `--device`): the compute device for the
torch / ultralytics backends (`sam`, `sam2`, `segformer`) **and** the fine-tune
trainer. `"auto"` = CUDA/ROCm → Intel `xpu` → Apple `mps` → `cpu`. Force a
specific one with `cpu` / `cuda` / `cuda:0` / `mps` / `xpu`, or the aliases
`amd`/`rocm` (→ `cuda`) and `intel` (→ `xpu`). Override one backend with
`backends.<name>.device`. The resolved device — with the GPU name and, on AMD,
the ROCm version — shows in the startup banner, `/api/config`, the header, and
the Train panel.

| vendor | torch | ONNX Runtime | notes |
|---|---|---|---|
| **NVIDIA** | stock CUDA wheels | `onnxruntime-gpu` → `CUDAExecutionProvider` | fp16 AMP in the trainer |
| **AMD (Linux)** | a **ROCm** build of torch — `pip install torch --index-url https://download.pytorch.org/whl/rocm6.x` | `onnxruntime-rocm` → `ROCM` / `MIGraphX` EP | ROCm torch reports as `"cuda"`, so nothing else changes; fp16 AMP works |
| **AMD / Intel (Windows)** | CPU only (no mainline torch GPU path) | `onnxruntime-directml` → `DmlExecutionProvider` | `segformer` ONNX inference is GPU-accelerated; SAM & fine-tune stay on CPU |
| **Intel GPU (Linux)** | Intel Extension for PyTorch (`torch.xpu`) | `onnxruntime-openvino` | trainer runs fp32 on xpu |
| **Apple** | stock wheels (`mps`) | `CoreMLExecutionProvider` | |

- **SAM / SAM2**: ultralytics runs on whatever `device` resolves to — SAM2.1-L
  drops from ~15 s to well under a second on any GPU.
- **Fine-tune**: on CUDA **or ROCm** the trainer turns on **fp16 mixed
  precision** (`torch.autocast` + `GradScaler`), `pin_memory`, 2 DataLoader
  workers; the Train panel bumps the default batch to 8. `--no-amp` disables it.
- **`segformer` ONNX inference**: `_session()` scans
  `onnxruntime.get_available_providers()` and uses the first GPU provider it
  finds (`CUDA` / `ROCM` / `MIGraphX` / `DML` / `OpenVINO` / `CoreML`), else
  CPU — so the right `onnxruntime-*` wheel is all that's needed. The
  HF-checkpoint path uses the GPU directly via torch, no extra install.

If a GPU is physically present but the installed torch can't use it (e.g. a
CUDA-build torch on an AMD box), the startup banner and `/api/config`
`device_hint` say so and the app stays on CPU rather than failing.

#### Integrated AMD Radeon (RDNA3 iGPU, e.g. Radeon 780M / `gfx1103`)

Worth wiring up for **`sam2`** (SAM2.1-L: ~15–25 s → ~3–6 s) and
**fine-tuning** (~2–3× vs a 16-thread CPU); `sam` / `spectral` / `nir` /
`thermal` / `change` see no benefit. VRAM is not the constraint if the UEFI
allocates the iGPU a few GB — SegFormer-B0 512² fp16 and SAM2.1-L both fit
with room to spare.

A default PyPI torch is a **CUDA build** and will never see an AMD iGPU, so
ROCm torch goes in its own virtualenv. One-time setup (paths below are just
suggestions — nothing depends on them):

```bash
sudo dnf install rocm-hip rocminfo rocm-opencl        # Fedora 41+ ships ROCm 6.x
sudo usermod -aG render,video "$USER"                 # then re-login
python3.12 -m venv ~/.venv-rocm                       # ROCm torch has no py3.14 wheels yet
~/.venv-rocm/bin/pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch
~/.venv-rocm/bin/pip install rasterio opencv-contrib-python-headless scipy pillow \
    scikit-image transformers accelerate ultralytics onnxruntime
export HSA_OVERRIDE_GFX_VERSION=11.0.0                # gfx1103 isn't officially supported; pass as gfx1100
rocminfo | grep -E 'gfx|Marketing'                    # sanity check
```

Then launch the annotator with that interpreter — `device: auto` picks up the
iGPU (ROCm torch reports as `"cuda"`), fp16 AMP turns on, and the Train panel
raises the default batch:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 ~/.venv-rocm/bin/python server.py --annotator <you>
```

Caveats: `gfx1103` is unofficial — most ops work under the override, a few may
fault; if so, fall back to `--device cpu` for that task. The Ryzen **NPU**
(`amdxdna`, `/dev/accel0`) is not a practical target — the Linux Ryzen AI stack
is ONNX-/INT8-only with a Windows-centric toolchain.

### Which segmentation backends are available

Each entry of **`backends`** has `"enabled": true|false` (a disabled backend is
left out of the registry entirely — it never reaches the UI) plus
backend-specific settings. An enabled backend still runs its own `available()`
check and shows in the UI greyed-out with the reason on hover if its
model / weights / interpreter are missing.

| backend | `enabled` check → shown | further `available()` check |
|---|---|---|
| `nir` | config only | none — always usable |
| `thermal` | config only | none — always usable |
| `sam` / `sam2` | config | `ultralytics` imports **and** a checkpoint resolves (`backends.<name>.weights`, cwd- then repo-root-relative; or `allow_download: true`). `sam` defaults to FastSAM, `sam2` to SAM 2.1-L. |
| `tinysam` | config | `script` + `weights` files exist **and** `python -c "import tinysam"` succeeds in `python` (probed once, cached) |
| `segformer` | config | a model resolves — `backends.segformer.onnx`, else auto-search of this repo only (`work/runs/*/best_hf`, then `annotator/weights/*.onnx`) — **and** `onnxruntime` (for `.onnx`) or `transformers`+`torch` (for an HF dir) importable |

```jsonc
"backends": {
  "spectral": { "enabled": true },
  "nir":     { "enabled": true },
  "thermal": { "enabled": true },
  "change":  { "enabled": true },
  "sam": {
    "enabled": true,
    "weights": "photo_processing/annotator/weights/FastSAM-s.pt",  // .pt path; rel = cwd then repo root
    "model": "fastsam",                       // fastsam | sam | sam2
    "allow_download": false                   // true = let ultralytics fetch weights on first use
  },
  "sam2": {
    "enabled": true,
    "weights": "photo_processing/annotator/weights/sam2.1_l.pt",
    "model": "sam2",                          // uses ultralytics SAM(); sam2.1_l is the best available
    "allow_download": false
  },
  "tinysam": {
    "enabled": true,
    "python": "/path/to/venv/bin/python",   // env with tinysam+timm+torch; null = this interpreter
    "tinysam_path": "/path/to/TinySAM",     // your TinySAM checkout (has weights/tinysam.pth)
    "weights": null,                         // override weights path; null = <tinysam_path>/weights/tinysam.pth
    "script": "/path/to/tinysam_water.py"   // the driver script; required — none ships here
  },
  "segformer": {
    "enabled": true,
    "onnx": "/path/to/segformer_5band.onnx", // null = auto-search
    "size": null,                            // "512x512" or [512,512]; null = read from the ONNX graph
    "water_class": 1
  }
}
```

CLI shortcuts: `--device auto|cpu|cuda|rocm|mps|xpu`, `--disable nir` (repeatable), `--tinysam-python`,
`--segformer-onnx`, `--segformer-size HxW`, `--segformer-water-class`.

## The RL decision log — `work/label_log.csv`

One row per Save or Reject. Columns that matter for the routing reward
(`r_i = α·u·Q − β·cost − γ·u·(1−Q)`):

| column | role |
|---|---|
| `route` | the action: `auto_accepted` / `auto_edited` / `manual_from_scratch` / `rejected` |
| `seed_backend`, `backend_params` | which auto-segmenter was offered |
| `auto_vs_final_iou` | **Q_i** — auto-label quality vs. the accepted mask |
| `edited_pixel_frac` | how much of the auto mask the human had to change |
| `active_seconds` | **h_i** — human cost (idle gaps > 12 s don't count) |
| `n_strokes`, `n_undos` | effort proxies |
| `seg_mean_entropy`, `seg_low_conf_frac` | SegFormer uncertainty (when it was the seed) |
| `ensemble_agreement` | mean pairwise IoU across backends on this scene (when **Run all** was used) |
| `ensemble_disagreement_frac` | fraction of pixels the backends split on — high ⇒ hard scene ⇒ route to human |
| `n_backends_run` | how many backends the ensemble ran |
| `features_json` | per-scene band / NDWI / texture descriptors = the policy's state |

`work/features/<scene>.json` caches the features; `work/state.json` holds
per-scene review status.

## Files

```
CLAUDE.md                      architecture, invariants, gotchas — read before changing anything
LABELING_GUIDE.md              class definitions — read before labeling
pyproject.toml                 uv dependencies: core + sam/train/onnx/export/tinysam groups
requirements.txt               same deps for a plain `pip install -r`
server.py                      stdlib http.server app + config + rendering + save/ensemble/train
backends.py                    the 8 auto-seg adapters behind one interface + build_registry()
autolabel.py                   spectral_water() fusion engine + derive_prompts() (no torch)
trainer.py                     5-band SegFormer fine-tune (HF transformers) + ONNX export
rl_features.py                 scene_features(), mask IoU / edit-fraction, label_log writer
frontend/                      index.html · style.css · app.js  (canvas labeler + train panel)
annotator_config.example.json  copy to annotator_config.json and edit
work/                          label_log.csv, state.json, features/, cache/<id>/, dataset/, runs/<ts>/
```
