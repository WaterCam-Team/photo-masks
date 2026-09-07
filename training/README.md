# Water-segmentation training pipeline

Fine-tunes semantic-segmentation models on the gold masks the annotator
produces, for the UFONet 5-band capture rig. Modality-aware: the same scenes
train as 5-band fusion, plain RGB, NIR-unfiltered RGB or thermal alone, so the
value of each sensor is **measured** rather than assumed.

```
annotator/  →  training.manifest  →  training.train   →  best_hf/ + norm.json
(gold masks)   (scene pointers)      training.compare    ↳ annotator backend
                                                         ↳ ONNX for the nodes
```

## Quickstart

Run from the repo root, using the annotator's environment:

```bash
# 1. which scenes are gold, and how they split (stable as labels accumulate)
uv run --project annotator python -m training.manifest --out annotator/work/scenes.csv --folds 5

# 2. what the bands look like (optional: training does this itself)
uv run --project annotator python -m training.stats --modality fiveband

# 3. one run
uv run --project annotator python -m training.train \
    --manifest annotator/work/scenes.csv --modality fiveband \
    --arch segformer-b0 --out annotator/work/runs/fiveband-b0 --epochs 60

# 4. the sensor-value comparison, cross-validated (the defensible form)
uv run --project annotator python -m training.compare --cv \
    --modalities rgb_std rgb_nofilt lwir fiveband --archs segformer-b0 \
    --epochs 40 --out annotator/work/compare
```

`--cv` runs a full leave-one-session-out sweep per cell and compares the means.
Without it each cell is scored on a single split — which right now is **one
scene**, and that scene is the pathological low-water one, so the ranking would
be noise. Cost is folds × cells (20 runs ≈ 40 min on CPU for the four
modalities at B0).

To ask which *band* carries the value rather than which sensor, add the two
fusion ablations — `rgb_nir` (drop thermal) and `rgb_thermal` (drop NIR):

```bash
uv run --project annotator python -m training.compare --cv \
    --modalities fiveband rgb_nir rgb_thermal rgb_std --epochs 40 \
    --out annotator/work/ablation
```

**Splits are grouped by capture session, and this matters more than it sounds.**
The rig fires repeatedly within a session, so `20251229-1427`, `-14270`,
`-1428` and `-1429` are the same view seconds apart. Splitting them per scene
puts near-duplicate frames on both sides: measured on these 13 scenes, a
per-scene split reported **0.98 mIoU** where a session-grouped split of the
same data reported **~0.6**. The first number is memorisation. `--group-by
scene` opts out if you genuinely have independent scenes.

Cross-validation, which is the honest way to report a number while the labelled
set is small — `--fold k` makes fold *k* the val set and the rest train, whole
sessions at a time:

```bash
uv run --project annotator python -m training.cv --modality fiveband --epochs 40
```

### First real result (13 gold scenes, 5 sessions)

Leave-one-session-out CV, SegFormer-B0, 5-band, ~2 min/fold on CPU:

| fold | held-out session | val | mIoU | water IoU | boundary F1 |
|---|---|---|---|---|---|
| 1 | Onondaga Lake Park Nov 9 2025 | 1 | 0.983 | 0.983 | 0.693 |
| 3 | example_data | 2 | 0.889 | 0.879 | 0.275 |
| 4 | Brooklyn Dec 2025 | 8 | 0.680 | 0.708 | 0.035 |
| 0 | Kaitlyn | 1 | 0.644 | 0.597 | 0.101 |
| 2 | Summer 2025 | 1 | 0.348 | **0.094** | 0.069 |

**mean 0.709 ± 0.220**, scene-weighted 0.707 — against **0.980** from a
per-scene split of the same data. Read the spread, not the mean: this is 13
scenes over 5 sessions, and fold 4 trains on only 5 scenes because one session
holds 8 of the 13.

The `Summer 2025` failure is the informative one. That scene is ~7.6% water
(the label log shows it took 13 correction strokes and the auto mask scored IoU
0.0), while the training scenes average ~45% water — so the model floods it.
More low-water scenes is the fix, not more epochs.

## Sensor-value result (13 gold scenes, 5 sessions, SegFormer-B0)

Leave-one-session-out CV, identical hyperparameters, every modality read from
the same co-registered pixels. Full table and per-fold detail in
`annotator/work/compare/comparison.md`.

| modality | ch | mean mIoU | ± | water IoU | boundary F1 |
|---|---|---|---|---|---|
| `fiveband` | 5 | **0.709** | 0.220 | **0.652** | 0.234 |
| `rgb_thermal` | 4 | 0.687 | 0.231 | 0.642 | **0.253** |
| `rgb_nir` | 4 | 0.652 | 0.204 | 0.604 | 0.114 |
| `rgb_std` | 3 | 0.620 | 0.210 | 0.593 | 0.077 |
| `rgb_nofilt` | 3 | 0.581 | 0.153 | 0.531 | 0.056 |
| `lwir` | 1 | 0.569 | 0.220 | 0.443 | 0.095 |

**Fusion wins, and thermal is worth about twice what NIR is.** Paired per fold
(which cancels "this session is hard" — the between-session spread of ±0.22 is
larger than every gap in the table):

| change | mean Δ mIoU | folds won |
|---|---|---|
| RGB → RGB+thermal | +0.067 | 4/5 |
| RGB+NIR → 5-band (adds thermal) | +0.056 | 4/5 |
| RGB → RGB+NIR | +0.032 | 3/5 |
| RGB+thermal → 5-band (adds NIR) | +0.022 | 4/5 |

Both orderings agree on the ranking, and `fiveband` beats every single-sensor
modality on 4/5 folds. Thermal also dominates **waterline** localisation: the
two thermal-bearing modalities score boundary F1 0.23–0.25 against 0.06–0.11
for everything else, which for a flood-extent product is the metric that
matters.

**NIR helps only when it stays separable.** `rgb_nofilt` — simply removing the
NIR-cut filter, so NIR leaks into the visible channels — is the *worst* RGB
variant (0.581, losing to plain `rgb_std` by 0.039). The same NIR information
supplied as its own band gains +0.032. That is a direct argument for the rig's
two-shot NIR-ON/OFF differencing over a cheaper filter-off design.

**Treat all of this as directional, not established.** 13 scenes over 5
sessions; 4/5 folds is a sign test at p≈0.19, and every gap is smaller than the
between-session spread. The ranking is consistent across independent
comparisons, which is what makes it worth believing so far — not the margins.
More sessions, not more epochs, is what would settle it.

## Modalities

Every modality is sampled on the co-registered grid of
`color_preserved_5_band.tiff` — the grid the gold masks are drawn on. Band
provenance was **measured**, not assumed (`modalities.verify_provenance()`
re-checks it on any scene):

| modality | ch | source | measured |
|---|---|---|---|
| `fiveband` | 5 | TIFF bands 0–4 | — |
| `rgb_std` | 3 | TIFF bands 0,1,2 | identical to `*-NIR-OFF.jpg` downscaled 2× (MAE 0.00, all 13 scenes) |
| `rgb_nofilt` | 3 | `*-NIR-ON.jpg`, downscaled to the TIFF grid | valid because the optical frame is resized, not warped (above) |
| `lwir` | 1 | TIFF band 3 | **not** the raw `.pgm` (MAE 12–1465 — the Lepton is warped, not resized) |
| `nir` | 1 | TIFF band 4 | identical to `NIR_band.png` (MAE 0.00, all 13 scenes) |
| `rgb_nir` | 4 | bands 0,1,2,4 | fusion ablation: no thermal |
| `rgb_thermal` | 4 | bands 0,1,2,3 | fusion ablation: no NIR |

Two consequences worth keeping in mind:

* `rgb_nofilt` is only legitimate while co-registration **resizes** the optical
  frame rather than warping it. That holds for every session captured so far
  (MAE 0.00 against a plain 2× downscale, on all 13 labelled scenes). Re-run
  `verify_provenance()` on new capture sessions before trusting it.
* `lwir` must come from the TIFF band. The raw `.pgm` is in the Lepton's own
  160×120 geometry and does not line up with the mask.

## Design decisions, and why

**Normalisation is measured over the training split and frozen into the
checkpoint.** `segformer_5band/PERFORMANCE.md` identified per-image per-band
min–max as the original pipeline's main accuracy bug — it "destroys absolute
radiometric values [...] the model cannot learn" water's characteristic low NIR
reflectance, and a single specular pixel sets the range. `stats.py` accumulates
exact 256-bin histograms (every band is uint8) for exact means, stds and
percentiles in one streaming pass, and writes them to `best_hf/norm.json`.
The annotator's `segformer` backend reads that file, so **a served model always
uses the statistics it learned**. Checkpoints without one are treated as legacy
min–max. Choose with `--norm-method meanstd|percentile|minmax`.

**Geometry is preserved.** Frames are 4:3; squashing them to a 512×512 square
stretches every shoreline by a third relative to inference. Training takes a
fixed crop from a randomly rescaled frame instead, and validation runs whole
frames at native resolution, so a reported mIoU means what it means in a paper.
Padding is labelled `IGNORE` (255), never background.

**Photometric augmentation touches only visible-light channels.** Jittering
brightness on a thermal or NIR-difference band fabricates radiometry the sensor
cannot produce — and those absolute values are the signature the model is
supposed to learn.

**Losses optimise what is reported.** Cross-entropy optimises per-pixel
likelihood, not IoU. Default is `ce+lovasz` (Lovász-softmax is a convex
surrogate for IoU itself), matching what the prior B2 modality configs used;
`ce`, `dice`, `ce+dice`, `lovasz` are also available. Class weights come from
the measured water fraction — near-neutral here (~46% water), which is why they
matter mostly for the single-band modalities and puddle-sized scenes.

**Boundary F1 is imported from `agreement.py`**, not reimplemented. The
waterline metric scoring the model is the same function used for
inter-annotator agreement, so "is the model inside the range two humans
disagree by?" is a question the numbers can answer directly.

**n-channel stems are warm-started, not randomised.** RGB filters are copied
across and each extra band starts as their mean; a 1-channel modality gets that
mean as its whole stem. `MODALITY_COMPARISON.md` noted the prior LWIR config
gave up here and trained from random init, naming this averaging as the fix it
never applied.

## Models

| arch | notes |
|---|---|
| `segformer-b0` … `b5` | per-pixel classifier; `--loss` applies; ONNX-exportable for the nodes |
| `mask2former-tiny/small/base` | **mask** classifier trained by Hungarian matching — it brings its own loss, so `--loss` does not apply and the run metadata records `pixel_loss_applies: false` |

The stem is located structurally (the first `Conv2d` in module order) rather
than by attribute path: `transformers` moved SegFormer's stem from
`segformer.encoder.patch_embeddings[0].proj` to
`segformer.stages.0.patch_embeddings.proj` between 4.x and 5.x, which silently
turned the old warm-start into a no-op.

## Run directory

```
ckpt.pt        model + optimizer + scheduler + epoch  (--resume)
best.pt        weights at the best val mIoU
best_hf/       HF checkpoint + norm.json   ← what inference loads
norm.json      the frozen band statistics
config.json    every hyperparameter, plus the exact scene stems per split
metrics.json   per-eval history, and the held-out test result if there is one
```

## Relationship to the annotator

`annotator/trainer.py` is a thin adapter over this package that keeps the Train
panel's CLI and JSONL event protocol stable. It accepts either `--manifest`
(preferred) or the legacy `--data-root` `img_dir/ann_dir` tree from
`export_dataset.py`. The legacy tree can only serve band-subset modalities:
the export copies the TIFF but not the NIR-ON frame, so `rgb_nofilt` needs a
manifest.

Mask encodings differ between the two paths and both are handled: the annotator
writes `water_mask.png` as 0/255, `export_dataset.py` converts to class indices
0/1. `data.read_mask` thresholds at `> 0` for that reason — at `> 127` every
exported mask reads as entirely background, which is empty labels, no error,
and a model that predicts nothing.

## Limitations

* **Labels are the bottleneck, not the pipeline.** Reported numbers need
  ~30–50 labelled scenes plus a held-out set; use `--fold` until then.
* CPU training works but is slow. `--batch` 8–16 and `--accum` on a GPU.
* ONNX export covers SegFormer only (opset 17 — 13 fails on
  `scaled_dot_product_attention` in `transformers` 5.x) and needs
  `uv sync --group export`.

## Deploying to the camera nodes

Export writes `annotator/weights/segformer_5band.onnx` (the Train panel's
"Export for deployment" hard-codes that path), plus `<stem>.norm.json`.

**Normalisation is compiled into the graph.** The exported model takes raw
resized bands in 0–255 units and normalises them itself, and the graph is
stamped with metadata (`normalization`, `input_range`, `norm_mean`,
`norm_std`, `modality`, `bands`) readable through onnxruntime's
`custom_metadata_map`. `SU-WaterCam/tools/segformer_preprocess.py` reads that
and skips its own normalisation. Before this, it always applied per-band
min–max: on a mean/std-trained model that produced **99.7% of the frame called
water** against 52.4% correct, silently. `--no-embed-norm` exports the legacy
contract (pre-normalised input); the Pi then reads the statistics from the
metadata instead.

**The Train panel's "Export for deployment" builds the node bundle directly** —
no second pass with a different script on the Pi:

```
segformer_5band_fp32.onnx    benchmark baseline          (15.1 MB)
segformer_5band_int8.onnx    what the node runs           (4.6 MB)
segformer_5band_prep.onnx    quantization intermediate
deploy.json                  what these are, and the checks that were run
```

INT8 static quantization (per-channel, QDQ) is calibrated on your labelled
scenes, because "representative data" means the exact preprocessing the graph
sees in the field — raw 0–255 bands, since normalisation is compiled in.
Quantization is lossy and silent, so the bundle is verified against its own
full-precision twin and the agreement lands in `deploy.json` (0.9953 pixel
agreement, 0.9915 water IoU on the current model) rather than being assumed.

Per `SU-WaterCam/docs/SEGFORMER_OPTIMIZATION.md`, B0 at 512² is ~10–18 s fp32
and ~4–7 s INT8 on a Pi 4B — an **estimate extrapolated from Cortex-A72
benchmarks, not a measurement**. On x86 INT8 measured slightly *slower* than
fp32 here, which is normal for QDQ without VNNI; time it on a node before
believing the speedup. B0 is already the right variant — that document calls
using B0 "the single highest-impact change for inference speed".

Copy the directory to `/home/pi/segformer_5band/` and
`SU-WaterCam/tools/segformer_daemon.py` picks it up. `--export-onnx` still
produces a single fp32 file for other consumers; the CLI is
`python trainer.py --export-pi <hf_dir> --out-dir <dir> [--no-int8] [--calib …]`.
