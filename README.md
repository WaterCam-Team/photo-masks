# photo_processing — 5-band water-segmentation labeling tools

Tools for turning co-registered 5-band camera captures into a
water-segmentation training set: batch auto-labeling, a browser-based labeler
with switchable auto-segmentation backends and an in-tool SegFormer fine-tune
loop, and an exporter that emits MMSeg-format datasets with a provenance trail.

> **Co-registration is not part of this repo.** Every tool here starts from a
> 5-band TIFF that a capture rig has *already* co-registered. Nothing in this
> repository aligns raw frames.

## The 5-band input

One `color_preserved_5_band.tiff` per scene directory, uint8, typically
1296×972:

| band (1-based) | content |
|---|---|
| 1, 2, 3 | **R, G, B** (optical) |
| 4 | Thermal / LWIR, normalised to 0–255 |
| 5 | NIR difference (NIR-ON frame minus NIR-OFF frame) |

Co-registration also writes `final_5_band.tiff`, and **the two are not the same
image**: `final_5_band` is **B, G, R**, thermal, NIR at a squashed 512×512,
because it carries OpenCV's native channel order straight from `cv2.imread`.
Everything in this repo reads `color_preserved`. Feeding a model the other file
swaps red and blue and nothing raises — the shape is right, the mask looks like
a mask, and the water is in the wrong place. `SU-WaterCam/docs/CAPTURE_PIPELINE_NOTES.md`
is the canonical table.

Captures made from 2026-09-26 on state their own order in a `BAND_ORDER` TIFF
tag (`red,green,blue,thermal,nir`), in the same format the exported `.onnx`
declares, so file and model can be compared without interpreting prose. Older
captures carry only the band descriptions and are read by keyword.

Water absorbs NIR, so band 5 is the primary cue — but it does **not** separate
water from snow, which is why a human stays in the loop.

## Workflow

```
   (capture rig)   ──►   pipeline.py   ──►   annotator/   ──►   export_dataset.py
 co-registration          auto-label        review / label        MMSeg dataset
  — not in this repo                          (or review.py)
```

| stage | what it does |
|---|---|
| **`pipeline.py`** | Batch: scan a captures root, run a CPU auto-labeler on every 5-band TIFF, write `water_mask_auto.png` per scene and a `manifest.csv` |
| **`annotator/`** | Local web app: paint masks by hand, **click on the water and let SAM2 outline it**, or seed from any of 8 auto-segmentation backends, compare them, correct, and save. Writes `water_mask.png` and logs every decision. **Start here** — see [`annotator/README.md`](annotator/README.md) |
| **`review.py`** | Minimal terminal alternative to the annotator: approve / reject / flag the batch masks one at a time |
| **`export_dataset.py`** | Emit `img_dir/` + `ann_dir/` in MMSeg format. `--gold-only` exports *only* human-verified masks and writes `dataset_provenance.csv` |
| **`agreement.py`** | Score how much two labelers agree on the same scenes (IoU, Cohen's κ, boundary F1) — the noise ceiling your model mIoU should be read against |

## Quick start

```bash
cd annotator
uv sync                                  # or: pip install -r requirements.txt
uv run server.py --annotator <your-name>
# open http://127.0.0.1:8000
```

With no configuration the annotator scans the bundled `example_data/`, so you
get real scenes to click through immediately. Point it at your own captures
with `--root /path/to/captures`.

Batch-label and export instead:

```bash
python pipeline.py /path/to/captures --preview     # auto-label -> manifest.csv
python review.py  /path/to/captures                # approve / reject
python export_dataset.py /path/to/captures /path/to/dataset/
```

Or export only what a human actually verified in the annotator:

```bash
python export_dataset.py --gold-only /path/to/dataset/ \
    --label-log annotator/work/label_log.csv
```

## Label quality

`--gold-only` exists because auto-generated masks and human-verified masks are
not the same thing and must not be mixed silently. It exports a scene only when
the latest decision in the annotator's log was made by a person — an accepted
auto mask, an edited one, or a hand-drawn one — and never falls back to an
unverified auto mask. `dataset_provenance.csv` records, per scene, which route
it came from, which backend seeded it, how much the human changed, and how long
it took.

With more than one labeler, turn on the agreement sample (`replicate_pct` in
the annotator's config). A deterministic slice of scenes is then served to a
second person **blind**, every annotator's mask is kept, and `agreement.py`
scores how much they actually agree. Those double-labeled scenes are routed to
validation rather than training, because they are the only ones whose human
noise floor is known — a model scoring above it is fitting label noise. See
**Measuring agreement between labelers** in the annotator README.

## Requirements

Python 3.10+. Core dependencies are numpy, rasterio,
**opencv-contrib-python-headless** (the contrib build specifically — plain
opencv silently disables the guided-filter boundary refinement), scipy, Pillow
and scikit-image. The model-based backends and the fine-tune loop are optional
extras; a missing one disables its backend in the UI with the reason on hover
rather than breaking the app. Full details, including GPU setup for NVIDIA /
AMD / Intel / Apple, are in [`annotator/README.md`](annotator/README.md).

## Repository layout

```
pipeline.py              batch auto-labeling -> manifest.csv
review.py                terminal review loop over manifest.csv
export_dataset.py        MMSeg export, incl. --gold-only + provenance
agreement.py             inter-annotator agreement over the annotator's masks
example_data/            two real co-registered scenes, enough to try everything
annotator/               the labeling web app (see its own README)
  server.py                stdlib HTTP server: config, discovery, rendering, save/ensemble/train
  backends.py              8 auto-seg adapters + InteractiveSam (SAM2 click-to-segment)
  autolabel.py             spectral fusion engine + SAM prompt derivation (no torch)
  trainer.py               5-band SegFormer fine-tune + ONNX export
  rl_features.py           scene features, mask IoU, decision-log writer
  frontend/                canvas labeler — zero JS dependencies
  LABELING_GUIDE.md        class definitions: read before labeling
  CLAUDE.md                architecture, invariants, gotchas: read before changing code
```

## Security

The annotator is a **localhost tool with no authentication**. Anyone who can
reach its port can read your imagery, overwrite masks and start training jobs.
Keep the default `--host 127.0.0.1` and use an SSH tunnel for remote work; see
the callout in [`annotator/README.md`](annotator/README.md).

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
