# Model metadata, schema 1

Every exported SegFormer `.onnx` carries this in its `metadata_props`, and a
`model.json` next to it repeats it (typed) with the file's `sha256`, which the
`.onnx` cannot contain itself. The node daemon reads it and refuses a model it
cannot feed correctly. **This file is identical in `segformer_5band` and
`photo_processing`**; the code is `tools/model_metadata.py` /
`training/model_metadata.py`.

`metadata_props` is flat, string to string:

| key | value |
|---|---|
| `schema` | `1`. A daemon refuses a larger number. |
| `model_name`, `model_version` | free text and a version or date (`2026-10-15`). |
| `bands` | ordered, comma-separated, from `red,green,blue,thermal,nir`. The daemon warns on an order mismatch with the TIFF; a subset (e.g. `red,green,blue`) is read from the 5-band TIFF by name. |
| `in_channels` | integer, equals the number of bands and the graph input's channel axis. |
| `classes`, `num_classes` | ordered class names (`background,water`); water's index is its position. |
| `normalization` | `minmax`, `meanstd` or `percentile` (the node applies it), or `embedded:<method>` (the graph applies it, feed raw 0-255). |
| `norm_mean`, `norm_std` | JSON lists, one number per band (`meanstd`). |
| `norm_p_lo`, `norm_p_hi` | JSON lists, one number per band (`percentile`). |
| `input_layout` | `NCHW`. |
| `input_hw` | `dynamic`, or `H,W` for a frozen graph. |
| `trained_size` | `H,W` the model was trained and traced at. |
| `input_range` | `raw_0_255` (normalization embedded) or `normalised`. |
| `resize_interpolation` | `area` or `bilinear`: how the node must resize bands to match training. |
| `arch` | optional, e.g. `segformer-b0`. |

`model.json` has the same keys with natural types (`bands` and `classes` as
lists, `norm_*` as number lists, `schema`/`in_channels`/`num_classes` as
integers) plus `file` (the `.onnx` name) and `sha256` (of that file).

A model with no metadata at all is legal: the daemon treats it as per-image
`minmax`, `area` resize, binary classes (the deployed `segformer_5band_dyn.onnx`).
