"""Water-segmentation training pipeline for the UFONet 5-band capture rig.

Trains and compares semantic-segmentation models on the gold masks produced by
the annotator (`annotator/`). Modality-aware: the same scenes can be trained as
5-band fusion, plain RGB, NIR-unfiltered RGB, or thermal alone, so the value of
each sensor is measurable rather than assumed.

Entry points (run from the repo root, using the annotator's environment):

    uv run --project annotator python -m training.manifest   # gold scenes -> manifest
    uv run --project annotator python -m training.stats      # per-band normalisation
    uv run --project annotator python -m training.train      # one training run
    uv run --project annotator python -m training.cv         # leave-one-session-out CV
    uv run --project annotator python -m training.compare    # modality x model matrix
"""

__all__ = ["modalities", "data", "stats", "losses", "metrics", "models", "engine"]
