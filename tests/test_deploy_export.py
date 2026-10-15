"""deploy.build on a tiny random SegFormer: metadata round-trip and calibration.

Run from the repo root with the annotator environment plus pytest:

    uv run --project annotator --with pytest python -m pytest tests

Skips where torch / transformers / onnx / onnxruntime are missing. No captures:
the model is randomly initialised and nothing here reads an image.
"""
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('onnx')
ort = pytest.importorskip('onnxruntime')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training import deploy  # noqa: E402
from training import model_metadata as MM  # noqa: E402
from training import stats as ST  # noqa: E402


@pytest.fixture(scope='module')
def hf_dir(tmp_path_factory):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    d = tmp_path_factory.mktemp('hf') / 'tiny_run'
    cfg = SegformerConfig(
        num_labels=2, num_channels=5, depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64], num_attention_heads=[1, 1, 2, 2],
        decoder_hidden_size=16, sr_ratios=[8, 4, 2, 1])
    SegformerForSemanticSegmentation(cfg).save_pretrained(d)
    return d


def stats(method):
    s = ST.BandStats.identity('fiveband', 5, method)
    s.mean[:] = [10, 20, 30, 40, 50]
    s.std[:] = [2, 3, 4, 5, 6]
    return s


def test_int8_without_calibration_scenes_is_an_error(hf_dir, tmp_path):
    with pytest.raises(deploy.CalibrationError, match='real calibration scenes'):
        deploy.build(hf_dir, tmp_path, stats('meanstd'), [], size=64, int8=True)
    assert not list(tmp_path.glob('*.onnx'))        # refused before exporting


def test_unreadable_calibration_scenes_are_an_error():
    rows = [{'tiff_path': '/nonexistent/a.tiff', 'stem': 'a'}]
    reader = deploy._CalibrationReader(rows, 'fiveband', 'input', 64)
    assert reader.get_next() is None and reader.used == 0


def test_fp32_export_carries_metadata_and_a_matching_sidecar(hf_dir, tmp_path):
    info = deploy.build(hf_dir, tmp_path, stats('meanstd'), [], size=64, int8=False,
                        model_name='tiny', model_version='2026-10-15')
    fp32 = tmp_path / deploy.FP32_NAME
    sess = ort.InferenceSession(str(fp32), providers=['CPUExecutionProvider'])
    meta = dict(sess.get_modelmeta().custom_metadata_map)

    assert meta['schema'] == '1' and meta['in_channels'] == '5'
    assert meta['bands'] == 'red,green,blue,thermal,nir'
    assert meta['classes'] == 'background,water'
    assert meta['normalization'] == 'embedded:meanstd'
    assert json.loads(meta['norm_mean']) == [10, 20, 30, 40, 50]
    assert meta['resize_interpolation'] == 'area' and meta['trained_size'] == '64,64'
    assert (meta['model_name'], meta['model_version']) == ('tiny', '2026-10-15')
    assert MM.validate(meta) == []

    out = sess.run(None, {'input': np.zeros((1, 5, 64, 96), np.float32)})[0]
    assert out.shape == (1, 2, 64, 96)

    side = json.loads((tmp_path / 'model.json').read_text())
    assert side['file'] == deploy.FP32_NAME
    assert side['sha256'] == MM.sha256_file(fp32) == info['sha256']
    assert side['bands'] == meta['bands'].split(',')
    assert info['model'] == {'name': 'tiny', 'version': '2026-10-15'}
