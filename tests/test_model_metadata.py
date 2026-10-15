"""Schema-1 model metadata (training/model_metadata.py): pure Python, no torch."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training import model_metadata as MM  # noqa: E402

BANDS = ['red', 'green', 'blue', 'thermal', 'nir']


def build(**kw):
    args = dict(model_name='t', model_version='2026-10-15', bands=BANDS,
                classes=['background', 'water'], normalization='minmax')
    args.update(kw)
    return MM.build_metadata(**args)


def test_minmax_defaults_and_derived_fields():
    p = build()
    assert p['schema'] == '1' and p['in_channels'] == '5' and p['num_classes'] == '2'
    assert p['bands'] == 'red,green,blue,thermal,nir'
    assert p['input_range'] == 'normalised' and p['resize_interpolation'] == 'area'
    assert p['trained_size'] == '512,512' and p['input_hw'] == 'dynamic'
    assert MM.validate(p) == []


def test_embedded_normalization_means_raw_input():
    p = build(normalization='embedded:meanstd',
              norm_stats={'norm_mean': [1] * 5, 'norm_std': [2] * 5})
    assert p['input_range'] == 'raw_0_255'


def test_meanstd_needs_one_statistic_per_band():
    with pytest.raises(ValueError, match='one value per band'):
        build(normalization='meanstd', norm_stats={'norm_mean': [1] * 3, 'norm_std': [1] * 3})
    with pytest.raises(ValueError, match='missing'):
        build(normalization='meanstd')


def test_unknown_normalization_and_interpolation_are_rejected():
    with pytest.raises(ValueError, match='unknown normalization'):
        build(normalization='zscore')
    with pytest.raises(ValueError, match='resize_interpolation'):
        build(resize_interpolation='cubic')


def test_sidecar_is_typed_and_carries_the_sha(tmp_path):
    onnx = tmp_path / 'm.onnx'
    onnx.write_bytes(b'not really onnx')
    p = build(normalization='meanstd',
              norm_stats={'norm_mean': [1.5] * 5, 'norm_std': [2] * 5})
    path = MM.write_sidecar(onnx, p)
    d = json.loads(open(path).read())
    assert os.path.basename(path) == 'model.json'
    assert d['bands'] == BANDS and d['in_channels'] == 5 and d['schema'] == 1
    assert d['norm_mean'] == [1.5] * 5 and d['file'] == 'm.onnx'
    assert d['sha256'] == MM.sha256_file(onnx) and len(d['sha256']) == 64
