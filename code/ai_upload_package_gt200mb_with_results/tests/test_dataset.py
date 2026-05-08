import json
from pathlib import Path

import numpy as np

from tpgr.data.sequence_dataset import SequenceDataset


def test_dataset_loading(tmp_path: Path):
    seq = np.zeros((12, 17, 3), dtype=np.float32)
    npz_path = tmp_path / "sample.npz"
    np.savez_compressed(npz_path, keypoints=seq)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "sample_id": "s1",
        "keypoints_path": "sample.npz",
        "label": "STOP",
        "split": "train",
        "attributes": {"lighting": "day"},
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    ds = SequenceDataset(manifest, split="train", clip_len=16, representation="features", training=False)
    item = ds[0]
    assert tuple(item["x"].shape) == (16, 85)
    assert int(item["y"]) == 1


def test_dataset_loading_with_resample_mode(tmp_path: Path):
    seq = np.zeros((3, 17, 3), dtype=np.float32)
    seq[:, :, 2] = 1.0
    seq[:, 0, 0] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    npz_path = tmp_path / "sample.npz"
    np.savez_compressed(npz_path, keypoints=seq)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "sample_id": "s1",
        "keypoints_path": "sample.npz",
        "label": "STOP",
        "split": "train",
        "attributes": {},
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    ds = SequenceDataset(
        manifest,
        split="train",
        clip_len=8,
        representation="features",
        training=False,
        temporal_mode="resample",
    )
    item = ds[0]

    assert tuple(item["x"].shape) == (8, 85)
