from __future__ import annotations

from typing import Iterable

import numpy as np


COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

WHOLEBODY_FOCUS_INDICES = tuple(range(17)) + tuple(range(17, 23)) + tuple(range(91, 133))


def _as_keypoints(sequence: np.ndarray) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"keypoint sequence must be [T, V, C], got {arr.shape}")
    if arr.shape[-1] == 2:
        score = np.ones((*arr.shape[:2], 1), dtype=np.float32)
        arr = np.concatenate([arr, score], axis=-1)
    if arr.shape[-1] < 3:
        raise ValueError(f"keypoint channel dimension must be at least 2, got {arr.shape}")
    return arr[..., :3].astype(np.float32, copy=False)


def _safe_center_scale(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = arr[..., :2]
    score = arr[..., 2]
    visible = score > 0.05
    centers = np.zeros((arr.shape[0], 2), dtype=np.float32)
    scales = np.ones((arr.shape[0], 1), dtype=np.float32)
    for t in range(arr.shape[0]):
        if arr.shape[1] > 12 and visible[t, [5, 6, 11, 12]].any():
            idx = [i for i in [5, 6, 11, 12] if i < arr.shape[1] and visible[t, i]]
            pts = xy[t, idx]
        else:
            pts = xy[t, visible[t]]
        if pts.size == 0:
            pts = xy[t]
        centers[t] = pts.mean(axis=0)
        span = np.ptp(pts, axis=0) if len(pts) > 1 else np.asarray([1.0, 1.0], dtype=np.float32)
        scales[t, 0] = float(max(span.max(), 1.0))
    return centers, scales


def normalize_keypoints_sequence(sequence: np.ndarray) -> np.ndarray:
    arr = _as_keypoints(sequence)
    centers, scales = _safe_center_scale(arr)
    out = arr.copy()
    out[..., :2] = (out[..., :2] - centers[:, None, :]) / scales[:, None, :]
    out[..., 2] = np.clip(out[..., 2], 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def _temporal_velocity(xy: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(xy, dtype=np.float32)
    if len(xy) > 1:
        vel[1:] = xy[1:] - xy[:-1]
        vel[0] = vel[1]
    return vel


def _base_features(sequence: np.ndarray, include_velocity: bool = True) -> np.ndarray:
    norm = normalize_keypoints_sequence(sequence)
    parts = [norm.reshape(norm.shape[0], -1)]
    if include_velocity:
        parts.append(_temporal_velocity(norm[..., :2]).reshape(norm.shape[0], -1))
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def _edge_features(sequence: np.ndarray, edges: Iterable[tuple[int, int]] = COCO17_EDGES) -> np.ndarray:
    norm = normalize_keypoints_sequence(sequence)
    values = []
    for i, j in edges:
        if i >= norm.shape[1] or j >= norm.shape[1]:
            continue
        delta = norm[:, j, :2] - norm[:, i, :2]
        dist = np.linalg.norm(delta, axis=-1, keepdims=True)
        values.extend([delta, dist])
    if not values:
        return np.zeros((norm.shape[0], 0), dtype=np.float32)
    return np.concatenate(values, axis=-1).astype(np.float32, copy=False)


def _global_features(sequence: np.ndarray) -> np.ndarray:
    norm = normalize_keypoints_sequence(sequence)
    xy = norm[..., :2]
    score = norm[..., 2:3]
    mean_xy = xy.mean(axis=1)
    std_xy = xy.std(axis=1)
    mean_score = score.mean(axis=1)
    min_xy = xy.min(axis=1)
    max_xy = xy.max(axis=1)
    return np.concatenate([mean_xy, std_xy, mean_score, min_xy, max_xy], axis=-1).astype(np.float32, copy=False)


def _select_indices(sequence: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    arr = _as_keypoints(sequence)
    keep = [idx for idx in indices if idx < arr.shape[1]]
    if not keep:
        return arr[:, : min(arr.shape[1], 17)]
    return arr[:, keep]


def _fit_dim(features: np.ndarray, dim: int) -> np.ndarray:
    if features.shape[1] == dim:
        return features.astype(np.float32, copy=False)
    if features.shape[1] > dim:
        return features[:, :dim].astype(np.float32, copy=False)
    pad = np.zeros((features.shape[0], dim - features.shape[1]), dtype=np.float32)
    return np.concatenate([features, pad], axis=-1).astype(np.float32, copy=False)


def bbox_sequence_to_track_features(bbox_sequence: np.ndarray) -> np.ndarray:
    bbox = np.asarray(bbox_sequence, dtype=np.float32)
    if bbox.ndim != 2 or bbox.shape[1] != 4:
        return np.zeros((bbox.shape[0] if bbox.ndim else 0, 14), dtype=np.float32)
    x1, y1, x2, y2 = bbox.T
    w = np.maximum(x2 - x1, 1.0)
    h = np.maximum(y2 - y1, 1.0)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    area = w * h
    aspect = w / h
    base = np.stack([cx, cy, w, h, area, aspect], axis=-1)
    denom = np.maximum(np.nanmax(np.abs(base), axis=0, keepdims=True), 1.0)
    base = base / denom
    vel = np.zeros((bbox.shape[0], 4), dtype=np.float32)
    acc = np.zeros((bbox.shape[0], 4), dtype=np.float32)
    motion = np.stack([cx, cy, w, h], axis=-1)
    if len(bbox) > 1:
        vel[1:] = motion[1:] - motion[:-1]
        vel[0] = vel[1]
    if len(bbox) > 2:
        acc[1:] = vel[1:] - vel[:-1]
        acc[0] = acc[1]
    return np.concatenate([base, vel / 100.0, acc / 100.0], axis=-1).astype(np.float32, copy=False)


def infer_tcn_feature_dim(
    num_keypoints: int,
    include_velocity: bool = True,
    feature_variant: str = "base",
) -> int:
    variant = str(feature_variant or "base").lower()
    if variant in {"wholebody_focus", "focus"} and num_keypoints >= 100:
        return 456
    if variant in {"wholebody_focus_track", "focus_track"} and num_keypoints >= 100:
        return 470
    if variant in {"wholebody_full", "full"} and num_keypoints >= 100:
        return 826
    if variant in {"wholebody_full_track", "full_track"} and num_keypoints >= 100:
        return 840
    base = int(num_keypoints) * (5 if include_velocity else 3)
    if variant == "rich":
        return base + 3 * len(COCO17_EDGES) + 9
    if variant.endswith("_track"):
        return base + 14
    return base


def sequence_to_tcn_features(
    sequence: np.ndarray,
    include_velocity: bool = True,
    feature_variant: str = "base",
    bbox_track: np.ndarray | None = None,
) -> np.ndarray:
    arr = _as_keypoints(sequence)
    variant = str(feature_variant or "base").lower()
    if variant in {"wholebody_focus", "focus", "wholebody_focus_track", "focus_track"} and arr.shape[1] >= 100:
        selected = _select_indices(arr, WHOLEBODY_FOCUS_INDICES)
        features = np.concatenate([
            _base_features(selected, include_velocity=include_velocity),
            _edge_features(selected),
            _global_features(selected),
        ], axis=-1)
    elif variant in {"wholebody_full", "full", "wholebody_full_track", "full_track"} and arr.shape[1] >= 100:
        features = np.concatenate([
            _base_features(arr, include_velocity=include_velocity),
            _edge_features(arr),
            _global_features(arr),
        ], axis=-1)
    elif variant == "rich":
        features = np.concatenate([
            _base_features(arr, include_velocity=include_velocity),
            _edge_features(arr),
            _global_features(arr),
        ], axis=-1)
    else:
        features = _base_features(arr, include_velocity=include_velocity)
    if variant.endswith("_track"):
        if bbox_track is None:
            track = np.zeros((arr.shape[0], 14), dtype=np.float32)
        else:
            track = bbox_sequence_to_track_features(bbox_track)
            track = adapt_sequence_length(track, arr.shape[0], training=False, mode="resample")
        features = np.concatenate([features, track], axis=-1)
    target_dim = infer_tcn_feature_dim(arr.shape[1], include_velocity=include_velocity, feature_variant=variant)
    return _fit_dim(features, target_dim)


def sequence_to_graph_input(sequence: np.ndarray) -> np.ndarray:
    norm = normalize_keypoints_sequence(sequence)
    return np.transpose(norm, (2, 0, 1)).astype(np.float32, copy=False)


def resample_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    target = max(1, int(target_len))
    if arr.shape[0] == 0:
        return np.zeros((target, *arr.shape[1:]), dtype=np.float32)
    if arr.shape[0] == target:
        return arr.astype(np.float32, copy=True)
    if arr.shape[0] == 1:
        return np.repeat(arr, target, axis=0).astype(np.float32, copy=False)
    flat = arr.reshape(arr.shape[0], -1)
    src = np.linspace(0.0, 1.0, num=arr.shape[0], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=target, dtype=np.float32)
    out = np.stack([np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
    return out.reshape((target, *arr.shape[1:])).astype(np.float32, copy=False)


def adapt_sequence_length(
    sequence: np.ndarray,
    target_len: int,
    training: bool = False,
    mode: str = "crop_pad",
) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    target = max(1, int(target_len))
    if arr.shape[0] == 0:
        return np.zeros((target, *arr.shape[1:]), dtype=np.float32)
    mode_name = str(mode or "crop_pad").lower()
    if mode_name in {"resample", "linear"}:
        return resample_sequence(arr, target)
    if mode_name == "hybrid" and not training:
        return resample_sequence(arr, target)
    if arr.shape[0] == target:
        return arr.astype(np.float32, copy=True)
    if arr.shape[0] > target:
        if mode_name in {"tail_pad", "tail"}:
            return arr[-target:].astype(np.float32, copy=True)
        max_start = arr.shape[0] - target
        start = np.random.randint(0, max_start + 1) if training else max_start // 2
        return arr[start:start + target].astype(np.float32, copy=True)
    pad_count = target - arr.shape[0]
    pad_frame = arr[-1:]
    pad = np.repeat(pad_frame, pad_count, axis=0)
    if mode_name in {"tail_pad", "tail"}:
        return np.concatenate([pad, arr], axis=0).astype(np.float32, copy=False)
    return np.concatenate([arr, pad], axis=0).astype(np.float32, copy=False)

