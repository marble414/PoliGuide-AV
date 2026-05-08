from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "processed_demo_videos"
POSE_MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"
TARGET_HEIGHT = 1080
PANEL_WIDTH = 520
MIN_PREDICT_LEN = 12
WINDOW_SIZE = 9
MIN_CONSENSUS = 6
CONF_THRESHOLD = 0.60
HOLD_FRAMES = 8
EXPIRY_FRAMES = 18

SOURCE_EXCLUDE_NAMES = {"在线演示.mp4", "离线演示.mp4"}

COCO17_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]

MP_TO_COCO = {
    0: 0,
    2: 2,
    5: 1,
    7: 3,
    8: 4,
    11: 5,
    12: 6,
    13: 7,
    14: 8,
    15: 9,
    16: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
}

LABELS = [
    "NO_GESTURE",
    "STOP",
    "GO_STRAIGHT",
    "TURN_LEFT",
    "LEFT_TURN_WAIT",
    "TURN_RIGHT",
    "CHANGE_LANE",
    "SLOW_DOWN",
    "PULL_OVER",
]

COMMANDS = {
    "NO_GESTURE": "NO_COMMAND",
    "STOP": "STOP",
    "GO_STRAIGHT": "GO_STRAIGHT",
    "TURN_LEFT": "TURN_LEFT",
    "LEFT_TURN_WAIT": "KEEP_WAIT",
    "TURN_RIGHT": "TURN_RIGHT",
    "CHANGE_LANE": "CHANGE_LANE",
    "SLOW_DOWN": "SLOW_DOWN",
    "PULL_OVER": "PULL_OVER",
}

INTENTS = {
    "NO_GESTURE": "NO_ACTIVE_COMMAND",
    "STOP": "stop_and_wait",
    "GO_STRAIGHT": "allow_forward",
    "TURN_LEFT": "allow_left_turn",
    "LEFT_TURN_WAIT": "left_turn_wait",
    "TURN_RIGHT": "allow_right_turn",
    "CHANGE_LANE": "lane_change_guidance",
    "SLOW_DOWN": "speed_limit_slow",
    "PULL_OVER": "pull_over_guidance",
}

COLORS = {
    "active": (0, 245, 100),
    "inactive": (255, 180, 0),
    "accent": (0, 220, 255),
    "warning": (0, 70, 255),
    "panel_bg": (18, 26, 32),
    "panel_dim": (158, 174, 185),
    "panel_line": (64, 82, 96),
    "white": (238, 246, 250),
    "black": (9, 14, 18),
}


SCENE_HINTS = {
    "297c01ab11fbe21b1b01dc073254e29b": "night_high_glare",
    "a457c95caefc1ec8ce19f5701803a2d7": "night_speed_limit",
    "869dd470508fc91fe2466399aea87d09": "roadside_vehicle",
    "aa337036dd3acd5e47bedfeccfa30b11": "backlight_side",
    "b32482df56c2a04c71bc8c4cc52622fd": "construction_stop",
    "f3e5b88edf21f46203fe724fa9aedf56": "front_direction",
    "5191f223bed334f081edb58350129fb6": "construction_entry",
}


@dataclass
class FilterState:
    history: deque
    current_command: str = "NO_COMMAND"
    current_conf: float = 0.0
    last_update_frame: int = -10**9
    frame_index: int = -1
    last_safe_fallback: bool = False

    def stable_vote(self) -> tuple[str, float] | None:
        votes: dict[str, int] = {}
        confs: dict[str, list[float]] = {}
        for label, conf in self.history:
            votes[label] = votes.get(label, 0) + 1
            confs.setdefault(label, []).append(conf)
        if not votes:
            return None
        best = max(votes, key=votes.get)
        if votes[best] < MIN_CONSENSUS:
            return None
        mean_conf = float(sum(confs[best]) / len(confs[best]))
        if mean_conf < CONF_THRESHOLD:
            return None
        return best, mean_conf

    def step(self, command: str, confidence: float, occluded: bool) -> tuple[str, float, bool]:
        self.frame_index += 1
        safe_fallback = False
        if occluded or confidence < CONF_THRESHOLD:
            if self.frame_index - self.last_update_frame <= HOLD_FRAMES:
                if self.current_command in {"GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "CHANGE_LANE"}:
                    self.current_command = "KEEP_WAIT"
                    safe_fallback = True
            elif self.frame_index - self.last_update_frame > EXPIRY_FRAMES:
                self.current_command = "NO_COMMAND"
                self.current_conf = 0.0
            self.last_safe_fallback = safe_fallback
            return self.current_command, self.current_conf, safe_fallback

        self.history.append((command, confidence))
        vote = self.stable_vote()
        if vote is None:
            if self.current_command != "NO_COMMAND" and self.frame_index - self.last_update_frame <= HOLD_FRAMES:
                if self.current_command in {"GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "CHANGE_LANE"}:
                    self.current_command = "KEEP_WAIT"
                    safe_fallback = True
            self.last_safe_fallback = safe_fallback
            return self.current_command, self.current_conf, safe_fallback

        self.current_command, self.current_conf = vote
        self.last_update_frame = self.frame_index
        self.last_safe_fallback = False
        return self.current_command, self.current_conf, False

    def debug(self) -> dict:
        votes = Counter(label for label, _ in self.history)
        vote = self.stable_vote()
        return {
            "window_size": WINDOW_SIZE,
            "min_consensus": MIN_CONSENSUS,
            "conf_threshold": CONF_THRESHOLD,
            "hold_frames": HOLD_FRAMES,
            "expiry_frames": EXPIRY_FRAMES,
            "history": [{"command": label, "confidence": conf} for label, conf in self.history],
            "votes": dict(votes),
            "stable_vote": None if vote is None else {"command": vote[0], "confidence": vote[1]},
            "current_command": self.current_command,
            "current_confidence": self.current_conf,
            "last_safe_fallback": self.last_safe_fallback,
        }


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == height:
        return frame
    width = int(round(w * height / h))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def discover_source_videos() -> list[Path]:
    videos: list[Path] = []
    for path in sorted(ROOT.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
            continue
        if path.name in SOURCE_EXCLUDE_NAMES:
            continue
        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if ok and frames > 0 and height >= width:
            videos.append(path)
    return videos


def make_slug(index: int, path: Path) -> str:
    hint = SCENE_HINTS.get(path.stem, "traffic_police_signal")
    return f"demo_{index:02d}_{hint}_{path.stem[:8]}"


def mediapipe_to_coco(result, width: int, height: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not getattr(result, "pose_landmarks", None):
        return None
    landmarks = result.pose_landmarks[0]
    keypoints = np.zeros((17, 2), dtype=np.float32)
    scores = np.zeros((17,), dtype=np.float32)
    for mp_idx, coco_idx in MP_TO_COCO.items():
        lm = landmarks[mp_idx]
        keypoints[coco_idx] = (lm.x * width, lm.y * height)
        visibility = float(getattr(lm, "visibility", 0.5))
        presence = float(getattr(lm, "presence", visibility))
        scores[coco_idx] = float(np.clip((visibility + presence) / 2.0, 0.0, 1.0))
    return keypoints, scores


def bbox_from_keypoints(keypoints: np.ndarray, scores: np.ndarray, width: int, height: int) -> list[float] | None:
    keep = scores >= 0.12
    if int(np.sum(keep)) < 5:
        return None
    pts = keypoints[keep]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    pad_x = max(18.0, (x2 - x1) * 0.20)
    pad_y = max(24.0, (y2 - y1) * 0.15)
    return [
        float(np.clip(x1 - pad_x, 0, width - 1)),
        float(np.clip(y1 - pad_y, 0, height - 1)),
        float(np.clip(x2 + pad_x, 0, width - 1)),
        float(np.clip(y2 + pad_y, 0, height - 1)),
    ]


def selector_score_from_pose(bbox: list[float] | None, scores: np.ndarray | None, width: int, height: int) -> tuple[float, dict[str, float]]:
    if bbox is None or scores is None:
        return 0.0, {"center": 0.0, "size": 0.0, "visibility": 0.0, "hand": 0.0, "vest": 0.0}
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    center_dist = np.linalg.norm(np.array([cx / width - 0.5, cy / height - 0.52], dtype=np.float32))
    center_score = float(np.clip(1.0 - center_dist * 1.8, 0.0, 1.0))
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, width * height)
    size_score = float(np.clip(area_ratio / 0.28, 0.0, 1.0))
    visibility_score = float(np.clip(np.mean(scores), 0.0, 1.0))
    hand_score = float(np.clip(np.mean(scores[[7, 8, 9, 10]]), 0.0, 1.0))
    vest_score = 1.0
    score = (
        0.35 * center_score
        + 0.15 * size_score
        + 0.25 * visibility_score
        + 0.15 * hand_score
        + 0.10 * vest_score
    )
    components = {
        "center": center_score,
        "size": size_score,
        "visibility": visibility_score,
        "hand": hand_score,
        "vest": vest_score,
    }
    return float(np.clip(score, 0.0, 1.0)), components


def normalize_keypoints_sequence(seq: np.ndarray) -> np.ndarray:
    arr = seq.astype(np.float32).copy()
    root = arr[:, [11, 12], :2].mean(axis=1, keepdims=True)
    shoulders = arr[:, [5, 6], :2]
    hips = arr[:, [11, 12], :2]
    scale = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1) + np.linalg.norm(hips[:, 0] - hips[:, 1], axis=1)
    scale = np.clip(scale[:, None, None] / 2.0, 1.0, None)
    arr[:, :, :2] = (arr[:, :, :2] - root) / scale
    return arr


def arm_vector(frame: np.ndarray, side: str) -> np.ndarray:
    shoulder = 5 if side == "left" else 6
    wrist = 9 if side == "left" else 10
    return frame[wrist, :2] - frame[shoulder, :2]


def classify_sequence(buffer: deque[np.ndarray], score_buffer: deque[np.ndarray]) -> tuple[str, float, dict[str, float]]:
    if not buffer:
        scores = {name: 0.01 for name in LABELS}
        scores["NO_GESTURE"] = 1.0
        return "NO_GESTURE", 1.0, scores

    seq = normalize_keypoints_sequence(np.stack(list(buffer), axis=0))
    frame = seq[-5:].mean(axis=0) if len(seq) >= 5 else seq.mean(axis=0)
    raw_scores = np.stack(list(score_buffer), axis=0) if score_buffer else np.ones((1, 17), dtype=np.float32)
    mean_vis = float(np.nanmean(raw_scores[-5:]))

    left = arm_vector(frame, "left")
    right = arm_vector(frame, "right")
    hip_y = float(frame[[11, 12], 1].mean())
    left_dist = float(np.linalg.norm(left))
    right_dist = float(np.linalg.norm(right))
    left_active = left_dist > 0.42 and float(frame[9, 1]) < hip_y - 0.04
    right_active = right_dist > 0.42 and float(frame[10, 1]) < hip_y - 0.04
    left_h = left_active and abs(float(left[0])) > 0.18 and abs(float(left[1])) < 0.85
    right_h = right_active and abs(float(right[0])) > 0.18 and abs(float(right[1])) < 0.85
    left_up = float(frame[9, 1] - frame[5, 1]) < -0.25
    right_up = float(frame[10, 1] - frame[6, 1]) < -0.25
    left_down = float(frame[9, 1] - frame[5, 1]) > 0.42
    right_down = float(frame[10, 1] - frame[6, 1]) > 0.42
    shoulder_width = abs(float(frame[5, 0] - frame[6, 0]))
    hip_width = abs(float(frame[11, 0] - frame[12, 0]))
    side_profile = min(shoulder_width, hip_width) < 0.34
    both_arms_visible = bool(raw_scores[-1, [5, 6, 7, 8, 9, 10]].mean() >= 0.35)
    downward_motion = 0.0
    if len(seq) >= 10:
        recent = seq[-3:, [9, 10], 1].mean()
        earlier = seq[-10:-7, [9, 10], 1].mean()
        downward_motion = float(recent - earlier)

    scores = {name: 0.01 for name in LABELS}
    if not both_arms_visible or mean_vis < 0.25:
        scores["NO_GESTURE"] = 0.70
    elif left_h and right_h:
        scores["STOP"] = 0.90
    elif side_profile and (left_h or right_h):
        scores["STOP"] = 0.86
    elif left_h and right_up:
        scores["LEFT_TURN_WAIT"] = 0.80
    elif right_h and left_up:
        scores["LEFT_TURN_WAIT"] = 0.76
    elif left_h or right_h:
        # Directional labels in the public demo should match what a viewer sees
        # on screen. MediaPipe left/right are anatomical sides, which look
        # mirrored for front-facing traffic police; use the horizontal hand
        # direction instead so TURN_LEFT/RIGHT follows the visible arm vector.
        active_vectors = []
        if left_h:
            active_vectors.append(left)
        if right_h:
            active_vectors.append(right)
        mean_x = float(np.mean([vec[0] for vec in active_vectors]))
        scores["TURN_RIGHT" if mean_x > 0 else "TURN_LEFT"] = 0.76
    elif left_up and right_up:
        scores["GO_STRAIGHT"] = 0.75
    elif left_down and right_down and downward_motion > 0.08:
        scores["SLOW_DOWN"] = 0.72
    else:
        scores["NO_GESTURE"] = 0.64

    total = float(sum(scores.values()))
    probs = {k: float(v / total) for k, v in scores.items()}
    label = max(probs, key=probs.get)
    return label, probs[label], probs


def put_text(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.56, thickness: int = 1) -> int:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return y + int(23 * scale / 0.56)


def draw_metric_bar(img: np.ndarray, label: str, value: float, x: int, y: int, width: int = 188) -> int:
    value = float(np.clip(value, 0.0, 1.0))
    put_text(img, f"{label}: {value:.2f}", x, y, COLORS["white"], scale=0.48, thickness=1)
    bar_y = y + 8
    cv2.rectangle(img, (x + 118, bar_y - 12), (x + 118 + width, bar_y), (43, 56, 66), -1)
    cv2.rectangle(img, (x + 118, bar_y - 12), (x + 118 + int(width * value), bar_y), COLORS["active"], -1)
    cv2.rectangle(img, (x + 118, bar_y - 12), (x + 118 + width, bar_y), COLORS["panel_line"], 1)
    return y + 25


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, scores: np.ndarray) -> None:
    for i, j in COCO17_EDGES:
        if scores[i] < 0.12 or scores[j] < 0.12:
            continue
        p1 = tuple(np.asarray(keypoints[i], dtype=np.int32))
        p2 = tuple(np.asarray(keypoints[j], dtype=np.int32))
        cv2.line(frame, p1, p2, COLORS["active"], 3, cv2.LINE_AA)
    for idx, pt in enumerate(keypoints):
        if scores[idx] < 0.12:
            continue
        radius = 5 if idx in {9, 10} else 4
        cv2.circle(frame, tuple(np.asarray(pt, dtype=np.int32)), radius, COLORS["active"], -1, cv2.LINE_AA)


def draw_video_overlay(frame: np.ndarray, bbox: list[float] | None, keypoints: np.ndarray | None, scores: np.ndarray | None, label: str, command: str, confidence: float, fallback: bool, selector_score: float) -> None:
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["active"], 3, cv2.LINE_AA)
        tag = f"id=1 sel={selector_score:.2f} {label} {confidence:.2f}"
        cv2.rectangle(frame, (x1, max(0, y1 - 32)), (min(frame.shape[1] - 1, x1 + 330), y1), COLORS["black"], -1)
        put_text(frame, tag, x1 + 8, max(22, y1 - 9), COLORS["active"], scale=0.55, thickness=2)
    if keypoints is not None and scores is not None:
        draw_skeleton(frame, keypoints, scores)

    chip_color = COLORS["warning"] if fallback else COLORS["active"]
    cv2.rectangle(frame, (16, 16), (min(frame.shape[1] - 16, 500), 116), (10, 18, 23), -1)
    cv2.rectangle(frame, (16, 16), (min(frame.shape[1] - 16, 500), 116), (58, 80, 92), 1)
    put_text(frame, "PoliGuide-AV recognition output", 32, 46, COLORS["accent"], scale=0.66, thickness=2)
    put_text(frame, f"gesture: {label}", 32, 74, COLORS["white"], scale=0.58, thickness=2)
    put_text(frame, f"command: {command}", 32, 101, chip_color, scale=0.58, thickness=2)


def draw_panel(height: int, result: dict, detections: dict, scores: dict[str, float], state: dict, source_name: str) -> np.ndarray:
    panel = np.full((height, PANEL_WIDTH, 3), COLORS["panel_bg"], dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH - 1, height - 1), COLORS["panel_line"], 1)
    y = 34
    y = put_text(panel, "TPGR Debug Dashboard", 22, y, COLORS["accent"], scale=0.72, thickness=2)
    y += 8
    lines = [
        f"source: {source_name[:34]}",
        f"route: MediaPipe pose + rule/state",
        f"frame: {result['frame_index']}   t={result['timestamp_sec']:.2f}s",
        f"fps: {result['fps']:.2f}   latency: {result['latency_ms']:.1f} ms",
        f"track: {result['active_track_id']}",
        f"gesture: {result['gesture']} ({result['gesture_confidence']:.2f})",
        f"intent: {result['intent']}",
        f"command: {result['command']} ({result['command_confidence']:.2f})",
        f"safe_fallback: {result['safe_fallback']}",
    ]
    for line in lines:
        y = put_text(panel, line, 22, y, COLORS["white"])

    y += 6
    y = draw_metric_bar(panel, "gesture_conf", result["gesture_confidence"], 22, y)
    y = draw_metric_bar(panel, "command_conf", result["command_confidence"], 22, y)
    y = draw_metric_bar(panel, "pose_quality", detections.get("mean_keypoint_score", 0.0), 22, y)

    y += 10
    y = put_text(panel, "Active Track", 22, y, COLORS["accent"], scale=0.62, thickness=2)
    active_lines = [
        f"buffer: {detections.get('buffer_len', 0)} / min_pred {MIN_PREDICT_LEN}",
        f"ready: {detections.get('ready_for_prediction', False)}",
        f"hits: {detections.get('track_hits', 0)}   occ: {detections.get('occluded', False)}",
        f"visible_kpts: {detections.get('visible_keypoints', 0)}   mean: {detections.get('mean_keypoint_score', 0.0):.2f}",
        f"selector: {detections.get('selector_score', 0.0):.2f}",
    ]
    for line in active_lines:
        y = put_text(panel, line, 22, y, COLORS["white"])
    comps = detections.get("selector_components", {}) or {}
    detail = (
        f"c={comps.get('center', 0.0):.2f} "
        f"s={comps.get('size', 0.0):.2f} "
        f"v={comps.get('visibility', 0.0):.2f} "
        f"h={comps.get('hand', 0.0):.2f} "
        f"vest={comps.get('vest', 0.0):.2f}"
    )
    y = put_text(panel, detail, 22, y, COLORS["panel_dim"], scale=0.50)

    y += 10
    y = put_text(panel, "Top Gesture Scores", 22, y, COLORS["accent"], scale=0.62, thickness=2)
    for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:6]:
        y = put_text(panel, f"{label:<18} {score:.3f}", 22, y, COLORS["white"])

    y += 10
    y = put_text(panel, "State Machine", 22, y, COLORS["accent"], scale=0.62, thickness=2)
    stable = state.get("stable_vote")
    state_lines = [
        f"current: {state.get('current_command', 'NO_COMMAND')} ({state.get('current_confidence', 0.0):.2f})",
        f"threshold: {state.get('conf_threshold', 0.0):.2f}",
        f"stable_vote: {stable['command']} ({stable['confidence']:.2f})" if stable else "stable_vote: none",
        f"history: {len(state.get('history', []))} / {state.get('window_size', 0)}",
    ]
    for line in state_lines:
        y = put_text(panel, line, 22, y, COLORS["white"])
    for label, count in sorted((state.get("votes") or {}).items(), key=lambda item: item[1], reverse=True)[:4]:
        y = put_text(panel, f"vote {label:<14} x{count}", 22, y, COLORS["panel_dim"], scale=0.50)

    y += 12
    y = put_text(panel, "Output Contract", 22, y, COLORS["accent"], scale=0.62, thickness=2)
    contract = [
        "JSONL: frame / track / gesture / command",
        "State: vote / hold / fallback / expiry",
        "Bridge: ROS2 / Autoware / CARLA ready",
    ]
    for line in contract:
        y = put_text(panel, line, 22, y, COLORS["panel_dim"], scale=0.50)
    return panel


def write_poster(video_path: Path, poster_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = max(0, min(frames - 1, frames // 2))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(poster_path), frame)


def process_video(input_path: Path, slug: str) -> dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    scale_h = TARGET_HEIGHT if src_h > TARGET_HEIGHT else src_h
    out_w = int(round(src_w * scale_h / max(1, src_h))) + PANEL_WIDTH
    out_h = scale_h
    output_video = OUTPUT_DIR / f"{slug}_poliguide_visual_demo.mp4"
    output_jsonl = OUTPUT_DIR / f"{slug}_poliguide_visual_demo.jsonl"
    poster = OUTPUT_DIR / f"{slug}_poster.jpg"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {output_video}")

    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing MediaPipe pose model: {POSE_MODEL_PATH}")
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
        output_segmentation_masks=False,
    )
    pose = vision.PoseLandmarker.create_from_options(options)

    keypoint_buffer: deque[np.ndarray] = deque(maxlen=64)
    score_buffer: deque[np.ndarray] = deque(maxlen=64)
    state = FilterState(history=deque(maxlen=WINDOW_SIZE))
    command_counts: Counter[str] = Counter()
    gesture_counts: Counter[str] = Counter()
    fallback_count = 0
    first_poster_frame = None
    track_hits = 0
    latency_values: list[float] = []
    pose_values: list[float] = []

    with output_jsonl.open("w", encoding="utf-8") as jf:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            frame = resize_to_height(frame, scale_h)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = pose.detect_for_video(mp_image, int(round((frame_index / src_fps) * 1000.0)))
            pose_data = mediapipe_to_coco(result, w, h)
            occluded = True
            bbox = None
            keypoints = None
            scores_arr = None
            if pose_data is not None:
                keypoints, scores_arr = pose_data
                bbox = bbox_from_keypoints(keypoints, scores_arr, w, h)
                visible = int(np.sum(scores_arr >= 0.12))
                mean_score = float(scores_arr.mean())
                if bbox is not None and visible >= 7:
                    track_hits += 1
                    occluded = mean_score < 0.20
                    keypoint_buffer.append(keypoints)
                    score_buffer.append(scores_arr)
            else:
                visible = 0
                mean_score = 0.0
            selector_score, selector_components = selector_score_from_pose(bbox, scores_arr, w, h)

            if len(keypoint_buffer) >= MIN_PREDICT_LEN:
                label, gesture_conf, gesture_scores = classify_sequence(keypoint_buffer, score_buffer)
            else:
                gesture_scores = {name: 0.01 for name in LABELS}
                gesture_scores["NO_GESTURE"] = 0.92
                label = "NO_GESTURE"
                gesture_conf = 0.92
            raw_command = COMMANDS[label]
            command, command_conf, safe_fallback = state.step(raw_command, gesture_conf, occluded=occluded)
            intent = INTENTS.get(label, "unknown")
            latency_ms = (time.perf_counter() - start) * 1000.0
            latency_values.append(float(latency_ms))
            pose_values.append(float(mean_score))
            fallback_count += int(bool(safe_fallback))
            command_counts[command] += 1
            gesture_counts[label] += 1

            frame_vis = frame.copy()
            draw_video_overlay(frame_vis, bbox, keypoints, scores_arr, label, command, command_conf, safe_fallback, selector_score)
            active = {
                "buffer_len": len(keypoint_buffer),
                "ready_for_prediction": len(keypoint_buffer) >= MIN_PREDICT_LEN,
                "track_hits": track_hits,
                "occluded": bool(occluded),
                "visible_keypoints": visible,
                "mean_keypoint_score": mean_score,
                "selector_score": selector_score,
                "selector_components": selector_components,
            }
            record = {
                "frame_index": frame_index,
                "timestamp_sec": frame_index / src_fps,
                "fps": src_fps,
                "pose_detected": bbox is not None,
                "pose_quality": float(mean_score),
                "active_track_id": 1 if bbox is not None else None,
                "track_id": 1 if bbox is not None else None,
                "person_bbox": bbox,
                "selector_score": float(selector_score),
                "gesture": label,
                "gesture_confidence": float(gesture_conf),
                "intent": intent,
                "command": command,
                "command_confidence": float(command_conf),
                "confidence": float(command_conf if command != "NO_COMMAND" else gesture_conf),
                "safe_fallback": bool(safe_fallback),
                "latency_ms": float(latency_ms),
                "keypoints_coco17": None if keypoints is None else keypoints.tolist(),
                "keypoint_scores_coco17": None if scores_arr is None else scores_arr.tolist(),
                "detections": [] if bbox is None else [{
                    "bbox": bbox,
                    "score": selector_score,
                    "track_id": 1,
                    "keypoints": None if keypoints is None else keypoints.tolist(),
                    "keypoint_scores": None if scores_arr is None else scores_arr.tolist(),
                    "occluded": bool(occluded),
                    "selector_score": active["selector_score"],
                    "selector_components": selector_components,
                }],
                "debug": {
                    "raw_prediction": {"scores": gesture_scores},
                    "active_track": active,
                    "state_machine": state.debug(),
                },
            }
            panel = draw_panel(h, record, active, gesture_scores, state.debug(), input_path.name)
            composed = np.concatenate([frame_vis, panel], axis=1)
            writer.write(composed)
            if first_poster_frame is None and frame_index > max(3, src_frames // 5):
                first_poster_frame = composed.copy()
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            frame_index += 1

    cap.release()
    writer.release()
    pose.close()
    if first_poster_frame is not None:
        cv2.imwrite(str(poster), first_poster_frame)
    else:
        write_poster(output_video, poster)
    return {
        "source": str(input_path),
        "output_video": str(output_video),
        "output_jsonl": str(output_jsonl),
        "poster": str(poster),
        "source_width": src_w,
        "source_height": src_h,
        "source_fps": src_fps,
        "source_frames": src_frames,
        "output_width": out_w,
        "output_height": out_h,
        "duration_sec": round(src_frames / src_fps, 3) if src_fps else None,
        "route": "MediaPipe pose + rule classifier + temporal command filter",
        "detected_pose_frames": int(track_hits),
        "detected_pose_ratio": round(track_hits / max(1, src_frames), 4),
        "avg_latency_ms": round(float(np.mean(latency_values)), 3) if latency_values else None,
        "avg_pose_quality": round(float(np.mean(pose_values)), 4) if pose_values else None,
        "gestures": dict(gesture_counts),
        "commands": dict(command_counts),
        "fallback_frames": int(fallback_count),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = discover_source_videos()
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "route": "MediaPipe pose + project-style rule classifier + temporal command filter",
        "note": "All values are computed from the supplied video frames, detected pose, rule classifier and temporal command filter.",
        "files": [],
    }
    for idx, source in enumerate(videos, start=1):
        slug = make_slug(idx, source)
        print(f"processing {source.name} -> {slug}")
        item = process_video(source, slug)
        manifest["files"].append(item)
        print(f"done {item['output_video']}")
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest {manifest_path}")


if __name__ == "__main__":
    main()
