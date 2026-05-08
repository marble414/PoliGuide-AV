from __future__ import annotations

from typing import List

import cv2
import numpy as np

from tpgr.data.features import COCO17_EDGES

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
WHOLEBODY_FOOT_EDGES = [
    (15, 17), (15, 18), (15, 19),
    (16, 20), (16, 21), (16, 22),
]

COLORS = {
    "active": (0, 255, 0),
    "inactive": (255, 180, 0),
    "text": (255, 255, 255),
    "warning": (0, 0, 255),
    "panel_bg": (24, 24, 24),
    "panel_dim": (170, 170, 170),
    "accent": (0, 220, 255),
}


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, scores: np.ndarray | None = None, active: bool = False) -> np.ndarray:
    canvas = frame
    color = COLORS["active"] if active else COLORS["inactive"]
    scores = np.asarray(scores if scores is not None else np.ones((len(keypoints),)), dtype=np.float32)
    edges = list(COCO17_EDGES)
    if len(keypoints) >= 133:
        edges.extend(WHOLEBODY_FOOT_EDGES)
        edges.extend([(91 + i, 91 + j) for i, j in HAND_EDGES])
        edges.extend([(112 + i, 112 + j) for i, j in HAND_EDGES])
    for i, j in edges:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if scores[i] < 0.05 or scores[j] < 0.05:
            continue
        p1 = tuple(np.asarray(keypoints[i, :2], dtype=np.int32))
        p2 = tuple(np.asarray(keypoints[j, :2], dtype=np.int32))
        cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
    for idx, pt in enumerate(keypoints):
        if idx < len(scores) and scores[idx] < 0.05:
            continue
        cv2.circle(canvas, tuple(np.asarray(pt[:2], dtype=np.int32)), 3, color, -1, cv2.LINE_AA)
    return canvas


def draw_detections(frame: np.ndarray, detections: List[dict], active_track_id: int | None, debug: bool = False) -> np.ndarray:
    canvas = frame
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        track_id = det.get("track_id")
        active = track_id == active_track_id
        color = COLORS["active"] if active else COLORS["inactive"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if active else 2)
        label = f"id={track_id} sel={det.get('selector_score', 0.0):.2f}"
        if debug:
            label += f" hit={det.get('track_hits', 0)} occ={int(bool(det.get('occluded', False)))}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if "keypoints" in det:
            draw_skeleton(
                canvas,
                np.asarray(det["keypoints"], dtype=np.float32),
                np.asarray(det.get("keypoint_scores", np.ones(len(det["keypoints"]))), dtype=np.float32),
                active=active,
            )
    return canvas


def _put_text(canvas: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.54, thickness: int = 1) -> int:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return y + int(22 * scale / 0.54)


def _draw_basic_panel(canvas: np.ndarray, result, x: int = 12, y: int = 24) -> np.ndarray:
    lines = [
        f"track: {result.active_track_id}",
        f"gesture: {result.gesture} ({result.gesture_confidence:.2f})",
        f"intent: {result.intent}",
        f"command: {result.command} ({result.command_confidence:.2f})",
        f"fallback: {result.safe_fallback}",
        f"latency: {result.latency_ms:.1f} ms",
    ]
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        y += 24
    return canvas


def draw_result_panel(
    frame: np.ndarray,
    result,
    debug: bool = False,
    topk: int = 5,
    panel_width: int = 440,
) -> np.ndarray:
    if not debug:
        return _draw_basic_panel(frame, result)

    panel = np.full((frame.shape[0], panel_width, 3), COLORS["panel_bg"], dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, frame.shape[0] - 1), (70, 70, 70), 1)

    y = 28
    y = _put_text(panel, "TPGR Debug Dashboard", 16, y, COLORS["accent"], scale=0.7, thickness=2)
    y += 8
    summary_lines = [
        f"frame: {result.frame_index}   t={result.timestamp_sec:.2f}s",
        f"fps: {result.fps:.2f}   latency: {result.latency_ms:.1f} ms",
        f"track: {result.active_track_id}",
        f"gesture: {result.gesture} ({result.gesture_confidence:.2f})",
        f"intent: {result.intent}",
        f"command: {result.command} ({result.command_confidence:.2f})",
        f"safe_fallback: {result.safe_fallback}",
    ]
    for line in summary_lines:
        y = _put_text(panel, line, 16, y, COLORS["text"])

    debug_info = getattr(result, "debug", {}) or {}
    active = debug_info.get("active_track") or {}
    if y < panel.shape[0] - 24:
        y += 10
        y = _put_text(panel, "Active Track", 16, y, COLORS["accent"], scale=0.62, thickness=2)
        active_lines = [
            f"buffer: {active.get('buffer_len', 0)} / min_pred {debug_info.get('min_predict_len', 0)}",
            f"ready: {active.get('ready_for_prediction', False)}",
            f"hits: {active.get('track_hits', 0)}   occ: {active.get('occluded', False)}",
            f"visible_kpts: {active.get('visible_keypoints', 0)}   mean: {active.get('mean_keypoint_score', 0.0):.2f}",
            f"selector: {active.get('selector_score', 0.0):.2f}",
        ]
        for line in active_lines:
            if y >= panel.shape[0] - 16:
                break
            y = _put_text(panel, line, 16, y, COLORS["text"])

    raw_prediction = debug_info.get("raw_prediction") or {}
    scores = raw_prediction.get("scores", {})
    if scores and y < panel.shape[0] - 24:
        y += 10
        y = _put_text(panel, "Top Gesture Scores", 16, y, COLORS["accent"], scale=0.62, thickness=2)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max(1, topk)]
        for label, score in ranked:
            if y >= panel.shape[0] - 16:
                break
            y = _put_text(panel, f"{label:<18} {score:.3f}", 16, y, COLORS["text"])

    state = debug_info.get("state_machine") or {}
    if y < panel.shape[0] - 24:
        y += 10
        y = _put_text(panel, "State Machine", 16, y, COLORS["accent"], scale=0.62, thickness=2)
        stable_vote = state.get("stable_vote")
        state_lines = [
            f"current: {state.get('current_command', 'NO_COMMAND')} ({state.get('current_confidence', 0.0):.2f})",
            f"threshold: {state.get('conf_threshold', 0.0):.2f}",
            f"stable_vote: {stable_vote['command']} ({stable_vote['confidence']:.2f})" if stable_vote else "stable_vote: none",
            f"history: {len(state.get('history', []))} / {state.get('window_size', 0)}",
        ]
        for line in state_lines:
            if y >= panel.shape[0] - 16:
                break
            y = _put_text(panel, line, 16, y, COLORS["text"])

        votes = state.get("votes", {})
        if votes and y < panel.shape[0] - 16:
            vote_items = sorted(votes.items(), key=lambda item: item[1], reverse=True)[:3]
            for label, count in vote_items:
                if y >= panel.shape[0] - 16:
                    break
                y = _put_text(panel, f"vote {label:<14} x{count}", 16, y, COLORS["panel_dim"])

    tracks = debug_info.get("tracks", [])
    if tracks and y < panel.shape[0] - 24:
        y += 10
        y = _put_text(panel, "Track Ranking", 16, y, COLORS["accent"], scale=0.62, thickness=2)
        for item in tracks[:3]:
            if y >= panel.shape[0] - 16:
                break
            y = _put_text(
                panel,
                f"id={item.get('track_id')} sel={item.get('selector_score', 0.0):.2f} hit={item.get('track_hits', 0)} vis={item.get('visible_keypoints', 0)}",
                16,
                y,
                COLORS["text"],
            )
            comps = item.get("selector_components", {})
            detail = (
                f"c={comps.get('center', 0.0):.2f} "
                f"s={comps.get('size', 0.0):.2f} "
                f"v={comps.get('visibility', 0.0):.2f} "
                f"h={comps.get('hand', 0.0):.2f} "
                f"a={comps.get('track_age', 0.0):.2f} "
                f"st={comps.get('stability', 0.0):.2f} "
                f"vest={comps.get('vest', 0.0):.2f}"
            )
            if y >= panel.shape[0] - 16:
                break
            y = _put_text(panel, detail, 16, y, COLORS["panel_dim"], scale=0.5)

    evaluation = debug_info.get("evaluation") or {}
    if evaluation and y < panel.shape[0] - 24:
        y += 10
        y = _put_text(panel, "Evaluation", 16, y, COLORS["accent"], scale=0.62, thickness=2)
        eval_lines = [
            f"clip: {evaluation.get('clip_index', 0)} / {evaluation.get('num_clips', 0)}",
            f"sample: {evaluation.get('sample_id', '-')}",
            f"gt: {evaluation.get('ground_truth', '-')}",
            f"frame_pred: {evaluation.get('frame_prediction', '-')}",
            f"clip_pred: {evaluation.get('clip_prediction', '-')}",
            f"clip_correct: {evaluation.get('clip_correct', '-')}",
            f"clip_frames: {evaluation.get('clip_frames', 0)}",
            f"running_acc: {evaluation.get('running_accuracy', 0.0):.3f}",
        ]
        for line in eval_lines:
            if y >= panel.shape[0] - 16:
                break
            y = _put_text(panel, line, 16, y, COLORS["text"])

    return np.concatenate([frame, panel], axis=1)
