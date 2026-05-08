from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

from tpgr.data.labels import safe_hold_command


class TemporalCommandFilter:
    def __init__(
        self,
        window_size: int = 7,
        min_consensus: int = 5,
        conf_threshold: float = 0.60,
        hold_frames: int = 8,
        expiry_frames: int = 20,
    ) -> None:
        self.window_size = window_size
        self.min_consensus = min_consensus
        self.conf_threshold = conf_threshold
        self.hold_frames = hold_frames
        self.expiry_frames = expiry_frames
        self.history: Deque[Tuple[str, float]] = deque(maxlen=window_size)
        self.current_command = "NO_COMMAND"
        self.current_conf = 0.0
        self.last_update_frame = -10**9
        self.frame_index = -1
        self.last_safe_fallback = False

    def _stable_vote(self) -> tuple[str, float] | None:
        votes = {}
        confs = {}
        for label, conf in self.history:
            votes[label] = votes.get(label, 0) + 1
            confs.setdefault(label, []).append(conf)
        if not votes:
            return None
        best_label = max(votes, key=votes.get)
        if votes[best_label] < self.min_consensus:
            return None
        mean_conf = sum(confs[best_label]) / len(confs[best_label])
        if mean_conf < self.conf_threshold:
            return None
        return best_label, float(mean_conf)

    def step(self, command: str, confidence: float, occluded: bool = False) -> tuple[str, float, bool]:
        self.frame_index += 1
        safe_fallback = False

        if occluded or confidence < self.conf_threshold:
            if self.frame_index - self.last_update_frame <= self.hold_frames:
                safe_cmd = safe_hold_command(self.current_command)
                safe_fallback = safe_cmd != self.current_command
                self.current_command = safe_cmd
            elif self.frame_index - self.last_update_frame > self.expiry_frames:
                self.current_command = "NO_COMMAND"
                self.current_conf = 0.0
            self.last_safe_fallback = safe_fallback
            return self.current_command, self.current_conf, safe_fallback

        self.history.append((command, confidence))
        vote = self._stable_vote()
        if vote is None:
            if self.current_command != "NO_COMMAND" and self.frame_index - self.last_update_frame <= self.hold_frames:
                safe_cmd = safe_hold_command(self.current_command)
                safe_fallback = safe_cmd != self.current_command
                self.current_command = safe_cmd
            self.last_safe_fallback = safe_fallback
            return self.current_command, self.current_conf, safe_fallback

        stable_cmd, stable_conf = vote
        self.current_command = stable_cmd
        self.current_conf = stable_conf
        self.last_update_frame = self.frame_index
        self.last_safe_fallback = False
        return self.current_command, self.current_conf, False

    def get_debug_state(self) -> dict:
        votes = {}
        confs = {}
        history = []
        for label, conf in self.history:
            votes[label] = votes.get(label, 0) + 1
            confs.setdefault(label, []).append(float(conf))
            history.append({"command": label, "confidence": float(conf)})

        stable_vote = self._stable_vote()
        return {
            "window_size": int(self.window_size),
            "min_consensus": int(self.min_consensus),
            "conf_threshold": float(self.conf_threshold),
            "hold_frames": int(self.hold_frames),
            "expiry_frames": int(self.expiry_frames),
            "history": history,
            "votes": {label: int(count) for label, count in votes.items()},
            "vote_mean_confidence": {
                label: float(sum(values) / len(values))
                for label, values in confs.items()
            },
            "stable_vote": None if stable_vote is None else {
                "command": stable_vote[0],
                "confidence": float(stable_vote[1]),
            },
            "current_command": self.current_command,
            "current_confidence": float(self.current_conf),
            "last_update_frame": int(self.last_update_frame),
            "frame_index": int(self.frame_index),
            "last_safe_fallback": bool(self.last_safe_fallback),
        }

    def reset(self) -> None:
        self.history.clear()
        self.current_command = "NO_COMMAND"
        self.current_conf = 0.0
        self.last_update_frame = -10**9
        self.frame_index = -1
        self.last_safe_fallback = False
