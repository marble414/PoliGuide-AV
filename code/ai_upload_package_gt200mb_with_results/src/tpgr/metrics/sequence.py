from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def edit_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    dp = [[0] * (len(seq_b) + 1) for _ in range(len(seq_a) + 1)]
    for i in range(len(seq_a) + 1):
        dp[i][0] = i
    for j in range(len(seq_b) + 1):
        dp[0][j] = j
    for i in range(1, len(seq_a) + 1):
        for j in range(1, len(seq_b) + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def sequence_exact_match(pred_sequences: Iterable[Sequence[str]], gt_sequences: Iterable[Sequence[str]]) -> float:
    pred_sequences = list(pred_sequences)
    gt_sequences = list(gt_sequences)
    if not pred_sequences:
        return 0.0
    correct = sum(1 for pred, gt in zip(pred_sequences, gt_sequences) if list(pred) == list(gt))
    return correct / len(pred_sequences)


def compress_commands(seq: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in seq:
        if not out or item != out[-1]:
            out.append(item)
    return out


def sequence_edit_similarity(pred: Sequence[str], gt: Sequence[str]) -> float:
    pred_c = compress_commands(pred)
    gt_c = compress_commands(gt)
    if not gt_c:
        return 1.0 if not pred_c else 0.0
    ed = edit_distance(pred_c, gt_c)
    return 1.0 - ed / max(len(gt_c), 1)


def command_switch_delay(gt: Sequence[str], pred: Sequence[str]) -> float:
    change_points = [i for i in range(1, len(gt)) if gt[i] != gt[i - 1]]
    if not change_points:
        return 0.0
    delays = []
    for idx in change_points:
        target = gt[idx]
        found = None
        for j in range(idx, min(len(pred), idx + 60)):
            if pred[j] == target:
                found = j
                break
        delays.append((found - idx) if found is not None else 60)
    return sum(delays) / len(delays)


def stable_output_latency(
    gt: Sequence[str],
    pred: Sequence[str],
    stable_window: int = 3,
    background_label: str = "NO_COMMAND",
) -> float:
    onsets = [i for i in range(len(gt)) if gt[i] != background_label and (i == 0 or gt[i - 1] != gt[i])]
    if not onsets:
        return 0.0
    delays = []
    for onset in onsets:
        target = gt[onset]
        found = None
        for j in range(onset, len(pred) - stable_window + 1):
            if all(item == target for item in pred[j:j + stable_window]):
                found = j
                break
        delays.append((found - onset) if found is not None else len(pred) - onset)
    return sum(delays) / len(delays)


def classwise_jaccard(pred: Sequence[str], gt: Sequence[str], classes: Sequence[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for name in classes:
        pred_mask = [item == name for item in pred]
        gt_mask = [item == name for item in gt]
        intersection = sum(1 for p, g in zip(pred_mask, gt_mask) if p and g)
        union = sum(1 for p, g in zip(pred_mask, gt_mask) if p or g)
        scores[name] = float(intersection / union) if union > 0 else 0.0
    return scores


def macro_jaccard(pred: Sequence[str], gt: Sequence[str], classes: Sequence[str]) -> float:
    scores = classwise_jaccard(pred, gt, classes)
    if not scores:
        return 0.0
    return float(sum(scores.values()) / len(scores))
