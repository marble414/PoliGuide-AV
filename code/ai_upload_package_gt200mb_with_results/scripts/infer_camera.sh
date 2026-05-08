#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
CONFIG="${1:-configs/deploy_mediapipe_cpu.yaml}"
CAMERA_ID="${2:-0}"
python -m tpgr.cli.infer_camera --config "$CONFIG" --camera-id "$CAMERA_ID"
