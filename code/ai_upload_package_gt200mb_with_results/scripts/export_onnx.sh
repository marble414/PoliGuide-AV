#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
CONFIG="${1:-configs/toy_tcn.yaml}"
CKPT="${2:-runs/toy_tcn/best.pt}"
OUT="${3:-exports/toy_tcn.onnx}"
python -m tpgr.cli.export_onnx --config "$CONFIG" --checkpoint "$CKPT" --output "$OUT"
