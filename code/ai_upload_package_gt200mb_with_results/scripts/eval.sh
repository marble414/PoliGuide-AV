#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
CONFIG="${1:-configs/toy_tcn.yaml}"
CKPT="${2:-runs/toy_tcn/best.pt}"
OUT="${3:-runs/toy_tcn/eval_test}"
python -m tpgr.cli.evaluate --config "$CONFIG" --checkpoint "$CKPT" --split test --output-dir "$OUT"
