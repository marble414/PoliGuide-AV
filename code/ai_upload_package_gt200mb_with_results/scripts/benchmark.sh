#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
CONFIG="${1:-configs/toy_tcn.yaml}"
MODE="${2:-model}"
python -m tpgr.cli.benchmark --config "$CONFIG" --mode "$MODE" --iterations 100
