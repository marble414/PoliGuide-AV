#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
CONFIG="${1:-configs/deploy_precomputed_demo.yaml}"
INPUT="${2:-data/processed/toy/demo/toy_demo.mp4}"
OUTPUT_VIDEO="${3:-outputs/demo.mp4}"
OUTPUT_JSONL="${4:-outputs/demo.jsonl}"
python -m tpgr.cli.infer_video --config "$CONFIG" --input "$INPUT" --output-video "$OUTPUT_VIDEO" --output-jsonl "$OUTPUT_JSONL"
