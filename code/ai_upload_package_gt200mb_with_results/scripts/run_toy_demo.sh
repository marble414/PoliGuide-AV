#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python tools/generate_toy_dataset.py --output-dir data/processed/toy
python -m tpgr.cli.train --config configs/toy_tcn.yaml
python -m tpgr.cli.evaluate --config configs/toy_tcn.yaml --checkpoint runs/toy_tcn/best.pt --split test --output-dir runs/toy_tcn/eval_test
python -m tpgr.cli.infer_video \
  --config configs/deploy_toy_tcn_demo.yaml \
  --input data/processed/toy/demo/toy_demo.mp4 \
  --output-video outputs/toy_tcn_demo.mp4 \
  --output-jsonl outputs/toy_tcn_demo.jsonl
