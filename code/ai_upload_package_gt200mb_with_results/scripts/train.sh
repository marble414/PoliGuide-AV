#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python -m tpgr.cli.train --config "${1:-configs/toy_tcn.yaml}"
