#!/usr/bin/env bash
set -euo pipefail

python experiments/run_suite.py \
  --suite experiments/suites/validation_methodology.yaml \
  "$@"
