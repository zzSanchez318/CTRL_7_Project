#!/usr/bin/env bash
set -euo pipefail
python3 "$(dirname "$0")/replicate_process_moments.py" "$@"
