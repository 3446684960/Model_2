#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${HOME}/autodl-pub/Masico}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/runs/real_world}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
MODEL="${1:-all}"
STAGE="${2:-all}"

if [[ "$MODEL" == "--help" || "$MODEL" == "-h" ]]; then
    printf 'Usage: bash run_realworld.sh [all|MODEL] [all|train|generate|test]\n'
    printf 'Use python train.py --list-models to list the 15 available models.\n'
    exit 0
fi
if (( $# > 2 )); then
    printf 'Expected a model and a stage. See --help.\n' >&2
    exit 2
fi
case "$STAGE" in
    all|train|generate|test) ;;
    *) printf 'Unknown stage %s\n' "$STAGE" >&2; exit 2 ;;
esac

if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
    "$PYTHON_BIN" -u "${PROJECT_DIR}/train.py" \
        --models "$MODEL" --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT" --device "$GPU"
fi
if [[ "$STAGE" == "generate" || "$STAGE" == "all" ]]; then
    "$PYTHON_BIN" -u "${PROJECT_DIR}/test.py" \
        --models "$MODEL" --stage generate --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT" --device "$GPU"
fi
if [[ "$STAGE" == "test" || "$STAGE" == "all" ]]; then
    "$PYTHON_BIN" -u "${PROJECT_DIR}/test.py" \
        --models "$MODEL" --stage evaluate \
        --output-root "$OUTPUT_ROOT" --device "$GPU"
fi
