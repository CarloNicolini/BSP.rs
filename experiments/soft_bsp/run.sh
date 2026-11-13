#!/bin/sh
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 {smoke|pilot|heldout|diagnostic} OUTPUT_DIRECTORY [WORKERS]" >&2
    exit 2
fi
case "$1" in
    smoke) expected=2 ;;
    pilot) expected=8 ;;
    heldout|diagnostic) expected=40 ;;
    *) echo "Unknown campaign mode: $1" >&2; exit 2 ;;
esac
mkdir -p "$2"
output=$(cd "$2" && pwd)
cd "$(dirname "$0")/../.."
export SOFT_BSP_MODE="$1" SOFT_BSP_OUT="$output" SOFT_BSP_WORKERS="${3:-1}"
export SOFT_BSP_MINISAT="${SOFT_BSP_MINISAT:-$(command -v minisat || true)}"
if [ -z "$SOFT_BSP_MINISAT" ]; then
    unset SOFT_BSP_MINISAT
fi
{
    git rev-parse HEAD
    rustc -Vv
    uname -smr
    printf 'mode=%s\nworkers=%s\nminisat=%s\n' "$SOFT_BSP_MODE" "$SOFT_BSP_WORKERS" "${SOFT_BSP_MINISAT:-disabled}"
} > "$output/environment.txt"
cargo test --locked --release --lib soft_bsp_campaign -- --ignored --nocapture
python3 experiments/soft_bsp/analyze.py "$output" --expected "$expected" > "$output/summary.json"
