#!/bin/sh
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 {development|heldout|replay-flips|replay-sp} OUTPUT_DIRECTORY [WORKERS]" >&2
    exit 2
fi
case "$1" in
    development|heldout|replay-flips|replay-sp) ;;
    *) echo "Unknown campaign mode: $1" >&2; exit 2 ;;
esac
mkdir -p "$2"
output=$(cd "$2" && pwd)
cd "$(dirname "$0")/../.."
export SOFT_BSP_MODE="$1" SOFT_BSP_OUT="$output" SOFT_BSP_WORKERS="${3:-1}"
{
    git rev-parse HEAD
    rustc -Vv
    uname -smr
    printf 'mode=%s\nworkers=%s\n' "$SOFT_BSP_MODE" "$SOFT_BSP_WORKERS"
} > "$output/environment-$1.txt"
case "$1" in
    replay-*)
        SOFT_BSP_REPLAY="${1#replay-}" cargo test --locked --release --lib net_bsp_resource_replay -- --ignored --nocapture
        ;;
    *)
        cargo test --locked --release --lib net_bsp_campaign -- --ignored --nocapture
        python3 experiments/soft_bsp/net_analysis.py "$1" "$output" > "$output/summary.json"
        ;;
esac
