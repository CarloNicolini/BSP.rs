#!/bin/sh
# Select an available native GPU build; otherwise retain the CPU build.
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] && \
    command -v xcrun >/dev/null 2>&1 && xcrun --find clang >/dev/null 2>&1; then
    echo metal
elif [ "$(uname -s)" = Linux ] && \
    command -v "${NVCC:-nvcc}" >/dev/null 2>&1 && \
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && \
    [ -n "$(nvidia-smi -L 2>/dev/null)" ]; then
    echo cuda
else
    echo cpu
fi
