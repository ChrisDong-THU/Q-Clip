#!/bin/bash
start_gpu_alive() {
    python keep_gpu_alive.py \
        > /dev/null 2>&1 &
    GPU_KEEPALIVE_PID=$!
    trap 'kill ${GPU_KEEPALIVE_PID} 2>/dev/null' EXIT INT TERM
}