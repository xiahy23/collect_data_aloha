#!/bin/bash
set -euo pipefail

SESSION=camera_preview

echo "[INFO] stopping camera preview session..."

for win in 2 1 0; do
    tmux send-keys -t "$SESSION:$win" C-c 2>/dev/null || true
    sleep 1
done

sleep 2
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "[INFO] stopped"
