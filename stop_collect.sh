#!/bin/bash

set -euo pipefail
SESSION=collect

echo "[INFO] 正在关闭各节点..."

# 按顺序发送 Ctrl+C 停止各窗口进程
for win in 3 2 1 0; do
    tmux send-keys -t $SESSION:$win C-c 2>/dev/null || true
    sleep 1
done

echo "[INFO] 等待节点退出..."
sleep 3

# 关闭整个 session
tmux kill-session -t $SESSION 2>/dev/null && \
    echo "[INFO] tmux session '$SESSION' 已关闭" || \
    echo "[WARN] session '$SESSION' 不存在或已关闭"

echo "[INFO] 停止完成"