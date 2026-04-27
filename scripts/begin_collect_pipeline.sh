#!/bin/bash
set -euo pipefail

# UI + pedal data-collection pipeline launcher.
# Mirrors start_collect_pedal.sh but launches the new pipeline collector.

source /home/agilex/miniconda3/etc/profile.d/conda.sh
conda activate aloha
cd ~/cobot_magic/collect_data/piper_sdk_demo

DATASET_DIR=${DATASET_DIR:-~/data}
TASK_NAME=${TASK_NAME:-aloha_pipeline}
MAX_TIMESTEPS=${MAX_TIMESTEPS:-9000}
LAMP_PORT=${LAMP_PORT:-/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0}
SESSION=${SESSION:-collect_pipeline}

SETUP_CMD="source /home/agilex/miniconda3/etc/profile.d/conda.sh; \
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash; \
export ROS_MASTER_URI=http://localhost:11311; \
export ROS_HOSTNAME=localhost;  \
conda activate aloha"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n "roscore"
tmux send-keys -t "$SESSION:0" "$SETUP_CMD && roscore" Enter

echo "[INFO] waiting for roscore..."
until bash -lc "$SETUP_CMD && rostopic list >/dev/null 2>&1"; do
    sleep 1
done

tmux new-window -t "$SESSION" -n "piper_ms"
# CAN 重新配置后需要 2-3s 稳定; piper_sdk 首次 open CAN 偶发 ConnectionError -> 自动重试
tmux send-keys -t "$SESSION:1" "$SETUP_CMD && \
cd ~/cobot_magic/Piper_ros_private-ros-noetic/ && \
bash can_config.sh && \
sleep 3 && \
source devel/setup.bash && \
until roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false; do \
    echo '[WARN] roslaunch exited, retry in 3s...'; sleep 3; \
done" Enter

echo "[INFO] waiting for master-arm topics..."
until bash -lc "$SETUP_CMD && rostopic list | grep -q '^/master/joint_left$'"; do
    sleep 1
done
until bash -lc "$SETUP_CMD && rostopic list | grep -q '^/master/joint_right$'"; do
    sleep 1
done

echo "[INFO] waiting for puppet-arm topics..."
until bash -lc "$SETUP_CMD && rostopic list | grep -q '^/puppet/joint_left$'"; do
    sleep 1
done
until bash -lc "$SETUP_CMD && rostopic list | grep -q '^/puppet/joint_right$'"; do
    sleep 1
done

tmux new-window -t "$SESSION" -n "camera"
tmux send-keys -t "$SESSION:2" "$SETUP_CMD && \
cd ~/cobot_magic/camera_ws/ && \
source devel/setup.bash && \
roslaunch astra_camera multi_camera.launch" Enter

echo "[INFO] waiting for camera topics..."
until bash -lc "$SETUP_CMD && rostopic list | grep -q '^/camera_f/color/image_raw$'"; do
    sleep 1
done

tmux new-window -t "$SESSION" -n "collect"
tmux send-keys -t "$SESSION:3" "$SETUP_CMD && \
cd ~/cobot_magic/collect_data/scripts && \
python collect_data_pipeline.py \
    --dataset_dir ${DATASET_DIR} \
    --task_name ${TASK_NAME} \
    --lamp_port ${LAMP_PORT} \
    --max_timesteps ${MAX_TIMESTEPS}" Enter

tmux select-window -t "$SESSION:3"
tmux attach-session -t "$SESSION"
