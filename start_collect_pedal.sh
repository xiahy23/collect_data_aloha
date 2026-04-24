#!/bin/bash
set -euo pipefail

DATASET_DIR=~/data
TASK_NAME=my_task
MAX_TIMESTEPS=9000
EPISODE_IDX=-1
SESSION=collect_pedal

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

tmux new-window -t "$SESSION" -n "piper_master"
tmux send-keys -t "$SESSION:1" "$SETUP_CMD && \
cd ~/cobot_magic/Piper_ros_private-ros-noetic/ && \
bash can_muti_activate.sh && \
cd ~/cobot_magic/piper_ws/ && \
source devel/setup.bash && \
roslaunch piper start_master_aloha.launch" Enter

echo "[INFO] waiting for master-arm topics..."
until bash -lc "$SETUP_CMD && rostopic list | grep -q '^/master/joint_left$'"; do
    sleep 1
done
until bash -lc "$SETUP_CMD && rostopic list | grep -q '^/master/joint_right$'"; do
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
cd ~/cobot_magic/collect_data/ && \
python collect_data_master_with_cam_pedal.py \
    --dataset_dir ${DATASET_DIR} \
    --task_name ${TASK_NAME} \
    --max_timesteps ${MAX_TIMESTEPS} \
    --episode_idx ${EPISODE_IDX}" Enter

tmux select-window -t "$SESSION:3"
tmux attach-session -t "$SESSION"
