#!/bin/bash
set -euo pipefail

# UI + pedal data-collection pipeline launcher.
# Mirrors start_collect_pedal.sh but launches the new pipeline collector.

source /home/agilex/miniconda3/etc/profile.d/conda.sh
conda activate aloha
cd ~/cobot_magic/collect_data/piper_sdk_demo

python3 master_arm_control.py --can can_left --to-master
python3 master_arm_control.py --can can_right --to-master
read -rp $'\n[INFO] 请手动断电重启两臂，完成后按 Enter 键继续...\n'

DATASET_DIR=${DATASET_DIR:-~/data}
TASK_NAME=${TASK_NAME:-aloha_pipeline}
MAX_TIMESTEPS=${MAX_TIMESTEPS:-9000}
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
cd ~/cobot_magic/collect_data/scripts && \
python collect_data_pipeline.py \
    --dataset_dir ${DATASET_DIR} \
    --task_name ${TASK_NAME} \
    --max_timesteps ${MAX_TIMESTEPS}" Enter

tmux select-window -t "$SESSION:3"
tmux attach-session -t "$SESSION"
