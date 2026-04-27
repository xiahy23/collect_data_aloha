#!/bin/bash
set -euo pipefail

SESSION=camera_preview

SETUP_CMD="source /home/agilex/miniconda3/etc/profile.d/conda.sh; \
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash; \
export ROS_MASTER_URI=http://localhost:11311; \
export ROS_HOSTNAME=localhost; \
unset ROS_IP; \
conda activate aloha"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n "roscore"
tmux send-keys -t "$SESSION:0" "$SETUP_CMD && roscore" Enter

echo "[INFO] waiting for roscore..."
until bash -lc "$SETUP_CMD && rostopic list >/dev/null 2>&1"; do
    sleep 1
done

tmux new-window -t "$SESSION" -n "camera"
tmux send-keys -t "$SESSION:1" "$SETUP_CMD && \
cd ~/cobot_magic/camera_ws/ && \
source devel/setup.bash && \
roslaunch astra_camera multi_camera.launch" Enter

echo "[INFO] waiting for camera publishers..."
until bash -lc "$SETUP_CMD && rostopic info /camera_f/color/image_raw 2>/dev/null | grep -qv 'Publishers: None'"; do
    sleep 1
done
until bash -lc "$SETUP_CMD && rostopic info /camera_l/color/image_raw 2>/dev/null | grep -qv 'Publishers: None'"; do
    sleep 1
done
until bash -lc "$SETUP_CMD && rostopic info /camera_r/color/image_raw 2>/dev/null | grep -qv 'Publishers: None'"; do
    sleep 1
done

tmux new-window -t "$SESSION" -n "preview"
tmux send-keys -t "$SESSION:2" "$SETUP_CMD && \
cd ~/cobot_magic/collect_data/scripts && \
python preview_three_cameras_web.py" Enter

echo "[INFO] preview ready: http://127.0.0.1:8000/"
echo "[INFO] tmux session: $SESSION"

tmux select-window -t "$SESSION:2"
tmux attach-session -t "$SESSION"
