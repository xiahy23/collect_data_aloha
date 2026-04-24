#!/bin/bash
set -euo pipefail
# ========== 可配置参数 ==========
DATASET_DIR=~/data
TASK_NAME=my_task
MAX_TIMESTEPS=500
EPISODE_IDX=0
SESSION=collect
# =================================

SETUP_CMD="source /home/agilex/miniconda3/etc/profile.d/conda.sh; \
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash; \
export ROS_MASTER_URI=http://localhost:11311; \
export ROS_HOSTNAME=localhost; \
conda activate aloha"

# 如果 session 已存在则先删除
tmux kill-session -t $SESSION 2>/dev/null || true

# 创建新 session，第一个窗口：roscore
tmux new-session -d -s $SESSION -n "roscore"
tmux send-keys -t $SESSION:0 "$SETUP_CMD && roscore" Enter

echo "[INFO] 等待 roscore 启动..."
until bash -lc "$SETUP_CMD && rostopic list >/dev/null 2>&1"; do
    sleep 1
done

# 窗口2：CAN + 主臂
tmux new-window -t $SESSION -n "piper_master"
tmux send-keys -t $SESSION:1 "$SETUP_CMD && \
cd ~/cobot_magic/Piper_ros_private-ros-noetic/ && \
bash can_muti_activate.sh && \
cd ~/cobot_magic/piper_ws/ && \
source devel/setup.bash && \
roslaunch piper start_master_aloha.launch" Enter

echo "[INFO] 等待主臂节点启动..."
until bash -lc "$SETUP_CMD && rostopic list | grep -q '^/master/joint_left$'"; do
    sleep 1
done
until bash -lc "$SETUP_CMD && rostopic list | grep -q '^/master/joint_right$'"; do
    sleep 1
done

# 窗口3：相机
tmux new-window -t $SESSION -n "camera"
tmux send-keys -t $SESSION:2 "$SETUP_CMD && \
cd ~/cobot_magic/camera_ws/ && \
source devel/setup.bash && \
roslaunch astra_camera multi_camera.launch" Enter

echo "[INFO] 等待相机节点启动..."
until bash -lc "$SETUP_CMD && rostopic list | grep -q '^/camera_f/color/image_raw$'"; do
    sleep 1
done

# 窗口4：数据采集
tmux new-window -t $SESSION -n "collect"
tmux send-keys -t $SESSION:3 "$SETUP_CMD && \
cd ~/cobot_magic/collect_data/ && \
python collect_data_master_with_cam.py \
    --dataset_dir ${DATASET_DIR} \
    --task_name ${TASK_NAME} \
    --max_timesteps ${MAX_TIMESTEPS} \
    --episode_idx ${EPISODE_IDX}" Enter

# 进入 tmux，默认显示采集窗口
tmux select-window -t $SESSION:3
tmux attach-session -t $SESSION