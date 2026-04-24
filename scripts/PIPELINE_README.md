# Data Collection Pipeline (UI + Pedal)

基于 `collect_data_master_with_cam_pedal.py` 与 `dataarm_notifier` 扩展的完整数据采集流水线，包含：

- 一个 Tkinter 图形界面（采集端运行；通过 X11 转发或本机显示打开）。
- 采集前可配置 instruction 列表，运行中可继续增删。
- 同一 instruction 下可连续采集多条；支持 UI 按钮 / 踏板（Enter 键）控制 **Start / Stop / Stop & Start Next**。
- 状态灯：青色 = 空闲，绿色 = 录制中，黄色 = 保存中。
- 失败的 episode 可以在 UI 中直接选中删除（同步从磁盘删除）。
- 切换 instruction 直接在 UI 列表中点选；新数据写入对应子目录。
- 配套转换脚本将采集结果转为 **LeRobot v2.0** 数据集，支持关节角 / EE pose 与 delta / 绝对值动作。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| [scripts/collect_data_pipeline.py](scripts/collect_data_pipeline.py) | UI + 踏板 数据采集主程序 |
| [scripts/start_collect_pipeline.sh](scripts/start_collect_pipeline.sh) | 一键启动 roscore / piper / camera / 采集端（tmux） |
| [scripts/stop_collect_pipeline.sh](scripts/stop_collect_pipeline.sh) | 关闭 tmux 会话 |
| [scripts/convert_to_lerobot.py](scripts/convert_to_lerobot.py) | HDF5 -> LeRobot v2.0 转换脚本 |

## 磁盘布局

```
<dataset_dir>/<task_name>/
    pipeline_meta.json                # 记录 instruction 列表与 slug 映射
    pick_up_red_cube/
        episode_0.hdf5
        episode_1.hdf5
    place_block_into_box/
        episode_0.hdf5
        ...
```

每个 hdf5 与原 pedal 版本结构一致，并新增 attrs：`instruction`、`frame_rate`，便于后续转换。

## 运行步骤

### 1. 启动整套服务

```bash
cd ~/cobot_magic/collect_data/scripts
# 可通过环境变量覆盖默认值
DATASET_DIR=~/data TASK_NAME=my_task MAX_TIMESTEPS=9000 \
  bash start_collect_pipeline.sh
```

脚本会顺序启动：roscore → piper master → 多相机 → UI 采集进程，并 attach 到最后一个 tmux 窗口。

> 如需在 SSH 中显示 UI，请确保已开启 X11 转发（`ssh -X` 或 `ssh -Y`），或将 `DISPLAY` 指向支持的会话。

### 2. UI 操作流程

1. 程序启动后底部状态栏显示 `Waiting for ROS topics...`，待变为 `Ready` 即可使用。
2. 在左侧 `Instructions` 面板：
   - 点击 **Add** 增加 instruction。
   - 选中后点击 **Remove** 仅从列表移除（不删除已采集的 hdf5 文件）。
3. 选中目标 instruction，点击 **Start (Pedal)** 或踩一次踏板：开始录制，状态灯变绿。
4. 再次点击 **Stop & Save (Pedal)** 或踩一次踏板：停止并保存，状态灯黄→青。
5. 想立即采下一条同 instruction，可点击 **Stop & Start Next**，会自动保存当前 episode 并马上开启下一条。
6. 切换 instruction：直接在左侧列表点选；右侧 `Episodes` 面板会刷新该 instruction 的所有已存 episode。
7. 失败 episode：右侧选中 → **Delete selected episode**，磁盘文件同步删除。

### 3. 关闭

```bash
bash stop_collect_pipeline.sh
```

## 命令行参数（直接运行 collect_data_pipeline.py）

```
--dataset_dir            数据根目录，默认 ./data
--task_name              任务名，默认 aloha_pipeline
--max_timesteps          单条最大帧数（达到自动停止保存），默认 9000
--frame_rate             采集帧率，默认 30
--jpeg_quality           JPEG 编码质量，默认 90
--camera_names           相机命名列表，默认 cam_high cam_left_wrist cam_right_wrist
--img_front_topic / --img_left_topic / --img_right_topic
--master_arm_left_topic / --master_arm_right_topic
--lamp_port              指示灯串口，默认自动检测 /dev/dataarm_notifier
--pedal_device           踏板设备，默认自动检测 /dev/input/by-id/*event-kbd
--trigger_key            触发键名，默认 enter
--instructions           可选：首次启动时初始化 instruction 列表
```

## LeRobot 2.0 转换

```bash
# 默认：state=joint, action=joint, action_type=delta（与原始 14 维 master 关节角一致）
python convert_to_lerobot.py \
    --src_dir ~/data/aloha_pipeline \
    --repo_id ~/data/local/my_task_lerobot

# 使用末端位姿 + 绝对动作
python convert_to_lerobot.py \
    --src_dir ~/data/my_task \
    --repo_id local/my_task_ee_abs \
    --state_mode ee --action_mode ee --action_type absolute

# 指定本地输出根目录（默认 ~/.cache/huggingface/lerobot）
python convert_to_lerobot.py \
    --src_dir ~/data/my_task \
    --repo_id local/my_task \
    --out_root ~/datasets/lerobot \
    --overwrite
```

参数说明：

| 参数 | 取值 | 默认 | 说明 |
| --- | --- | --- | --- |
| `--state_mode` | `joint` / `ee` | `joint` | 观测维度：14 维关节角 或 12 维双臂末端位姿 (xyz + rxryrz × 2) |
| `--action_mode` | `joint` / `ee` | `joint` | 动作维度同上 |
| `--action_type` | `delta` / `absolute` | `delta` | `delta`：相邻帧差，最后一帧丢弃；`absolute`：原值 |
| `--fps` | int | 取自 hdf5 attrs | 覆盖帧率 |
| `--robot_type` | str | `aloha-piper` | 写入 LeRobot meta 的机器人类型 |
| `--overwrite` | flag | — | 覆盖本地已存在的同名数据集目录 |

> **EE 模式说明**：脚本会优先读取 hdf5 中可选的 `/observations/ee_pose`；若不存在则尝试 FK：
> 1. 若你提供了 `piper_fk.py`（暴露 `fk(joint7) -> 6-vec`），会优先使用；
> 2. 否则尝试 `piper_sdk.PiperFK`；
> 3. 若都没有，会以清晰报错退出。
> 后续如需更精确的 EE 数据，建议在采集端将 `/follow/end_pose` 之类的话题写入 hdf5 的 `/observations/ee_pose`，转换脚本会自动直接采用。

## 常见问题

- **UI 一直显示 Waiting for ROS topics**：确认 `roscore`、`piper start_master_aloha.launch`、相机 launch 已在对应 tmux 窗口正常运行；可用 `rostopic list` 检查话题是否齐全。
- **踏板无效**：检查 `/dev/input/by-id/*event-kbd` 是否存在以及读权限（必要时配置 udev 规则），或使用 `--pedal_device` 显式指定。
- **指示灯不亮**：检查 `--lamp_port`（默认 `/dev/dataarm_notifier`）；不指定时会自动尝试 `/dev/ttyUSB*` / `/dev/ttyACM*`。
- **保存为空**：确认录制中相机/关节话题持续有数据；`max_timesteps` 到达后会自动停止保存。
