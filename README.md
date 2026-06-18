# Cobot Magic — Data Collection

基于 Piper 双臂（主从遥操）+ 多路 Astra 相机的数据采集工程，提供 Tkinter UI、踏板触发、HDF5 落盘以及到 LeRobot v2.0 的转换工具，并针对"形状插孔"（shape-to-hole）任务提供专用的确定性布局采集流水线。

## 目录结构

```
collect_data/
    scripts/            采集 / 启停 / 转换 / 可视化脚本（本文档主要内容）
    piper_sdk_demo/      Piper 机械臂 SDK 使用示例（主从配置、CAN 激活、关节控制等）
    dataarm_notifier/     指示灯（USB Lamp）与踏板/键盘监听模块，供采集脚本调用
    docs/                示例图片、布局总览 HTML 等文档资产
```

## 环境准备

```bash
conda activate aloha
pip install -r scripts/requirements.txt
```

机械臂、CAN 总线配置见 `piper_sdk_demo/README.MD`；相机依赖 `astra_camera` ROS 包（`~/cobot_magic/camera_ws`）。

## 数据采集流水线一览

`scripts/` 下提供多套采集流水线，按演进顺序：

| 启动脚本 | 采集程序 | 适用场景 |
| --- | --- | --- |
| `start_collect.sh` / `stop_collect.sh` | `collect_data.py` | 最基础的采集（无 UI/踏板） |
| `start_collect_pedal.sh` / `stop_collect_pedal.sh` | `collect_data_master_with_cam_pedal.py` | 加入踏板控制 |
| `begin_collect_pipeline.sh` | `collect_data_pipeline_subtask.py` | **UI + 踏板 + 多 instruction + subtask 推进**的通用流水线 |
| `begin_collect_pipeline_shape_holes.sh` | `collect_data_pipeline_shape_holes.py` | 在通用流水线基础上，叠加**形状插孔任务专用的确定性布局规划** |

后两者是当前主推的流水线，UI、磁盘布局、指示灯/踏板行为基本一致，下面统一说明。

## 一键启动脚本做了什么

`begin_collect_pipeline.sh` 与 `begin_collect_pipeline_shape_holes.sh` 结构一致，在同一个 tmux 会话里按顺序拉起：

1. **roscore**：等待 `rostopic list` 可用。
2. **piper_ms**：执行 `can_config.sh` 配置 CAN 总线后 `roslaunch piper start_ms_piper.launch`，启动失败自动重试；随后等待 `/master/joint_left`、`/master/joint_right`、`/puppet/joint_left`、`/puppet/joint_right` 话题出现。
3. **camera**：`roslaunch astra_camera multi_camera.launch` 启动三路相机，等待 `/camera_f/color/image_raw` 话题出现。
4. **collect**：进入 `scripts/` 运行对应的采集 Python 程序，并把窗口 attach 到前台。

可通过环境变量覆盖默认参数：

```bash
cd ~/cobot_magic/collect_data/scripts

# 通用 subtask 流水线
DATASET_DIR=~/data TASK_NAME=my_task MAX_TIMESTEPS=9000 \
  bash begin_collect_pipeline.sh

# 形状插孔流水线
DATASET_DIR=~/data TASK_NAME=shape_hole_pipeline MAX_TIMESTEPS=9000 \
  bash begin_collect_pipeline_shape_holes.sh
```

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATASET_DIR` | `~/data` | 数据集根目录 |
| `TASK_NAME` | `aloha_pipeline` / `shape_hole_pipeline` | 任务名，对应磁盘子目录 |
| `MAX_TIMESTEPS` | `9000` | 单条 episode 最大帧数，达到后自动停止保存 |
| `LAMP_PORT` | `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0` | 指示灯串口设备 |
| `SESSION` | `collect_pipeline` / `collect_shape_holes` | tmux 会话名 |

停止：`bash terminate_collect_pipeline.sh`（依次向各 tmux 窗口发 `Ctrl-C` 再关闭会话），或直接 `tmux kill-session -t <SESSION>`。

> 如需在 SSH 中显示 UI，请确保已开启 X11 转发（`ssh -X` / `ssh -Y`），或将 `DISPLAY` 指向支持的会话。

## 采集程序的共同行为（UI + 踏板）

- **指示灯**：青色 = 空闲，绿色 = 录制中，黄色 = 保存中。
- **Subtask 推进**：一条 episode 可拆分为多个子任务（subtask），踏板（默认触发键 `Enter`）按压含义：
  - 第 1 次按压：开始录制，灯变绿，激活第 1 个 subtask；
  - 第 2..N 次按压：仍在录制中，依次切到下一个 subtask；
  - 第 N+1 次按压（无更多 subtask 时）：停止并保存当前 episode。
  - 每个时间步会把当前激活的 subtask 文本写入 HDF5 的 `/observations/subtask`。
- **Instruction 管理**：UI 左侧维护 instruction 列表，可增删（删除仅从列表移除，不删磁盘数据）；不同 instruction 的数据分别存入对应子目录；同一 instruction 下可连续录制多条 episode。
- **Episode 管理**：右侧面板列出已存 episode，可选中删除（同步删除磁盘 hdf5）。
- **自动回零**（`--auto_home`，默认开启）：保存成功后调用 `go_home.py` 的 `go_home_and_wait`，驱动主从臂回零，便于下一条录制前姿态一致；可用 `--no_auto_home` 关闭。
- **磁盘布局**：

  ```
  <dataset_dir>/<task_name>/
      pipeline_meta.json                # instruction 列表与目录 slug 映射
      <instruction_slug>/
          episode_0.hdf5
          episode_1.hdf5
          ...
  ```

## `collect_data_pipeline_subtask.py`（通用版）

通用 subtask 采集器，不内置任务相关布局逻辑，subtask 列表通过命令行 `--subtasks` 指定，适用于任意"分阶段动作"任务：

```bash
python collect_data_pipeline_subtask.py \
    --subtasks "reach the handle of the container" \
               "grasp the container" \
               "lift the container"
```

## `collect_data_pipeline_shape_holes.py`（形状插孔专用版）

在通用 subtask 能力之上，针对"把形状放入对应孔位"任务增加了**确定性的逐 episode 布局规划**：

- **物体与孔位**：每条 episode 含上排 4 个实体（六棱柱 / 长方体 / 立方体 / 三棱柱）与下排 3 个孔（四种孔中取 3 个）。
- **布局覆盖**：
  - 上排 4 个位置的全部 `4! = 24` 种排列（`TOP_SHAPE_CYCLE`）；
  - 下排 3 个孔位的全部 `P(4,3) = 24` 种有序选择（`HOLE_SHAPE_CYCLE`）；
  - 二者笛卡尔积共 `24 × 24 = 576` 种组合，按 `--layout_pairing_seed`（默认 `20260516`）确定性打乱后循环使用，相同 seed 下序列可复现；每 576 条 episode 后完全重复，且每 24 条一个 block 内上排/下排各布局恰好各出现一次。
  - 程序会读取已采集 episode 中记录的 layout 信息，自动跳过已用过的布局对，从下一个未使用的布局继续采集。
- **UI 增强**：标准 UI 基础上新增布局画布，用图形/颜色直观展示当前 episode 该摆放的形状与孔位顺序，配合"下一条布局预览"在开始录制前先看到计划。
- **HDF5 附加内容**：除标准的图像/关节角/subtask 外，额外保存本条 episode 的布局计划（形状-位置、孔-位置映射），便于复现实验台摆放与后续核验。
- 命令行参数与通用版基本一致，额外增加 `--layout_pairing_seed`（布局配对随机种子）。

### 配套工具：布局总览生成器

`scripts/generate_shape_hole_layout_overview.py` 可独立运行，生成一份图形化总览（默认输出 `docs/shape_hole_200_layouts.html`），用图标罗列布局循环中若干条 episode 应该摆放的形状与孔位，便于打印或现场核对实际摆台是否与采集计划一致：

```bash
python generate_shape_hole_layout_overview.py \
    --output docs/shape_hole_200_layouts.html \
    --target_count 200 \
    --seed 20260516
```

## 命令行参数（两套流水线通用，直接运行 *.py 时可用）

```
--dataset_dir            数据根目录，默认 ./data
--task_name              任务名
--max_timesteps          单条最大帧数（达到自动停止保存），默认 9000
--frame_rate             采集帧率，默认 30
--jpeg_quality           JPEG 编码质量，默认 90
--camera_names           相机命名列表，默认 cam_high cam_left_wrist cam_right_wrist
--img_front_topic / --img_left_topic / --img_right_topic
--master_arm_left_topic / --master_arm_right_topic
--puppet_arm_left_topic / --puppet_arm_right_topic
--lamp_port              指示灯串口，默认自动检测
--pedal_device           踏板设备，默认自动检测 /dev/input/by-id/*event-kbd
--trigger_key            触发键名，默认 enter
--instructions           可选：首次启动时初始化 instruction 列表
--auto_home / --no_auto_home   保存后是否自动回零，默认开启
--layout_pairing_seed    （仅 shape_holes 版）布局配对随机种子，默认 20260516
```

## LeRobot 2.0 转换

```bash
# 默认：state=joint, action=joint, action_type=delta（与原始 14 维 master 关节角一致）
python convert_hdf5_to_lerobot_v21.py \
    --src_dir ~/data/aloha_pipeline \
    --repo_id ~/data/local/my_task_lerobot

# 使用末端位姿 + 绝对动作
python convert_hdf5_to_lerobot_v21.py \
    --src_dir ~/data/my_task \
    --repo_id local/my_task_ee_abs \
    --state_mode ee --action_mode ee --action_type absolute
```

| 参数 | 取值 | 默认 | 说明 |
| --- | --- | --- | --- |
| `--state_mode` | `joint` / `ee` | `joint` | 观测维度：14 维关节角 或 12 维双臂末端位姿 (xyz + rxryrz × 2) |
| `--action_mode` | `joint` / `ee` | `joint` | 动作维度同上 |
| `--action_type` | `delta` / `absolute` | `delta` | `delta`：相邻帧差，最后一帧丢弃；`absolute`：原值 |
| `--fps` | int | 取自 hdf5 attrs | 覆盖帧率 |
| `--robot_type` | str | `aloha-piper` | 写入 LeRobot meta 的机器人类型 |
| `--overwrite` | flag | — | 覆盖本地已存在的同名数据集目录 |

## 其他脚本

- `replay_data.py`：回放已采集 hdf5 中的动作序列，用于核验数据。
- `visualize_episodes.py`：将 episode 中的 qpos / action 等绘制成图（参考 `docs/episode_0_qpos.png`）。
- `preview_three_cameras_web.py` + `start_camera_preview_web.sh` / `stop_camera_preview_web.sh`：通过浏览器预览三路相机画面，无需 X11。
- `go_home.py`：调用 Piper ROS Service 让主从臂回零，被采集脚本的自动回零功能复用，也可单独运行排查回零问题。

## 常见问题

- **tmux 窗口卡在等待话题**：检查对应硬件（CAN / 相机）是否正常连接，可手动 `rostopic list` 排查。
- **UI 一直显示 Waiting for ROS topics**：确认 roscore、piper、相机 launch 均已在对应 tmux 窗口正常运行。
- **踏板无效**：检查 `/dev/input/by-id/*event-kbd` 是否存在及读权限（必要时配置 udev 规则），或用 `--pedal_device` 显式指定。
- **指示灯不亮**：检查 `--lamp_port` / `LAMP_PORT`；不指定时会自动尝试 `/dev/ttyUSB*` / `/dev/ttyACM*`。
- **保存为空**：确认录制中相机/关节话题持续有数据；`max_timesteps` 到达后会自动停止保存。
- **形状插孔版布局对不上**：确认 `--layout_pairing_seed` 与历史采集时一致；更换 seed 会得到不同的布局循环顺序。
