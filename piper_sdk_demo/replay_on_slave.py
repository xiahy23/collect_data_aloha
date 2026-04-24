#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_on_slave.py
==================
将主臂临时切成从臂模式，回放 HDF5 数据，完成后切回主臂模式。

完整流程：
  1. 连接两路 CAN，发送 MasterSlaveConfig(0xFC) 切为从臂
  2. 提示用户断电重启两臂
  3. 使能电机 + 设置运动模式
  4. 缓慢移动到 episode 第一帧位置
  5. 按 Enter 开始全速回放
  6. 回放结束后失能电机
  7. 发送 MasterSlaveConfig(0xFA) 切回主臂
  8. 提示用户再次断电重启

用法:
    conda activate aloha
    python replay_on_slave.py --hdf5 ~/data/my_task/episode_0.hdf5

注意:
  - 运行前确保 start_master_aloha.launch 已停止（会抢 CAN 总线）
  - 先在机械臂工作空间内无障碍物时测试低速 (--speed 10)
  - 两臂都需要断电重启才能切换模式
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import h5py
import numpy as np
from piper_sdk import C_PiperInterface

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
# JOINT_LIMITS_DEG = [
#     (-150.0, 150.0),
#     (   0.0, 180.0),
#     (-170.0,   0.0),
#     (-100.0, 100.0),
#     ( -70.0,  70.0),
#     (-120.0, 120.0),
# ]
JOINT_LIMITS_DEG = [
    (-180.0, 180.0)
]*6


# --------------------------------------------------------------------------
# 单位换算
# --------------------------------------------------------------------------
def rad_to_raw(angle_rad: float) -> int:
    """rad → 0.001 deg（JointCtrl 接口单位）"""
    return int(round(angle_rad * 1000.0 * 180.0 / math.pi))


def gripper_m_to_raw(m: float) -> int:
    """meters → grippers_angle raw（单位 1e-6 m = 0.001 mm）"""
    return int(round(m * 1_000_000))


# --------------------------------------------------------------------------
# 关节限位保护
# --------------------------------------------------------------------------
def clamp_joints_rad(joints_rad: list[float]) -> list[float]:
    out = []
    for i, (v, (lo_d, hi_d)) in enumerate(zip(joints_rad, JOINT_LIMITS_DEG)):
        v_deg = math.degrees(v)
        if v_deg < lo_d or v_deg > hi_d:
            print(f"  [warn] joint{i+1}={v_deg:.2f}° 超出限位 [{lo_d},{hi_d}°]，已截断")
            v = math.radians(max(lo_d, min(hi_d, v_deg)))
        out.append(v)
    return out


# --------------------------------------------------------------------------
# 发送单帧
# --------------------------------------------------------------------------
def send_frame(piper_l: C_PiperInterface, piper_r: C_PiperInterface,
               action_14: np.ndarray) -> None:
    """将 14 维 action 发送到左右两臂（rad + m）"""
    left  = action_14[:7]   # 左臂 joint0~5 (rad) + gripper (m)
    right = action_14[7:]   # 右臂 joint0~5 (rad) + gripper (m)

    jl = clamp_joints_rad(list(left[:6].astype(float)))
    jr = clamp_joints_rad(list(right[:6].astype(float)))

    raws_l = [rad_to_raw(j) for j in jl]
    raws_r = [rad_to_raw(j) for j in jr]
    gl_raw = gripper_m_to_raw(float(left[6]))
    gr_raw = gripper_m_to_raw(float(right[6]))

    piper_l.JointCtrl(*raws_l)
    piper_r.JointCtrl(*raws_r)
    piper_l.GripperCtrl(gl_raw, 1000, 0x01, 0)
    piper_r.GripperCtrl(gr_raw, 1000, 0x01, 0)


# --------------------------------------------------------------------------
# 使能 / 失能
# --------------------------------------------------------------------------
def enable_both(piper_l: C_PiperInterface, piper_r: C_PiperInterface,
                timeout: float = 10.0) -> bool:
    print("[enable] 使能两臂电机...")

    def all_enabled(p: C_PiperInterface) -> bool:
        st = p.GetArmLowSpdInfoMsgs()
        return all([
            st.motor_1.foc_status.driver_enable_status,
            st.motor_2.foc_status.driver_enable_status,
            st.motor_3.foc_status.driver_enable_status,
            st.motor_4.foc_status.driver_enable_status,
            st.motor_5.foc_status.driver_enable_status,
            st.motor_6.foc_status.driver_enable_status,
        ])

    t0 = time.time()
    while True:
        ok_l = all_enabled(piper_l)
        ok_r = all_enabled(piper_r)
        print(f"  left={ok_l}  right={ok_r}")
        if ok_l and ok_r:
            print("[enable] ✓ 使能成功")
            return True
        if time.time() - t0 > timeout:
            print("[enable] ✗ 超时！请确认：")
            print("         1) 机械臂已断电重启")
            print("         2) 已通过 --to-slave 切换过模式")
            return False
        piper_l.EnableArm(7)
        piper_r.EnableArm(7)
        piper_l.GripperCtrl(0, 1000, 0x01, 0)
        piper_r.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(0.5)


def disable_both(piper_l: C_PiperInterface, piper_r: C_PiperInterface) -> None:
    for _ in range(5):
        piper_l.DisableArm(7)
        piper_r.DisableArm(7)
        piper_l.GripperCtrl(0, 1000, 0x02, 0)
        piper_r.GripperCtrl(0, 1000, 0x02, 0)
        time.sleep(0.1)
    print("[disable] ✓ 电机已失能")


# --------------------------------------------------------------------------
# 模式切换
# --------------------------------------------------------------------------
def switch_mode(piper_l: C_PiperInterface, piper_r: C_PiperInterface,
                mode_hex: int, label: str) -> None:
    print(f"[mode] 发送 MasterSlaveConfig(0x{mode_hex:02X}) → {label}...")
    for _ in range(3):
        piper_l.MasterSlaveConfig(mode_hex, 0, 0, 0)
        piper_r.MasterSlaveConfig(mode_hex, 0, 0, 0)
        time.sleep(0.2)
    print(f"[mode] ✓ 已发送 {label} 配置")


# --------------------------------------------------------------------------
# 读取当前关节位置（从臂模式）
# --------------------------------------------------------------------------
def read_current_14(piper_l: C_PiperInterface,
                    piper_r: C_PiperInterface) -> np.ndarray:
    def arm_7(p: C_PiperInterface) -> list[float]:
        jm = p.GetArmJointMsgs().joint_state
        rads = [
            jm.joint_1 / 1000.0 * math.pi / 180.0,
            jm.joint_2 / 1000.0 * math.pi / 180.0,
            jm.joint_3 / 1000.0 * math.pi / 180.0,
            jm.joint_4 / 1000.0 * math.pi / 180.0,
            jm.joint_5 / 1000.0 * math.pi / 180.0,
            jm.joint_6 / 1000.0 * math.pi / 180.0,
        ]
        grip = p.GetArmGripperMsgs().gripper_state.grippers_angle / 1_000_000.0
        return rads + [grip]

    return np.array(arm_7(piper_l) + arm_7(piper_r))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--hdf5", required=True, help="HDF5 数据文件路径")
    ap.add_argument("--can-left",  default="can_left")
    ap.add_argument("--can-right", default="can_right")
    ap.add_argument("--speed", type=int, default=20,
                    help="运动速度百分比 0-100（默认 20，首次建议 10）")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="回放帧率（默认 30，与采集时一致）")
    ap.add_argument("--interp", type=int, default=3,
                    help="相邻帧插值步数，用于平滑 CAN 指令（默认 3）")
    ap.add_argument("--skip-mode-switch", action="store_true",
                    help="跳过模式切换步骤（臂已经是从臂模式时使用）")
    ap.add_argument("--no-log-actual", action="store_true",
                    help="不保存实际关节角记录（默认保存到 *_actual.npy）")
    args = ap.parse_args()

    # ── 加载数据 ──────────────────────────────────────────────────────────
    print(f"\n[data] 加载 {args.hdf5}")
    with h5py.File(args.hdf5, "r") as f:
        actions = f["action"][()].astype(np.float64)   # (T, 14)
    T = len(actions)
    print(f"[data] ✓ {T} 帧 × 14 维，预计回放时长 {T/args.fps:.1f}s")

    # ── 假零头检测 & 修复 ──────────────────────────────────────────────────
    # 采集时 CAN 数据未稳定导致的前几十帧为全零，需要用第一个有效帧填充，
    # 否则 replay 会先把手臂移到机械零位，再突然跳到真实起始位置。
    first_valid = 0
    for i in range(T):
        if np.any(np.abs(actions[i]) > 1e-3):
            first_valid = i
            break
    if first_valid > 5:
        print(f"[data] ⚠  检测到前 {first_valid} 帧为假零（CAN 未稳定），"
              f"已用第 {first_valid} 帧填充")
        actions[:first_valid] = actions[first_valid]
    elif first_valid > 0:
        print(f"[data] 前 {first_valid} 帧为零（手臂本就在零位）")

    

    # ── 连接 CAN ──────────────────────────────────────────────────────────
    print(f"\n[can] 连接 {args.can_left} / {args.can_right}")
    piper_l = C_PiperInterface(args.can_left)
    piper_r = C_PiperInterface(args.can_right)
    piper_l.ConnectPort()
    piper_r.ConnectPort()
    time.sleep(0.5)

    try:
        # ══════════════════════════════════════════════════════════════════
        # Step 1: 切换到从臂模式
        # ══════════════════════════════════════════════════════════════════
        if not args.skip_mode_switch:
            switch_mode(piper_l, piper_r, 0xFC, "从臂(slave)")
            print()
            print("=" * 62)
            print("  ⚠  请手动给 左臂 和 右臂 断电，然后重新上电。")
            print("     CAN 模块无需重启，只重启机械臂本体即可。")
            print("=" * 62)
            input("  → 断电重启完成后，按 Enter 继续...\n")
        else:
            print("[skip] 跳过模式切换，假设两臂已是从臂模式")

        # ══════════════════════════════════════════════════════════════════
        # Step 2: 先使能，使能成功后再设置运动模式
        # ══════════════════════════════════════════════════════════════════
        if not enable_both(piper_l, piper_r):
            print("[abort] 使能失败，退出")
            sys.exit(1)

        # 使能成功后设置 CAN 控制 + MoveJ + 速度
        piper_l.MotionCtrl_2(0x01, 0x01, args.speed, 0x00)
        piper_r.MotionCtrl_2(0x01, 0x01, args.speed, 0x00)
        time.sleep(0.3)
        print(f"[mode] MotionCtrl_2: CAN+MoveJ, speed={args.speed}%")

        # ══════════════════════════════════════════════════════════════════
        # Step 3: 缓慢移动到第一帧位置
        # ══════════════════════════════════════════════════════════════════
        print(f"\n[init] 读取当前位置...")
        time.sleep(0.3)
        cur_14 = read_current_14(piper_l, piper_r)
        target_14 = actions[0]
        print(f"[init] 以速度 {args.speed}% 移动到第一帧（80步，约2.4s）...")

        for frame in np.linspace(cur_14, target_14, 80):
            send_frame(piper_l, piper_r, frame)
            time.sleep(0.03)
        print("[init] ✓ 已到达第一帧位置")

        # ══════════════════════════════════════════════════════════════════
        # Step 4: 回放
        # ══════════════════════════════════════════════════════════════════
        dt_frame = 1.0 / args.fps
        dt_cmd   = dt_frame / args.interp
        print(f"\n[replay] 参数: fps={args.fps}, interp={args.interp}, "
              f"CAN间隔={dt_cmd*1000:.1f}ms")
        input(f"[replay] 按 Enter 开始回放 {T} 帧...\n")

        t_start = time.time()
        prev = actions[0].copy()
        actual_states = []   # [(left_6_rad, right_6_rad), ...]

        for i, action in enumerate(actions):
            t_frame_start = time.time()

            # 每 30 帧重发一次 MotionCtrl_2，防止 CAN 控制模式超时失效
            if i % 30 == 0:
                piper_l.MotionCtrl_2(0x01, 0x01, args.speed, 0x00)
                piper_r.MotionCtrl_2(0x01, 0x01, args.speed, 0x00)

            # 插值并发送
            for frame in np.linspace(prev, action, args.interp + 1)[1:]:
                send_frame(piper_l, piper_r, frame)
                time.sleep(dt_cmd)
            prev = action

            # 采样实际关节角（用于事后误差分析）
            if not args.no_log_actual:
                try:
                    jl = piper_l.GetArmJointMsgs().joint_state
                    jr = piper_r.GetArmJointMsgs().joint_state
                    actual_states.append([
                        jl.joint_1, jl.joint_2, jl.joint_3,
                        jl.joint_4, jl.joint_5, jl.joint_6,
                        jr.joint_1, jr.joint_2, jr.joint_3,
                        jr.joint_4, jr.joint_5, jr.joint_6,
                    ])
                except Exception:
                    pass  # 采样失败时跳过

            # 进度打印
            if (i + 1) % 100 == 0 or i == T - 1:
                elapsed = time.time() - t_start
                print(f"  [replay] {i+1}/{T}  elapsed={elapsed:.1f}s")

        elapsed_total = time.time() - t_start
        print(f"\n[replay] ✓ 回放完成，耗时 {elapsed_total:.1f}s（原始时长 {T/args.fps:.1f}s）")

        # 保存实际关节角数据
        if not args.no_log_actual and actual_states:
            import os
            log_path = os.path.splitext(args.hdf5)[0] + "_actual.npy"
            arr = np.array(actual_states, dtype=np.float32)  # (T, 12) 单位 0.001deg
            np.save(log_path, arr)
            print(f"[log] ✓ 实际关节角已保存到 {log_path}  shape={arr.shape}")

        # ══════════════════════════════════════════════════════════════════
        # Step 5: 回放完成，保持末位姿
        # ══════════════════════════════════════════════════════════════════
        # ⚠ 不在此处失能电机：DisableArm 会立即断力矩导致机械臂自由下落。
        # 保持最后一帧的 JointCtrl 使机械臂抱住末位姿，由 Step 6 断电重启
        # 来结束控制（master 模式内置重力补偿，重启后安全）。
        print("\n[hold] 回放完成，电机保持末位姿，等待断电重启...")

    except KeyboardInterrupt:
        print("\n[interrupt] 用户中断，保持当前位姿...")
        # Ctrl+C 时同样不立即失能，让 Step 6 的断电重启来结束

    # ══════════════════════════════════════════════════════════════════════
    # Step 6: 切回主臂模式（无论是否中断，都执行）
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_mode_switch:
        switch_mode(piper_l, piper_r, 0xFA, "主臂(master)")
        print()
        print("=" * 62)
        print("  ✓ 已发送主臂模式配置（0xFA）。")
        print("  ⚠  请手动给 左臂 和 右臂 断电，然后重新上电。")
        print("     重启后两臂将恢复为拖动示教（重力补偿）模式。")
        print("=" * 62)
        input("  → 断电重启完成后，按 Enter 退出...\n")
    else:
        # 没有断电重启：需要人工扶住后再失能
        input("[hold] --skip-mode-switch 模式。请用手扶住机械臂后，按 Enter 失能电机...\n")
        disable_both(piper_l, piper_r)

    print("[done] 脚本结束。")


if __name__ == "__main__":
    main()
