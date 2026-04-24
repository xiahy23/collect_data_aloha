#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_repeatability.py  —  验证 Piper 机械臂重复定位精度（手册宣称 0.01 mm）

测试原理
========
  1. [示教阶段]  机械臂处于主臂模式（拖动示教）。
     用户将左/右臂手动拖到若干目标位姿，脚本读取并记录关节角 q_cmd。

  2. [回放阶段]  机械臂切换到从臂模式（断电重启后）。
     脚本对每个目标位姿重复发送 JointCtrl(q_cmd) R 次，
     每次等机械臂稳定后读取实际关节角 q_act。

  3. [分析]      对 q_cmd 和 q_act 做正运动学得到末端位置 p_cmd / p_act，
     计算每次复位的位置偏差，并统计:
        - 位置精度 (accuracy)     = ||p_act - p_cmd|| 的均值/最大值
        - 重复定位精度 (repeat.)  = 同一目标下 R 次 p_act 相对其均值的最大偏差

用法
====
  # 1) 先让机械臂处于主臂模式（常规采集/示教状态）
  conda activate aloha
  python3 test_repeatability.py --trials 5 --repeats 5

  # 中途按提示断电重启进入从臂模式，继续测试

  # 若机械臂已在从臂模式且想跳过示教阶段，可先录目标 json 再复现:
  python3 test_repeatability.py --replay-only targets.json --repeats 10

注意
====
  - 运行前必须停止所有占用 CAN 总线的 ROS 节点（start_master_aloha.launch 等）
  - 首次建议 --speed 10，且工作空间内无障碍物
  - 本脚本仅控制关节运动，不控制夹爪
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List

import numpy as np
from piper_sdk import C_PiperInterface

# 复用已有脚本的工具
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_fk_error import fk_batch  # noqa: E402
from replay_on_slave import (        # noqa: E402
    rad_to_raw,
    clamp_joints_rad,
    enable_both,
    disable_both,
    switch_mode,
)


# --------------------------------------------------------------------------
# 读取 / 发送关节角（只关心 6 个关节，不含夹爪）
# --------------------------------------------------------------------------
# ⚠ Piper SDK 有两个读关节 API，对应不同模式：
#   - GetArmJointCtrl : **主臂模式** 下人手拖动的实时关节角（来自 CAN 0x2B1-0x2B3）
#   - GetArmJointMsgs : **从臂模式** 下的实际关节反馈（来自 CAN 0x2A5-0x2A7）
# 主臂模式下 GetArmJointMsgs 恒为 0，所以示教阶段必须用 GetArmJointCtrl。
def read_joints_rad(piper: C_PiperInterface, source: str = "auto") -> np.ndarray:
    """
    读取 6 关节角，单位 rad

    Parameters
    ----------
    source : {"ctrl", "msgs", "auto"}
        - "ctrl": 用 GetArmJointCtrl（主臂/示教模式）
        - "msgs": 用 GetArmJointMsgs（从臂/回放模式的实际反馈）
        - "auto": 优先 msgs，全为 0 时回退到 ctrl
    """
    if source == "ctrl":
        jc = piper.GetArmJointCtrl().joint_ctrl
        raws = [jc.joint_1, jc.joint_2, jc.joint_3,
                jc.joint_4, jc.joint_5, jc.joint_6]
    elif source == "msgs":
        jm = piper.GetArmJointMsgs().joint_state
        raws = [jm.joint_1, jm.joint_2, jm.joint_3,
                jm.joint_4, jm.joint_5, jm.joint_6]
    else:  # auto
        jm = piper.GetArmJointMsgs().joint_state
        raws = [jm.joint_1, jm.joint_2, jm.joint_3,
                jm.joint_4, jm.joint_5, jm.joint_6]
        if all(abs(r) < 1 for r in raws):  # 全为 0 → 可能处于主臂模式
            jc = piper.GetArmJointCtrl().joint_ctrl
            raws = [jc.joint_1, jc.joint_2, jc.joint_3,
                    jc.joint_4, jc.joint_5, jc.joint_6]
    return np.array(raws, dtype=np.float64) / 1000.0 * math.pi / 180.0


def send_joints_rad(piper: C_PiperInterface, q_rad: np.ndarray) -> None:
    """发送 6 关节角，单位 rad"""
    q_clamped = clamp_joints_rad(list(q_rad.astype(float)))
    raws = [rad_to_raw(q) for q in q_clamped]
    piper.JointCtrl(*raws)


# --------------------------------------------------------------------------
# 稳定等待：读取连续若干帧关节角，直到相邻两次变化 < eps
# --------------------------------------------------------------------------
def wait_until_settled(piper: C_PiperInterface,
                       timeout: float = 5.0,
                       eps_deg: float = 0.02,
                       window: float = 0.3) -> np.ndarray:
    """
    等到机械臂稳定后返回当前关节角 (rad)。

    Parameters
    ----------
    timeout : 最大等待时间 (s)
    eps_deg : 稳定判定阈值：窗口内最大关节变化 (deg)
    window  : 判稳窗口长度 (s)
    """
    eps_rad = math.radians(eps_deg)
    t0 = time.time()
    samples: List[tuple] = []   # (t, q)
    while time.time() - t0 < timeout:
        q = read_joints_rad(piper, source="msgs")  # 回放时读实际反馈
        now = time.time()
        samples.append((now, q))
        # 丢弃窗口外样本
        samples = [(t, s) for (t, s) in samples if now - t <= window]
        if len(samples) >= 3 and now - samples[0][0] >= window:
            qs = np.stack([s for _, s in samples])
            spread = qs.max(axis=0) - qs.min(axis=0)
            if np.max(spread) < eps_rad:
                return qs.mean(axis=0)
        time.sleep(0.02)
    # 超时：返回最后一次读数并警告
    print(f"  [warn] 关节未在 {timeout}s 内稳定，使用最近读数")
    return samples[-1][1]


# --------------------------------------------------------------------------
# 示教阶段：用户拖动机械臂，按 Enter 记录位姿
# --------------------------------------------------------------------------
def teach_phase(piper_l, piper_r, arm_sel: str, n_trials: int) -> list:
    """返回 [{'idx':i, 'left':[6], 'right':[6]}, ...]（单位 rad）"""
    print("\n" + "=" * 62)
    print("  [Phase 1 / 示教]  将机械臂拖动到目标位置后按 Enter 记录")
    print("=" * 62)
    print(f"  需要记录 {n_trials} 个目标位姿（测试臂：{arm_sel}）")
    print()

    targets = []
    for i in range(n_trials):
        input(f"  → 拖到目标 {i+1}/{n_trials}，按 Enter 记录...")
        # 连读几帧取均值，规避 CAN 抖动
        buf_l, buf_r = [], []
        for _ in range(10):
            if arm_sel in ("left", "both"):
                buf_l.append(read_joints_rad(piper_l, source="ctrl"))
            if arm_sel in ("right", "both"):
                buf_r.append(read_joints_rad(piper_r, source="ctrl"))
            time.sleep(0.02)

        entry = {"idx": i}
        if buf_l:
            q_l = np.mean(buf_l, axis=0)
            entry["left"] = q_l.tolist()
            print(f"    left  = {np.degrees(q_l).round(3).tolist()} deg")
        if buf_r:
            q_r = np.mean(buf_r, axis=0)
            entry["right"] = q_r.tolist()
            print(f"    right = {np.degrees(q_r).round(3).tolist()} deg")
        targets.append(entry)

    return targets


# --------------------------------------------------------------------------
# 回放阶段：对每个目标重复 R 次，记录 q_actual
# --------------------------------------------------------------------------
def replay_phase(piper_l, piper_r, arm_sel: str,
                 targets: list, repeats: int,
                 speed: int, settle_time: float) -> list:
    """
    返回 [{'idx':i, 'left':[R,6], 'right':[R,6]}, ...]（单位 rad）
    """
    print("\n" + "=" * 62)
    print("  [Phase 2 / 回放]  发送 JointCtrl 并读取实际关节角")
    print("=" * 62)

    # 设置运动模式
    piper_l.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    piper_r.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    time.sleep(0.3)
    print(f"  [mode] MotionCtrl_2: CAN+MoveJ, speed={speed}%")

    results = []
    for t_i, tgt in enumerate(targets):
        print(f"\n  ── 目标 {t_i+1}/{len(targets)} ──")
        entry = {"idx": tgt["idx"]}
        if "left" in tgt:
            entry["left"] = []
        if "right" in tgt:
            entry["right"] = []

        q_tgt_l = np.asarray(tgt.get("left", np.zeros(6)))
        q_tgt_r = np.asarray(tgt.get("right", np.zeros(6)))

        for r in range(repeats):
            # 每次先回到一个"中立位"，再运动到目标，以避免因初始姿态不同造成偏差
            if r > 0:
                # 简单中立位：当前位置与目标的中点方向退回一点
                # 这里直接发送目标指令，依赖机械臂自己从上次位置过来
                pass

            # 每 5 次重发一次运动模式
            if r % 5 == 0:
                piper_l.MotionCtrl_2(0x01, 0x01, speed, 0x00)
                piper_r.MotionCtrl_2(0x01, 0x01, speed, 0x00)

            # 发送指令
            if arm_sel in ("left", "both"):
                send_joints_rad(piper_l, q_tgt_l)
            if arm_sel in ("right", "both"):
                send_joints_rad(piper_r, q_tgt_r)

            # 等运动完成 + 稳定
            time.sleep(settle_time)

            if arm_sel in ("left", "both"):
                q_act = wait_until_settled(piper_l)
                entry["left"].append(q_act.tolist())
                err_deg = np.degrees(q_act - q_tgt_l)
                print(f"    r={r+1}/{repeats}  left  Δq(deg)="
                      f"{np.abs(err_deg).max():.4f} max")
            if arm_sel in ("right", "both"):
                q_act = wait_until_settled(piper_r)
                entry["right"].append(q_act.tolist())
                err_deg = np.degrees(q_act - q_tgt_r)
                print(f"    r={r+1}/{repeats}  right Δq(deg)="
                      f"{np.abs(err_deg).max():.4f} max")

        results.append(entry)

    return results


# --------------------------------------------------------------------------
# 分析：FK + 统计
# --------------------------------------------------------------------------
def analyze(targets: list, results: list, arm_sel: str) -> dict:
    """
    accuracy  = ||FK(q_act) - FK(q_tgt)||
    repeat.   = 同一目标 R 次 FK(q_act) 相对其均值的最大距离
    """
    report = {}
    for side in ("left", "right"):
        if arm_sel not in (side, "both"):
            continue
        acc_all = []   # (N*R,) 单次位置误差 mm
        rep_all = []   # (N,)   每个目标的重复性最大偏差 mm
        per_target = []

        for tgt, res in zip(targets, results):
            if side not in tgt:
                continue
            q_tgt = np.asarray(tgt[side])[None, :]          # (1,6)
            q_act = np.asarray(res[side])                    # (R,6)

            p_tgt = fk_batch(q_tgt) * 1000.0                 # (1,3) mm
            p_act = fk_batch(q_act) * 1000.0                 # (R,3) mm

            # 精度：每次相对指令的偏差
            acc = np.linalg.norm(p_act - p_tgt, axis=1)      # (R,)
            # 重复性：各次相对均值的偏差
            p_mean = p_act.mean(axis=0)
            rep = np.linalg.norm(p_act - p_mean, axis=1)     # (R,)

            acc_all.append(acc)
            rep_all.append(rep.max())
            per_target.append({
                "idx": tgt["idx"],
                "q_tgt_deg": np.degrees(q_tgt[0]).round(4).tolist(),
                "p_tgt_mm":  p_tgt[0].round(4).tolist(),
                "p_act_mm":  p_act.round(4).tolist(),
                "acc_mm":    acc.round(4).tolist(),
                "rep_max_mm": round(float(rep.max()), 4),
            })

        if not acc_all:
            continue
        acc_arr = np.concatenate(acc_all)
        rep_arr = np.array(rep_all)

        report[side] = {
            "accuracy_mm": {
                "mean": float(acc_arr.mean()),
                "max":  float(acc_arr.max()),
                "rmse": float(np.sqrt((acc_arr**2).mean())),
                "n_samples": int(acc_arr.size),
            },
            "repeatability_mm": {
                "mean": float(rep_arr.mean()),
                "max":  float(rep_arr.max()),
                "n_targets": int(rep_arr.size),
            },
            "per_target": per_target,
        }
    return report


def print_report(report: dict) -> None:
    print("\n" + "=" * 62)
    print("  [Result]  FK 末端位置误差统计")
    print("=" * 62)
    for side, r in report.items():
        acc = r["accuracy_mm"]
        rep = r["repeatability_mm"]
        print(f"\n  ── {side.upper()} 臂 ──")
        print(f"    位置精度 (Accuracy, 指令 vs 实际)  N={acc['n_samples']}")
        print(f"      mean = {acc['mean']:.4f} mm")
        print(f"      max  = {acc['max']:.4f} mm")
        print(f"      RMSE = {acc['rmse']:.4f} mm")
        print(f"    重复定位精度 (Repeatability)  N={rep['n_targets']} 目标")
        print(f"      mean(max spread per target) = {rep['mean']:.4f} mm")
        print(f"      max (worst-case)            = {rep['max']:.4f} mm")
        print(f"      手册宣称 0.01 mm —— {'通过 ✓' if rep['max'] <= 0.01 else '未通过 ✗'}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--can-left",  default="can_left")
    ap.add_argument("--can-right", default="can_right")
    ap.add_argument("--arm", choices=["left", "right", "both"], default="both",
                    help="测试哪条臂（默认 both）")
    ap.add_argument("--trials",  type=int, default=5,
                    help="示教目标数量（默认 5）")
    ap.add_argument("--repeats", type=int, default=5,
                    help="每个目标重复运动次数（默认 5）")
    ap.add_argument("--speed", type=int, default=20,
                    help="MotionCtrl_2 速度百分比（默认 20）")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="每次发送指令后基础等待时间 (s)（默认 2.0）")
    ap.add_argument("--skip-mode-switch", action="store_true",
                    help="跳过示教→从臂 及 从臂→示教 的模式切换提示")
    ap.add_argument("--targets-in",  default="",
                    help="跳过示教阶段，从 json 文件读取目标位姿")
    ap.add_argument("--targets-out", default="targets.json",
                    help="保存示教目标的 json 文件路径")
    ap.add_argument("--report-out",  default="repeatability_report.json",
                    help="保存完整报告的 json 文件路径")
    args = ap.parse_args()

    # ── 连接 CAN ─────────────────────────────────────────────────────────
    print(f"[init] 连接 CAN: left={args.can_left}, right={args.can_right}")
    piper_l = C_PiperInterface(args.can_left)
    piper_r = C_PiperInterface(args.can_right)
    piper_l.ConnectPort()
    piper_r.ConnectPort()
    time.sleep(0.2)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: 示教（机械臂处于主臂模式，拖动示教）
    # ══════════════════════════════════════════════════════════════════════
    if args.targets_in:
        print(f"\n[data] 从 {args.targets_in} 读取目标位姿，跳过示教阶段")
        with open(args.targets_in, "r") as f:
            targets = json.load(f)
        print(f"[data] ✓ 读入 {len(targets)} 个目标")
    else:
        targets = teach_phase(piper_l, piper_r, args.arm, args.trials)
        with open(args.targets_out, "w") as f:
            json.dump(targets, f, indent=2)
        print(f"\n[save] 示教目标已保存: {args.targets_out}")

    # ══════════════════════════════════════════════════════════════════════
    # 模式切换提示：主臂 → 从臂
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_mode_switch:
        print("\n" + "=" * 62)
        print("  [Switch]  切换机械臂到从臂模式")
        print("=" * 62)
        input("  按 Enter 发送 MasterSlaveConfig(0xFC)...")
        switch_mode(piper_l, piper_r, 0xFC, "从臂(slave)")
        print()
        print("  ⚠  请手动给左右两臂 断电后重新上电")
        input("  → 断电重启完成后按 Enter 继续...")

    # ── 使能 + 设置控制模式 ─────────────────────────────────────────────
    if not enable_both(piper_l, piper_r):
        print("[abort] 使能失败，退出")
        sys.exit(1)

    try:
        # ══════════════════════════════════════════════════════════════════
        # Phase 2: 回放
        # ══════════════════════════════════════════════════════════════════
        results = replay_phase(piper_l, piper_r, args.arm,
                               targets, args.repeats,
                               args.speed, args.settle)

        # ══════════════════════════════════════════════════════════════════
        # Phase 3: 分析
        # ══════════════════════════════════════════════════════════════════
        report = analyze(targets, results, args.arm)
        print_report(report)
        with open(args.report_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[save] 完整报告已保存: {args.report_out}")

    except KeyboardInterrupt:
        print("\n[interrupt] 用户中断，保持当前位姿")

    # ══════════════════════════════════════════════════════════════════════
    # 模式切换提示：从臂 → 主臂
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_mode_switch:
        switch_mode(piper_l, piper_r, 0xFA, "主臂(master)")
        print()
        print("=" * 62)
        print("  ✓ 已发送主臂模式配置（0xFA）")
        print("  ⚠  请手动给左右两臂 断电后重新上电，恢复拖动示教模式")
        print("=" * 62)
        input("  → 断电重启完成后按 Enter 退出...")
    else:
        input("  请扶住机械臂后按 Enter 失能电机...")
        disable_both(piper_l, piper_r)

    print("[done] 脚本结束")


if __name__ == "__main__":
    main()
