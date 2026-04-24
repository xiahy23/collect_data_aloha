#!/usr/bin/env python3
"""
eval_fk_error.py — 计算 Piper 机械臂 replay 误差

用法:
  python3 eval_fk_error.py --hdf5 ~/data/my_task/episode_0.hdf5

需要同目录下存在 replay 时生成的 episode_0_actual.npy。

输出:
  - 每帧末端位置误差 (mm)
  - 整体统计：RMSE、最大误差、误差分布
  - 可选 --plot 绘制误差曲线
"""

import argparse
import os
import sys

import h5py
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Piper 正运动学（FK）
# 参数来源: piper_ws/src/piper_description/urdf/piper_description.urdf
# ─────────────────────────────────────────────────────────────────────────────

def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def _rpy(r, p, y):
    """URDF 标准 RPY：先 Rx 再 Ry 再 Rz（外旋）"""
    return _rz(y) @ _ry(p) @ _rx(r)

def _T(xyz, rpy):
    """构造 4×4 齐次变换矩阵（固定偏置）"""
    t = np.eye(4)
    t[:3, 3] = xyz
    t[:3, :3] = _rpy(*rpy)
    return t

# URDF 中的关节参数（joint1~joint6）+ 末端固定变换（joint7）
# 格式：(xyz, rpy, axis_sign)
# axis_sign = +1 → 绕 +z 旋转；axis_sign = -1 → 绕 -z 旋转
_JOINT_PARAMS = [
    ([0.0,     0.0,      0.123    ], [0.0,       0.0,      -1.5708  ], +1),  # j1
    ([0.0,     0.0,      0.0      ], [1.5708,   -0.10095,  -1.5708  ], +1),  # j2
    ([0.28503, 0.0,      0.0      ], [0.0,       0.0,       1.3826  ], +1),  # j3
    ([0.021984,0.25075,  0.0      ], [-1.5708,   0.0,       0.0     ], -1),  # j4
    ([0.0,     0.0,      0.0      ], [1.5708,   -0.087266,  0.0     ], +1),  # j5
    ([0.0,     0.091,    0.0014165], [-1.5708,  -1.5708,    0.0     ], -1),  # j6
]

# 末端（tool tip）固定变换，对应 URDF joint7
_EE_T = _T([0.0, 0.0, 0.13503], [1.5708, 0.0, 1.5708])


def fk(joints_rad: np.ndarray) -> np.ndarray:
    """
    Piper 6-DOF 正运动学。

    Parameters
    ----------
    joints_rad : array-like, shape (6,)
        6 个关节角度，单位 rad

    Returns
    -------
    T : ndarray, shape (4, 4)
        末端（tool tip）相对 base_link 的齐次变换矩阵
    """
    T = np.eye(4)
    for xyz, rpy, sign in _JOINT_PARAMS:
        T_fixed = _T(xyz, rpy)
        theta = sign * joints_rad[..., 0] if False else sign * float(joints_rad[_JOINT_PARAMS.index((xyz, rpy, sign))])  # noqa
        T_joint = np.eye(4)
        T_joint[:3, :3] = _rz(theta)
        T = T @ T_fixed @ T_joint
    T = T @ _EE_T
    return T


def fk_batch(joints_rad_batch: np.ndarray) -> np.ndarray:
    """
    批量 FK。

    Parameters
    ----------
    joints_rad_batch : ndarray, shape (N, 6)

    Returns
    -------
    positions : ndarray, shape (N, 3)  单位 m
    """
    N = len(joints_rad_batch)
    positions = np.zeros((N, 3))
    for i, q in enumerate(joints_rad_batch):
        T = np.eye(4)
        for idx, (xyz, rpy, sign) in enumerate(_JOINT_PARAMS):
            T_fixed = _T(xyz, rpy)
            T_joint = np.eye(4)
            T_joint[:3, :3] = _rz(sign * float(q[idx]))
            T = T @ T_fixed @ T_joint
        T = T @ _EE_T
        positions[i] = T[:3, 3]
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# 单位换算工具
# ─────────────────────────────────────────────────────────────────────────────

def raw_0001deg_to_rad(raw):
    """0.001 deg → rad（GetArmJointMsgs 返回值的单位）"""
    return np.asarray(raw, dtype=np.float64) * (np.pi / 180.0 / 1000.0)


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Piper replay FK 误差分析")
    ap.add_argument("--hdf5", required=True, help="采集的 HDF5 文件路径")
    ap.add_argument("--actual", default="",
                    help="实际关节角 .npy 文件路径（默认为 hdf5同名_actual.npy）")
    ap.add_argument("--plot", action="store_true", help="绘制误差曲线")
    ap.add_argument("--arm", choices=["left", "right", "both"], default="both",
                    help="分析哪条臂（默认 both）")
    ap.add_argument("--max-lag", type=int, default=60,
                    help="滞后补偿搜索的最大帧数（默认 60）；设 0 禁用")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="录制帧率，用于将 lag 换算为毫秒（默认 30）")
    args = ap.parse_args()

    # ── 确定文件路径 ─────────────────────────────────────────────────────────
    hdf5_path = os.path.expanduser(args.hdf5)
    if args.actual:
        actual_path = os.path.expanduser(args.actual)
    else:
        actual_path = os.path.splitext(hdf5_path)[0] + "_actual.npy"

    if not os.path.exists(hdf5_path):
        sys.exit(f"[error] HDF5 文件不存在: {hdf5_path}")
    if not os.path.exists(actual_path):
        sys.exit(f"[error] 实际关节角文件不存在: {actual_path}\n"
                 f"  请先运行 replay_on_slave.py（不加 --no-log-actual）")

    # ── 加载数据 ─────────────────────────────────────────────────────────────
    with h5py.File(hdf5_path, "r") as f:
        cmd_actions = f["action"][()].astype(np.float64)  # (T, 14), rad + m

    actual_raw = np.load(actual_path).astype(np.float64)  # (T', 12), 单位 0.001deg
    print(f"[data] 指令帧数: {len(cmd_actions)},  实际记录帧数: {len(actual_raw)}")

    # 对齐帧数（取较短的）
    T = min(len(cmd_actions), len(actual_raw))
    cmd_actions = cmd_actions[:T]
    actual_raw  = actual_raw[:T]

    # 假零头修复（与 replay 脚本一致）
    first_valid = 0
    for i in range(T):
        if np.any(np.abs(cmd_actions[i]) > 1e-3):
            first_valid = i
            break
    if first_valid > 5:
        print(f"[data] 自动跳过前 {first_valid} 帧假零")
        cmd_actions = cmd_actions[first_valid:]
        actual_raw  = actual_raw[first_valid:]
        T -= first_valid

    # ── 拆分左右臂 ──────────────────────────────────────────────────────────
    # cmd: 左臂 j0-5 (rad), 左夹爪 (m), 右臂 j0-5 (rad), 右夹爪 (m)
    cmd_l_rad = cmd_actions[:, :6]       # (T, 6)
    cmd_r_rad = cmd_actions[:, 7:13]     # (T, 6)

    # actual: 左臂 j1-6 raw-0.001deg, 右臂 j1-6 raw-0.001deg
    act_l_rad = raw_0001deg_to_rad(actual_raw[:, :6])   # (T, 6)
    act_r_rad = raw_0001deg_to_rad(actual_raw[:, 6:12]) # (T, 6)

    # ── FK 批量计算 ──────────────────────────────────────────────────────────
    print("[fk] 计算末端位置...")
    p_cmd_l = fk_batch(cmd_l_rad) * 1000.0   # m → mm
    p_cmd_r = fk_batch(cmd_r_rad) * 1000.0
    p_act_l = fk_batch(act_l_rad) * 1000.0
    p_act_r = fk_batch(act_r_rad) * 1000.0

    err_l = np.linalg.norm(p_act_l - p_cmd_l, axis=1)  # (T,) mm
    err_r = np.linalg.norm(p_act_r - p_cmd_r, axis=1)

    # 关节角误差（度）
    joint_err_l = np.abs(np.degrees(act_l_rad - cmd_l_rad))  # (T, 6)
    joint_err_r = np.abs(np.degrees(act_r_rad - cmd_r_rad))

    # ── 滞后补偿误差搜索 ─────────────────────────────────────────────────────
    def find_best_lag(cmd_rad, act_rad, p_cmd_mm, p_act_mm, max_lag):
        """搜索最优时移量，返回 (best_lag, lag_rmse, lag_err_array, lag_jerr)"""
        if max_lag <= 0:
            return 0, np.sqrt(np.mean(np.linalg.norm(p_act_mm - p_cmd_mm, axis=1)**2)),                    np.linalg.norm(p_act_mm - p_cmd_mm, axis=1),                    np.abs(np.degrees(act_rad - cmd_rad))
        best_lag, best_rmse = 0, np.inf
        for lag in range(0, max_lag + 1):
            if lag == 0:
                e = np.linalg.norm(p_act_mm - p_cmd_mm, axis=1)
            else:
                e = np.linalg.norm(p_act_mm[lag:] - p_cmd_mm[:-lag], axis=1)
            rmse = np.sqrt(np.mean(e**2))
            if rmse < best_rmse:
                best_rmse, best_lag = rmse, lag
        if best_lag == 0:
            lag_err = np.linalg.norm(p_act_mm - p_cmd_mm, axis=1)
            lag_jerr = np.abs(np.degrees(act_rad - cmd_rad))
        else:
            lag_err = np.linalg.norm(p_act_mm[best_lag:] - p_cmd_mm[:-best_lag], axis=1)
            lag_jerr = np.abs(np.degrees(act_rad[best_lag:] - cmd_rad[:-best_lag]))
        return best_lag, best_rmse, lag_err, lag_jerr

    lag_l, lag_rmse_l, lag_err_l, lag_jerr_l = find_best_lag(
        cmd_l_rad, act_l_rad, p_cmd_l, p_act_l, args.max_lag)
    lag_r, lag_rmse_r, lag_err_r, lag_jerr_r = find_best_lag(
        cmd_r_rad, act_r_rad, p_cmd_r, p_act_r, args.max_lag)

    # ── 数据有效性检验 ────────────────────────────────────────────────────────
    def check_validity(label, act_rad, cmd_rad):
        warnings = []
        for j in range(6):
            cmd_range = np.degrees(cmd_rad[:, j].max() - cmd_rad[:, j].min())
            act_range = np.degrees(act_rad[:, j].max() - act_rad[:, j].min())
            if cmd_range > 5.0 and act_range < 0.5:
                warnings.append(
                    f"  ⚠  {label} j{j+1}: cmd 变化 {cmd_range:.1f}°，"
                    f"actual 近似静止（{act_range:.2f}°）→ 该关节未跟随指令"
                )
        return warnings

    all_warnings = []
    if args.arm in ("left", "both"):
        all_warnings += check_validity("左臂", act_l_rad, cmd_l_rad)
    if args.arm in ("right", "both"):
        all_warnings += check_validity("右臂", act_r_rad, cmd_r_rad)
    if all_warnings:
        print("\n[warn] 数据有效性检验：")
        for w in all_warnings:
            print(w)

    # ── 打印统计 ─────────────────────────────────────────────────────────────
    def stats(name, err, jerr):
        print(f"\n  ── {name} ──")
        print(f"    帧数                : {len(err)}")
        print(f"    末端位置 RMSE       : {np.sqrt(np.mean(err**2)):.2f} mm")
        print(f"    末端位置 最大误差   : {err.max():.2f} mm  (frame {err.argmax()})")
        print(f"    末端位置 均值       : {err.mean():.2f} mm")
        print(f"    <5mm 的帧占比       : {(err<5).mean()*100:.1f}%")
        print(f"    <10mm 的帧占比      : {(err<10).mean()*100:.1f}%")
        print(f"    <20mm 的帧占比      : {(err<20).mean()*100:.1f}%")
        print(f"\n    关节角误差 (deg):")
        print(f"           {'  j1':>6}{'  j2':>6}{'  j3':>6}{'  j4':>6}{'  j5':>6}{'  j6':>6}")
        print(f"      均值: {' '.join(f'{v:6.2f}' for v in jerr.mean(axis=0))}")
        print(f"      最大: {' '.join(f'{v:6.2f}' for v in jerr.max(axis=0))}")
        bad = [f"j{j+1}({jerr[:,j].max():.1f}°)" for j in range(6) if jerr[:,j].max() > 10.0]
        if bad:
            print(f"      ⚠  超过10°的关节: {', '.join(bad)}")
            print(f"         → 建议提高 --speed 或降低 --fps 以减少跟踪滞后")

    print("\n" + "=" * 60)
    print("  Piper Replay FK 误差报告")
    print("=" * 60)

    arms = []
    if args.arm in ("left", "both"):
        arms.append(("左臂 (left)", err_l, joint_err_l, lag_l, lag_rmse_l, lag_err_l, lag_jerr_l))
    if args.arm in ("right", "both"):
        arms.append(("右臂 (right)", err_r, joint_err_r, lag_r, lag_rmse_r, lag_err_r, lag_jerr_r))

    for name, err, jerr, lag, lag_rmse, lag_err, lag_jerr in arms:
        stats(name, err, jerr)
        if args.max_lag > 0:
            lag_ms = lag / args.fps * 1000
            print(f"\n    ── 滞后补偿误差（最优 lag={lag}帧 / {lag_ms:.0f}ms） ──")
            print(f"    末端位置 RMSE (lag补偿)  : {lag_rmse:.2f} mm")
            print(f"    末端位置 最大误差 (lag补偿): {lag_err.max():.2f} mm")
            print(f"    <5mm 的帧占比  (lag补偿)  : {(lag_err<5).mean()*100:.1f}%")
            print(f"    <10mm 的帧占比 (lag补偿)  : {(lag_err<10).mean()*100:.1f}%")
            if lag_rmse < np.sqrt(np.mean(err**2)) * 0.5:
                print(f"    → 补偿后误差明显减小，主要误差为跟踪滞后（运动跟不上速度）")
            else:
                print(f"    → 补偿后误差改善有限，可能存在定位偏差（零位/标定问题）")

    if args.arm == "both":
        err_both = (err_l + err_r) / 2
        print(f"\n  ── 双臂均值 ──")
        print(f"    末端位置 RMSE   : {np.sqrt(np.mean(err_both**2)):.2f} mm")
        print(f"    末端位置 最大误差: {err_both.max():.2f} mm")

    print("\n" + "=" * 60)

    # ── 可选绘图 ─────────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            t = np.arange(T)

            if args.arm in ("left", "both"):
                axes[0].plot(t, err_l, label="左臂末端误差", color="steelblue")
            if args.arm in ("right", "both"):
                axes[0].plot(t, err_r, label="右臂末端误差", color="tomato")
            axes[0].axhline(5, color="gray", linestyle="--", linewidth=0.8, label="5mm 基准")
            axes[0].axhline(10, color="orange", linestyle="--", linewidth=0.8, label="10mm 基准")
            axes[0].set_ylabel("末端位置误差 (mm)")
            axes[0].set_title("Piper Replay FK 误差分析")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 关节角误差
            for j in range(6):
                if args.arm in ("left", "both"):
                    axes[1].plot(t, joint_err_l[:, j], alpha=0.6,
                                 linestyle="-", label=f"L-j{j+1}")
                if args.arm in ("right", "both"):
                    axes[1].plot(t, joint_err_r[:, j], alpha=0.6,
                                 linestyle="--", label=f"R-j{j+1}")
            axes[1].set_xlabel("帧（Frame Index）")
            axes[1].set_ylabel("关节角误差 (deg)")
            axes[1].legend(ncol=6, fontsize=8)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            out_png = os.path.splitext(hdf5_path)[0] + "_fk_error.png"
            plt.savefig(out_png, dpi=150)
            print(f"[plot] 图像已保存到 {out_png}")
            plt.show()
        except ImportError:
            print("[warn] matplotlib 未安装，跳过绘图")


if __name__ == "__main__":
    main()
