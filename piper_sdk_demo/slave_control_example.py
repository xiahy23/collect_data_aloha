
#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Piper 从臂（运动输出臂）控制示例
================================

该脚本演示如何用 piper_sdk 直接通过 CAN 总线控制 Piper 机械臂运动。
本示例假定：
    - 目标 CAN 端口为 `can_left`（如需控制右臂改成 `can_right`）
    - 该 CAN 上挂载的是从臂（已通过 piper_slave_config.py 配置为 0xFC，
      或本来就是出厂从臂模式的机械臂）。
    - 主臂航插已断开（否则主臂会持续覆盖控制指令）。

用法:
    conda activate aloha
    python slave_control_example.py --can can_left --demo joint
    python slave_control_example.py --can can_left --demo gripper
    python slave_control_example.py --can can_left --demo home
    python slave_control_example.py --can can_left --demo teach   # 进入拖动示教(软件重力补偿)
    python slave_control_example.py --can can_left --demo disable

核心 API（来自 piper_sdk.C_PiperInterface）:
    ConnectPort()                       建立 CAN 连接 + 启动后台读线程
    EnableArm(7)                        使能全部 6 个电机（7=ALL）
    DisableArm(7)                       失能全部电机
    MotionCtrl_2(ctrl_mode, move_mode, spd, mit)
                                        切换控制模式 (CAN指令/关节模式/速度百分比)
        ctrl_mode  : 0x00 待机, 0x01 CAN 指令控制
        move_mode  : 0x00 MoveP, 0x01 MoveJ, 0x02 MoveL, 0x03 MoveC
        spd        : 0~100  运动速度百分比
        is_mit_mode: 0x00 位置-速度, 0xAD MIT, 0xFF 无效
    JointCtrl(j1..j6)                   六关节角度，单位 0.001 度
    EndPoseCtrl(X,Y,Z,RX,RY,RZ)         末端位姿（mm/0.001°），需先 ModeCtrl(MoveL/P/C)
    GripperCtrl(angle, effort, code, set_zero)
        angle  : 0.001°
        effort : 0~5000 (对应 0-5 N·m)
        code   : 0x00 失能 / 0x01 使能 / 0x02 失能并清错 / 0x03 使能并清错
    MotionCtrl_1(estop, track, grag_teach_ctrl)
        grag_teach_ctrl: 0x01 进入拖动示教, 0x02 退出, 0x03 复现轨迹 ...
    MasterSlaveConfig(linkage, fb_off, ctrl_off, link_off)
        linkage: 0xFA 设为主臂(示教输入), 0xFC 设为从臂(运动输出)

坐标/单位换算:
    rad -> SDK 单位:  raw = round(angle_rad * 1000 * 180 / pi)  ≈ angle_rad * 57295.78
    SDK 单位 -> rad:  rad = raw / 1000 * pi / 180               ≈ raw * 0.017444 / 1000
    夹爪角度 (m):     raw_mm = round(angle_m * 1000 * 1000)     # piper 文档单位 0.001 mm
"""
from __future__ import annotations

import argparse
import math
import time

from piper_sdk import C_PiperInterface

# ---- 单位换算常量 ----
# JointCtrl 输入单位是 0.001 deg → factor 把 rad 转成该单位
RAD_TO_RAW = 1000.0 * 180.0 / math.pi   # ≈ 57295.78


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def enable_arm(piper: C_PiperInterface, timeout: float = 5.0) -> bool:
    """循环发送 EnableArm 直到 6 个电机的 driver_enable_status 全部为 True。"""
    print("[enable] 开始使能...")
    t0 = time.time()
    while True:
        st = piper.GetArmLowSpdInfoMsgs()
        flags = [
            st.motor_1.foc_status.driver_enable_status,
            st.motor_2.foc_status.driver_enable_status,
            st.motor_3.foc_status.driver_enable_status,
            st.motor_4.foc_status.driver_enable_status,
            st.motor_5.foc_status.driver_enable_status,
            st.motor_6.foc_status.driver_enable_status,
        ]
        ok = all(flags)
        print(f"[enable] motor_enable={flags} all={ok}")
        if ok:
            print("[enable] 使能完成")
            return True
        if time.time() - t0 > timeout:
            print("[enable] 超时, 放弃")
            return False
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(0.5)


def disable_arm(piper: C_PiperInterface) -> None:
    print("[disable] 失能机械臂...")
    for _ in range(5):
        piper.DisableArm(7)
        piper.GripperCtrl(0, 1000, 0x02, 0)
        time.sleep(0.1)
    print("[disable] done")


def joint_rad_to_raw(joints_rad):
    return [int(round(j * RAD_TO_RAW)) for j in joints_rad]


# ----------------------------------------------------------------------
# Demo: 关节运动（在零位附近做一个小幅来回）
# ----------------------------------------------------------------------
def demo_joint(piper: C_PiperInterface):
    if not enable_arm(piper):
        return
    targets_rad = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.2, -0.2, 0.3, -0.2, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    for tgt in targets_rad:
        raw = joint_rad_to_raw(tgt)
        print(f"[joint] target rad={tgt}  raw={raw}")
        # 0x01 CAN指令控制, 0x01 MoveJ, 30% 速度
        piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        piper.JointCtrl(*raw)
        # 等待运动到位（简单等待，没做反馈判断）
        time.sleep(3.0)
    print("[joint] done")


# ----------------------------------------------------------------------
# Demo: 回零位
# ----------------------------------------------------------------------
def demo_home(piper: C_PiperInterface):
    if not enable_arm(piper):
        return
    piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)
    piper.JointCtrl(0, 0, 0, 0, 0, 0)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    print("[home] 已发送回零指令")
    time.sleep(4.0)


# ----------------------------------------------------------------------
# Demo: 夹爪开合
# ----------------------------------------------------------------------
def demo_gripper(piper: C_PiperInterface):
    if not enable_arm(piper):
        return
    # 张开 ~70mm  → raw = 0.07 * 1e6 = 70000
    print("[gripper] 张开")
    piper.GripperCtrl(70_000, 1000, 0x01, 0)
    time.sleep(2.0)
    print("[gripper] 闭合")
    piper.GripperCtrl(0, 1000, 0x01, 0)
    time.sleep(2.0)


# ----------------------------------------------------------------------
# Demo: 拖动示教（软件级重力补偿）
# ----------------------------------------------------------------------
def demo_teach(piper: C_PiperInterface):
    """对从臂启用拖动示教模式 → 软件级别的重力补偿。
    主臂物理上已经平衡，不需要这条；从臂没有配重，必须靠这个才能用手拖动。
    """
    if not enable_arm(piper):
        return
    print("[teach] 进入拖动示教模式 (grag_teach_ctrl=0x01)")
    piper.MotionCtrl_1(0x00, 0x00, 0x01)
    print("[teach] 现在用手拖动机械臂，30 秒后自动退出")
    time.sleep(30.0)
    print("[teach] 退出拖动示教 (grag_teach_ctrl=0x02)")
    piper.MotionCtrl_1(0x00, 0x00, 0x02)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
DEMOS = {
    "joint": demo_joint,
    "home": demo_home,
    "gripper": demo_gripper,
    "teach": demo_teach,
    "disable": lambda p: disable_arm(p),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can", default="can_left", help="CAN 端口名: can_left / can_right")
    ap.add_argument("--demo", default="home", choices=list(DEMOS), help="选择演示")
    args = ap.parse_args()

    print(f"[main] connecting to {args.can}")
    piper = C_PiperInterface(args.can)
    piper.ConnectPort()
    time.sleep(0.5)

    try:
        DEMOS[args.demo](piper)
    finally:
        # 安全退出：只在非 disable demo 后失能
        if args.demo not in ("disable", "teach"):
            disable_arm(piper)


if __name__ == "__main__":
    main()
