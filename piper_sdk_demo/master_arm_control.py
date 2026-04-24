#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Piper 主臂控制 / 回零脚本
========================

关键事实
--------
* 主臂(MasterSlaveConfig=0xFA)是被动臂: 静止时不发 CAN, 人手拖动才发 0x155
  关节控制帧。SDK 的 GetArmJointCtrl() 读到的是这些"控制指令镜像", 所以
  静止时输出 0, 拖动时输出真实角度。
* 在 0xFA 模式下电机驱动器**不工作**, EnableArm 永远超时, 因此无法用
  软件命令把主臂"驱动"到任何位置。
* 想让主臂能执行 JointCtrl 动作, 必须先切到从臂模式 (0xFC) 并断电重启。
  以下脚本支持完整流程。

完整工作流 (推荐)
-----------------
    # Step 1: 把主臂切换为从臂模式 (CAN 帧写一次, 重启后生效)
    python master_arm_control.py --can can_left --to-slave

    # Step 2: 用户手动给该机械臂断电 → 重新上电
    #         (CAN 模块不需要重启, 只重启机械臂)

    # Step 3: 验证它现在是从臂 (应该看到 0x2A5 等帧, EnableArm 能成功)
    python master_arm_control.py --can can_left --probe

    # Step 4a: 驱动到零位
    python master_arm_control.py --can can_left --send-joints 0,0,0,0,0,0

    # Step 4b: 任意关节目标 (单位: 度), 可选第 7 项为夹爪开口 (mm)
    python master_arm_control.py --can can_left --send-joints 10,20,-30,0,0,0
    python master_arm_control.py --can can_left --send-joints 0,30,-60,0,0,0,50 --speed 30

    # Step 5 (可选): 用完想恢复成主臂以做示教
    python master_arm_control.py --can can_left --to-master
    # 然后再次断电重启

只读 / 调试模式
---------------
    python master_arm_control.py --can can_left --probe       # 打印模式判断证据
    python master_arm_control.py --can can_left --once        # 只读一次关节
    python master_arm_control.py --can can_left               # 实时引导手动归零

注意
----
* 运行前确保 ROS 节点 start_master_aloha.launch 已停止, 否则会抢 CAN。
* --send-joints 在从臂模式下才有效; 主臂模式下使能失败后会安全退出。
* --to-slave / --to-master 必须断电重启机械臂才生效。
"""
from __future__ import annotations

import argparse
import math
import time

from piper_sdk import C_PiperInterface

RAD_TO_RAW = 1000.0 * 180.0 / math.pi
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]

# 关节限位 (度), 来自 SDK docstring
JOINT_LIMITS_DEG = [
    (-150.0, 150.0),
    (   0.0, 180.0),
    (-170.0,   0.0),
    (-100.0, 100.0),
    ( -70.0,  70.0),
    (-120.0, 120.0),
]


# ----------------------------------------------------------------------
# 读取 / 显示
# ----------------------------------------------------------------------
def read_joint_ctrl_rad(piper):
    """读 GetArmJointCtrl: 主臂模式下有人手拖动数据, 从臂模式下是上一次发送的控制指令。"""
    jc = piper.GetArmJointCtrl().joint_ctrl
    raws = [jc.joint_1, jc.joint_2, jc.joint_3, jc.joint_4, jc.joint_5, jc.joint_6]
    return [r / 1000.0 * math.pi / 180.0 for r in raws]


def read_joint_msgs_rad(piper):
    """读 GetArmJointMsgs: 从臂模式下的实际关节反馈 (0x2A5-0x2A7)。"""
    jm = piper.GetArmJointMsgs().joint_state
    raws = [jm.joint_1, jm.joint_2, jm.joint_3, jm.joint_4, jm.joint_5, jm.joint_6]
    return [r / 1000.0 * math.pi / 180.0 for r in raws]


def read_gripper_m(piper):
    return piper.GetArmGripperCtrl().gripper_ctrl.grippers_angle / 1_000_000.0


def fmt(joints):
    return "  ".join(
        f"{n}={math.degrees(j):+7.2f}d" for n, j in zip(JOINT_NAMES, joints)
    )


# ----------------------------------------------------------------------
# 模式 A: 手动归零引导
# ----------------------------------------------------------------------
def manual_home(piper, tol_deg, once):
    print(f"[manual] tolerance = {tol_deg} deg. drag each joint to 0 by hand.")
    tol_rad = tol_deg * math.pi / 180.0
    last = 0.0
    while True:
        joints = read_joint_ctrl_rad(piper)
        gripper = read_gripper_m(piper)
        diffs = [abs(j) for j in joints]
        ok = all(d <= tol_rad for d in diffs)
        now = time.time()
        if now - last >= 0.2:
            need = ",".join(JOINT_NAMES[i] for i, d in enumerate(diffs) if d > tol_rad)
            tag = "OK" if ok else "  "
            print(f"[{tag}] {fmt(joints)}  gripper={gripper*1000:5.1f}mm  todo=[{need}]")
            last = now
        if once:
            return
        if ok:
            print("[manual] all joints near zero, exit")
            return
        time.sleep(0.05)


# ----------------------------------------------------------------------
# Probe: 判断当前是主臂还是从臂模式
# ----------------------------------------------------------------------
def probe(piper, secs=2.0):
    print(f"[probe] sampling {secs}s of CAN traffic...")
    samples_ctrl = []
    samples_msgs = []
    motor_enable_seen = False
    t0 = time.time()
    while time.time() - t0 < secs:
        samples_ctrl.append(read_joint_ctrl_rad(piper))
        samples_msgs.append(read_joint_msgs_rad(piper))
        st = piper.GetArmLowSpdInfoMsgs()
        if any([
            st.motor_1.foc_status.driver_enable_status,
            st.motor_2.foc_status.driver_enable_status,
        ]):
            motor_enable_seen = True
        time.sleep(0.05)

    def nonzero(samples):
        return any(any(abs(v) > 1e-6 for v in s) for s in samples)

    ctrl_active = nonzero(samples_ctrl)
    msgs_active = nonzero(samples_msgs)
    status = piper.GetArmStatus()
    print(f"[probe] GetArmJointCtrl active : {ctrl_active}   (master 模式典型为 True 当人手拖动时)")
    print(f"[probe] GetArmJointMsgs active : {msgs_active}   (slave  模式典型为 True 持续上报)")
    print(f"[probe] motor low-spd 上报     : {motor_enable_seen}  (slave 模式才会上报)")
    print(f"[probe] arm_status 控制模式    : {status.arm_status.ctrl_mode}")
    if msgs_active or motor_enable_seen:
        print("[probe] 结论: 该臂处于 **从臂(0xFC)** 模式, 可以使能并 JointCtrl")
    else:
        print("[probe] 结论: 该臂可能仍处于 **主臂(0xFA)** 模式, 或电源未上 / 未激活。")
        print("        请确认: 1) 机械臂已上电  2) 已经 --to-slave 并断电重启过")


# ----------------------------------------------------------------------
# 模式切换
# ----------------------------------------------------------------------
def switch_to_slave(piper):
    print("[mode] sending MasterSlaveConfig(0xFC) -> slave (motion-output)")
    for _ in range(3):
        piper.MasterSlaveConfig(0xFC, 0, 0, 0)
        time.sleep(0.2)
    print("[mode] DONE. *** 现在请手动给机械臂断电再上电 ***")
    print("       重启后用 --probe 验证, 然后用 --send-joints 控制")


def switch_to_master(piper):
    print("[mode] sending MasterSlaveConfig(0xFA) -> master (teach-input)")
    for _ in range(3):
        piper.MasterSlaveConfig(0xFA, 0, 0, 0)
        time.sleep(0.2)
    print("[mode] DONE. *** 现在请手动给机械臂断电再上电 ***")


# ----------------------------------------------------------------------
# 使能 + 发送目标
# ----------------------------------------------------------------------
def enable_arm(piper, timeout=5.0):
    print("[enable] enabling motors ...")
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
        print(f"[enable] flags={flags} all={ok}")
        if ok:
            return True
        if time.time() - t0 > timeout:
            print("[enable] TIMEOUT.")
            print("        如果该臂仍是主臂(0xFA)模式则无法使能, 请先 --to-slave 并断电重启")
            return False
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(0.5)


def parse_joints_arg(s):
    """Parse '10,20,-30,0,0,0' or with optional 7th gripper-mm: '10,20,...,50'."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) not in (6, 7):
        raise ValueError("expected 6 or 7 comma-separated numbers (deg, optional gripper mm)")
    nums = [float(p) for p in parts]
    joints_deg = nums[:6]
    gripper_mm = nums[6] if len(nums) == 7 else None
    return joints_deg, gripper_mm


def clamp_with_warn(joints_deg):
    out = []
    for i, (v, (lo, hi)) in enumerate(zip(joints_deg, JOINT_LIMITS_DEG)):
        if v < lo or v > hi:
            print(f"[limit] joint{i+1}={v}deg out of [{lo},{hi}], CLAMPED")
            v = max(lo, min(hi, v))
        out.append(v)
    return out


def send_joints(piper, joints_deg, gripper_mm, speed_pct, hold_sec):
    if not enable_arm(piper):
        return
    joints_deg = clamp_with_warn(joints_deg)
    raws = [int(round(d * 1000)) for d in joints_deg]   # 0.001 度
    print(f"[send] target deg = {joints_deg}")
    print(f"[send] target raw = {raws}")
    print(f"[send] MotionCtrl_2(0x01, 0x01, {speed_pct}, 0x00) -> CAN+MoveJ")
    piper.MotionCtrl_2(0x01, 0x01, int(speed_pct), 0x00)
    piper.JointCtrl(*raws)
    if gripper_mm is not None:
        # 夹爪 raw 单位是 0.001 mm => mm * 1000
        g_raw = int(round(gripper_mm * 1000))
        print(f"[send] GripperCtrl(angle={g_raw}, effort=1000)")
        piper.GripperCtrl(g_raw, 1000, 0x01, 0)
    print(f"[send] holding {hold_sec}s, monitoring feedback...")
    t0 = time.time()
    last = 0.0
    while time.time() - t0 < hold_sec:
        if time.time() - last >= 0.2:
            cur = read_joint_msgs_rad(piper)
            print(f"  msgs: {fmt(cur)}")
            last = time.time()
        time.sleep(0.05)
    print("[send] done. (motors stay enabled - run --disable to release)")


def disable_arm(piper):
    for _ in range(5):
        piper.DisableArm(7)
        piper.GripperCtrl(0, 1000, 0x02, 0)
        time.sleep(0.1)
    print("[disable] done")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--can", default="can_left")
    ap.add_argument("--once", action="store_true", help="只读一次关节后退出")
    ap.add_argument("--tolerance-deg", type=float, default=2.0,
                    help="手动归零的容差")
    ap.add_argument("--speed", type=float, default=20.0,
                    help="--send-joints 时的运动速度百分比 (0-100)")
    ap.add_argument("--hold", type=float, default=5.0,
                    help="--send-joints 后保持监控的秒数")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--to-slave", action="store_true",
                   help="切到从臂模式 (之后手动断电重启)")
    g.add_argument("--to-master", action="store_true",
                   help="切回主臂模式 (之后手动断电重启)")
    g.add_argument("--probe", action="store_true",
                   help="判断当前模式 (主臂 or 从臂)")
    g.add_argument("--send-joints", type=str, metavar="J1,...,J6[,GRIP_MM]",
                   help="向从臂发送关节目标 (单位: 度)")
    g.add_argument("--disable", action="store_true",
                   help="发送 DisableArm 失能电机")
    args = ap.parse_args()

    print(f"[main] connecting to {args.can}")
    piper = C_PiperInterface(args.can)
    piper.ConnectPort()
    time.sleep(0.5)

    if args.to_slave:
        switch_to_slave(piper)
    elif args.to_master:
        switch_to_master(piper)
    elif args.probe:
        probe(piper)
    elif args.send_joints:
        joints_deg, gripper_mm = parse_joints_arg(args.send_joints)
        send_joints(piper, joints_deg, gripper_mm, args.speed, args.hold)
    elif args.disable:
        disable_arm(piper)
    else:
        try:
            manual_home(piper, args.tolerance_deg, args.once)
        except KeyboardInterrupt:
            print("\n[main] interrupted")


if __name__ == "__main__":
    main()
