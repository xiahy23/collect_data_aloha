#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""主从臂同时回零工具脚本 / 可导入库。

依赖 piper_start_ms_node.py 暴露的 ROS Service:
    /can_left/go_zero_master_slave
    /can_right/go_zero_master_slave

调用 Service 后硬件固件自动驱动主从臂回零。本工具:
    1. 调用两侧 go_zero_master_slave Service
    2. 监听 /puppet/joint_left, /puppet/joint_right，等到 6 个臂关节
       位置和速度满足阈值并保持 stable_seconds 秒 → 回零完成
    3. 超时返回 False

默认 timeout=3.0s，init_grace=0.2s，stable_seconds=0.3s——适配实际
硬件 ~2s 内完成回零的情况，最多等 3 秒。
"""

import argparse
import threading
import time

import rospy
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


LEFT_SERVICE   = "/can_left/go_zero_master_slave"
RIGHT_SERVICE  = "/can_right/go_zero_master_slave"
LEFT_RESTORE   = "/can_left/restore_ms_mode"
RIGHT_RESTORE  = "/can_right/restore_ms_mode"
LEFT_TOPIC     = "/puppet/joint_left"
RIGHT_TOPIC    = "/puppet/joint_right"


class _ArmMonitor:
    def __init__(self, topic):
        self.topic = topic
        self._lock = threading.Lock()
        self._pos  = None
        self._vel  = None
        self._sub  = rospy.Subscriber(
            topic, JointState, self._cb, queue_size=1, tcp_nodelay=True
        )

    def _cb(self, msg):
        with self._lock:
            self._pos = list(msg.position)
            self._vel = list(msg.velocity) if msg.velocity else [0.0] * len(self._pos)

    def state(self):
        with self._lock:
            return self._pos, self._vel

    def near_zero_and_still(self, pos_thr, vel_thr):
        with self._lock:
            if self._pos is None or self._vel is None:
                return False
            # 只检查前 6 个臂关节，joint[6] 是夹爪
            pos_ok = all(abs(p) < pos_thr for p in self._pos[:6])
            vel_ok = all(abs(v) < vel_thr for v in self._vel[:6])
            return pos_ok and vel_ok

    def close(self):
        try:
            self._sub.unregister()
        except Exception:
            pass


def _call_service(name, timeout=3.0):
    try:
        rospy.wait_for_service(name, timeout=timeout)
        proxy = rospy.ServiceProxy(name, Trigger)
        resp  = proxy()
        return bool(resp.success), resp.message
    except Exception as exc:
        return False, str(exc)


def go_home_and_wait(
    pos_threshold:  float = 0.05,
    vel_threshold:  float = 0.05,
    stable_seconds: float = 0.3,
    timeout:        float = 3.0,
    init_grace:     float = 0.2,
    poll_hz:        float = 20.0,
    log=print,
):
    """触发主从臂同步回零并等待硬件实际归位完成。

    Args:
        pos_threshold:  关节角阈值 (rad)，6 个臂关节均需 < 该值。
        vel_threshold:  关节速度阈值。
        stable_seconds: 满足条件的持续时间，超过该时长视为完成。
        timeout:        最长等待时间 (s)，默认 3 s。
        init_grace:     Service 调用后的初始空程 (s)，避免还未开始运动就误判完成。
        poll_hz:        状态检查频率。
        log:            日志回调，UI 模式下可传入队列推送函数。

    Returns:
        bool: True = 成功回零并稳定；False = 服务失败或超时。
    """
    log("[home] requesting go_zero_master_slave for both arms ...")
    ok_l, msg_l = _call_service(LEFT_SERVICE)
    ok_r, msg_r = _call_service(RIGHT_SERVICE)
    log(f"[home] left:  ok={ok_l} ({msg_l})")
    log(f"[home] right: ok={ok_r} ({msg_r})")
    if not (ok_l and ok_r):
        return False

    left  = _ArmMonitor(LEFT_TOPIC)
    right = _ArmMonitor(RIGHT_TOPIC)
    reached = False
    try:
        rate        = rospy.Rate(poll_hz)
        t0          = time.time()
        stable_since = None
        last_log    = 0.0

        while not rospy.is_shutdown():
            elapsed = time.time() - t0
            if elapsed > timeout:
                log(f"[home] TIMEOUT after {timeout:.1f}s")
                break

            # 初始空程：service 刚发出，臂还没动，跳过检测
            if elapsed < init_grace:
                rate.sleep()
                continue

            near = (
                left.near_zero_and_still(pos_threshold, vel_threshold)
                and right.near_zero_and_still(pos_threshold, vel_threshold)
            )

            if near:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stable_seconds:
                    log(f"[home] reached home in {elapsed:.1f}s")
                    reached = True
                    break
            else:
                stable_since = None

            # 每 0.2 s 打一次进度日志
            if elapsed - last_log >= 0.2:
                lp, _ = left.state()
                rp, _ = right.state()
                lp_s  = "?" if lp is None else "[" + ",".join(f"{p:+.2f}" for p in lp[:6]) + "]"
                rp_s  = "?" if rp is None else "[" + ",".join(f"{p:+.2f}" for p in rp[:6]) + "]"
                log(f"[home] t={elapsed:.1f}s L={lp_s} R={rp_s}")
                last_log = elapsed

            rate.sleep()
    finally:
        left.close()
        right.close()

    # 无论成功还是超时，都恢复主从联动模式（示教模式），
    # 否则主臂仍处于位置控制模式，无法被拖拽进行下一轮采集。
    log("[home] restoring master-slave teach mode ...")
    ok_lr, msg_lr = _call_service(LEFT_RESTORE)
    ok_rr, msg_rr = _call_service(RIGHT_RESTORE)
    log(f"[home] restore left:  ok={ok_lr} ({msg_lr})")
    log(f"[home] restore right: ok={ok_rr} ({msg_rr})")
    if not (ok_lr and ok_rr):
        log("[home] WARN: restore_ms_mode failed; master arm may not be in teach mode.")
        return False
    return reached


def main():
    parser = argparse.ArgumentParser(description="主从臂同步回零")
    parser.add_argument("--timeout",        type=float, default=3.0)
    parser.add_argument("--pos_threshold",  type=float, default=0.05)
    parser.add_argument("--vel_threshold",  type=float, default=0.05)
    parser.add_argument("--stable_seconds", type=float, default=0.3)
    args = parser.parse_args()

    rospy.init_node("piper_go_home", anonymous=True)
    ok = go_home_and_wait(
        pos_threshold  = args.pos_threshold,
        vel_threshold  = args.vel_threshold,
        stable_seconds = args.stable_seconds,
        timeout        = args.timeout,
    )
    print("DONE:", "OK" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
