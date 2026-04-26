# -- coding: UTF-8
"""
Pedal/Enter-controlled master-slave + camera data collection.

    Observation (qpos/qvel/effort) <- 双从臂 (puppet)
    Action <- 双主臂 (master) 操作者下发指令

Workflow:
    1. Start process and wait for ROS topics to become ready.
    2. Lamp shows IDLE (cyan).
    3. Press pedal once (mapped to Enter) to start recording, lamp turns green.
    4. Press pedal again to stop recording and save, lamp turns yellow while saving.
    5. Saving completes, lamp returns to idle and waits for the next episode.

This process stays alive and can collect multiple episodes without restarting.
"""

import argparse
import glob
import os
import threading
import time
from collections import deque

import cv2
import h5py
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState

from dataarm_notifier import RecordingController


def encode_jpeg(image, quality):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, encoded = cv2.imencode(".jpg", image, encode_params)
    if not ok:
        raise RuntimeError("failed to encode image to jpeg")
    return np.frombuffer(encoded.tobytes(), dtype=np.uint8)


def save_episode(args, timesteps, actions, dataset_path):
    data_size = len(actions)
    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/effort": [],
        "/action": [],
        "/base_action": [],
    }
    for cam_name in args.camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict["/observations/qpos"].append(ts["qpos"])
        data_dict["/observations/qvel"].append(ts["qvel"])
        data_dict["/observations/effort"].append(ts["effort"])
        data_dict["/action"].append(action)
        data_dict["/base_action"].append(np.array([0.0, 0.0], dtype=np.float32))
        for cam_name in args.camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(
                encode_jpeg(ts["images"][cam_name], args.jpeg_quality)
            )

    t0 = time.time()
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = True
        root.attrs["image_codec"] = "jpeg"
        root.attrs["jpeg_quality"] = int(args.jpeg_quality)
        root.attrs["image_shape"] = np.array([480, 640, 3], dtype=np.int32)

        obs = root.create_group("observations")
        image_grp = obs.create_group("images")
        image_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        for cam_name in args.camera_names:
            image_grp.create_dataset(cam_name, (data_size,), dtype=image_dtype)

        obs.create_dataset("qpos", (data_size, 14))
        obs.create_dataset("qvel", (data_size, 14))
        obs.create_dataset("effort", (data_size, 14))
        root.create_dataset("action", (data_size, 14))
        root.create_dataset("base_action", (data_size, 2))

        for name, array in data_dict.items():
            root[name][...] = array

    print(f"\033[32mSaved {dataset_path}.hdf5 in {time.time() - t0:.1f}s\033[0m")


def resolve_start_episode_idx(dataset_dir, task_name, requested_idx):
    if requested_idx >= 0:
        return requested_idx

    task_dir = os.path.join(dataset_dir, task_name)
    if not os.path.isdir(task_dir):
        return 0

    max_idx = -1
    pattern = os.path.join(task_dir, "episode_*.hdf5")
    for path in glob.glob(pattern):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            idx = int(name.split("_")[-1])
            max_idx = max(max_idx, idx)
        except ValueError:
            continue
    return max_idx + 1


class PedalControlledCollector:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()

        self.img_front_deque = deque()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.master_arm_left_deque = deque()
        self.master_arm_right_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()

        self.state_lock = threading.Lock()
        self.is_recording = False
        self.stop_requested = False
        self.current_episode = None
        self.next_episode_idx = resolve_start_episode_idx(
            args.dataset_dir, args.task_name, args.episode_idx
        )

        self.controller = RecordingController(
            port=args.lamp_port,
            pedal_device=args.pedal_device,
            trigger_key=args.trigger_key,
        )
        self.controller.on_recording_start(self.handle_recording_start)
        self.controller.on_recording_stop(self.handle_recording_stop)

        rospy.init_node("record_episodes_master_slave_cam_pedal", anonymous=True)

        rospy.Subscriber(args.img_front_topic, Image, self._img_front_cb, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(args.img_left_topic, Image, self._img_left_cb, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(args.img_right_topic, Image, self._img_right_cb, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(
            args.master_arm_left_topic, JointState, self._master_left_cb, queue_size=1000, tcp_nodelay=True
        )
        rospy.Subscriber(
            args.master_arm_right_topic, JointState, self._master_right_cb, queue_size=1000, tcp_nodelay=True
        )
        rospy.Subscriber(
            args.puppet_arm_left_topic, JointState, self._puppet_left_cb, queue_size=1000, tcp_nodelay=True
        )
        rospy.Subscriber(
            args.puppet_arm_right_topic, JointState, self._puppet_right_cb, queue_size=1000, tcp_nodelay=True
        )

    def _img_front_cb(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def _img_left_cb(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def _img_right_cb(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def _master_left_cb(self, msg):
        if len(self.master_arm_left_deque) >= 2000:
            self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)

    def _master_right_cb(self, msg):
        if len(self.master_arm_right_deque) >= 2000:
            self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)

    def _puppet_left_cb(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def _puppet_right_cb(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def wait_until_ready(self):
        print("Waiting for cameras and master-arm topics...")
        rate = rospy.Rate(50)
        t0 = time.time()
        while not rospy.is_shutdown():
            ready = (
                len(self.img_front_deque) > 0
                and len(self.img_left_deque) > 0
                and len(self.img_right_deque) > 0
                and len(self.master_arm_left_deque) > 0
                and len(self.master_arm_right_deque) > 0
                and len(self.puppet_arm_left_deque) > 0
                and len(self.puppet_arm_right_deque) > 0
            )
            if ready:
                print("\033[32mAll streams are ready.\033[0m")
                break
            if time.time() - t0 > 15.0:
                missing = []
                if len(self.img_front_deque) == 0:
                    missing.append(self.args.img_front_topic)
                if len(self.img_left_deque) == 0:
                    missing.append(self.args.img_left_topic)
                if len(self.img_right_deque) == 0:
                    missing.append(self.args.img_right_topic)
                if len(self.master_arm_left_deque) == 0:
                    missing.append(self.args.master_arm_left_topic)
                if len(self.master_arm_right_deque) == 0:
                    missing.append(self.args.master_arm_right_topic)
                if len(self.puppet_arm_left_deque) == 0:
                    missing.append(self.args.puppet_arm_left_topic)
                if len(self.puppet_arm_right_deque) == 0:
                    missing.append(self.args.puppet_arm_right_topic)
                raise RuntimeError(f"Timed out waiting for topics: {missing}")
            rate.sleep()

        print("Waiting for stable master/puppet CAN data...")
        t_can = time.time()
        while not rospy.is_shutdown():
            if (self._arm_valid(self.master_arm_left_deque)
                    and self._arm_valid(self.master_arm_right_deque)
                    and self._arm_valid(self.puppet_arm_left_deque)
                    and self._arm_valid(self.puppet_arm_right_deque)):
                print("\033[32mMaster/puppet CAN data is valid.\033[0m")
                return
            if time.time() - t_can > 5.0:
                print("[WARN] Joint data still near zero; continue anyway.")
                return
            rate.sleep()

    @staticmethod
    def _arm_valid(dq):
        if not dq:
            return False
        pos = list(dq[-1].position)
        return any(abs(v) > 1e-4 for v in pos[:6])

    def get_synced_frame(self):
        if (
            len(self.img_front_deque) == 0
            or len(self.img_left_deque) == 0
            or len(self.img_right_deque) == 0
            or len(self.master_arm_left_deque) == 0
            or len(self.master_arm_right_deque) == 0
            or len(self.puppet_arm_left_deque) == 0
            or len(self.puppet_arm_right_deque) == 0
        ):
            return None

        frame_time = min(
            self.img_front_deque[-1].header.stamp.to_sec(),
            self.img_left_deque[-1].header.stamp.to_sec(),
            self.img_right_deque[-1].header.stamp.to_sec(),
            self.master_arm_left_deque[-1].header.stamp.to_sec(),
            self.master_arm_right_deque[-1].header.stamp.to_sec(),
            self.puppet_arm_left_deque[-1].header.stamp.to_sec(),
            self.puppet_arm_right_deque[-1].header.stamp.to_sec(),
        )

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), "bgr8")

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), "bgr8")

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), "bgr8")

        while self.master_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_left_deque.popleft()
        master_left = self.master_arm_left_deque.popleft()

        while self.master_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_right_deque.popleft()
        master_right = self.master_arm_right_deque.popleft()

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_right = self.puppet_arm_right_deque.popleft()

        img_front = cv2.resize(img_front, (640, 480))
        img_left = cv2.resize(img_left, (640, 480))
        img_right = cv2.resize(img_right, (640, 480))
        return img_front, img_left, img_right, master_left, master_right, puppet_left, puppet_right

    def handle_recording_start(self):
        with self.state_lock:
            if self.is_recording:
                return
            self.current_episode = {
                "timesteps": [],
                "actions": [],
                "count": 0,
                "print_flag": True,
                "episode_idx": self.next_episode_idx,
            }
            self.is_recording = True
            self.stop_requested = False
            print(
                f"\033[32mRecording started: episode_{self.current_episode['episode_idx']} "
                f"(max {self.args.max_timesteps} steps)\033[0m"
            )

    def handle_recording_stop(self):
        with self.state_lock:
            if self.is_recording:
                self.stop_requested = True
                print("\033[33mStop requested. Saving current episode...\033[0m")

    def finalize_current_episode(self):
        with self.state_lock:
            episode = self.current_episode
            self.current_episode = None
            self.is_recording = False
            self.stop_requested = False

        if episode is None:
            self.controller.notifier.idle()
            return

        actions = episode["actions"]
        timesteps = episode["timesteps"]
        episode_idx = episode["episode_idx"]
        if len(actions) == 0:
            print("[WARN] Episode has no actions. Discarded.")
            self.controller.notifier.idle()
            return

        dataset_dir = os.path.join(self.args.dataset_dir, self.args.task_name)
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}")
        save_episode(self.args, timesteps, actions, dataset_path)
        self.next_episode_idx += 1
        self.controller.notifier.idle()
        print(f"[INFO] Ready for next episode. Next index: {self.next_episode_idx}")

    def append_recording_step(self, result):
        (img_front, img_left, img_right,
         master_left, master_right,
         puppet_left, puppet_right) = result

        # observation <- 从臂 (puppet) 实际状态
        qpos = np.concatenate((np.array(puppet_left.position), np.array(puppet_right.position)), axis=0)
        qvel = np.concatenate((np.array(puppet_left.velocity), np.array(puppet_right.velocity)), axis=0)
        effort = np.concatenate((np.array(puppet_left.effort), np.array(puppet_right.effort)), axis=0)

        # action <- 主臂 (master) 下发指令
        action = np.concatenate((np.array(master_left.position), np.array(master_right.position)), axis=0)

        obs = {
            "qpos": qpos,
            "qvel": qvel,
            "effort": effort,
            "images": {
                self.args.camera_names[0]: img_front,
                self.args.camera_names[1]: img_left,
                self.args.camera_names[2]: img_right,
            },
        }

        with self.state_lock:
            episode = self.current_episode
            if episode is None:
                return False

            episode["count"] += 1
            count = episode["count"]
            if count == 1:
                episode["timesteps"].append(obs)
                return False

            episode["actions"].append(action)
            episode["timesteps"].append(obs)
            if count % 50 == 0 or count == self.args.max_timesteps + 1:
                print(f"  Recorded: {count - 1}/{self.args.max_timesteps}")

            if len(episode["actions"]) >= self.args.max_timesteps:
                print("[INFO] Reached max_timesteps; saving episode automatically.")
                self.stop_requested = True
                return True
            return False

    def run(self):
        self.wait_until_ready()
        try:
            self.controller.start()
        except Exception as exc:
            print(f"[WARN] pedal/lamp unavailable: {exc}. Use keyboard Enter to control recording.")
        print(
            "\nPedal-controlled collector is ready.\n"
            "Press pedal/Enter once to start recording, press again to stop and save.\n"
        )

        rate = rospy.Rate(self.args.frame_rate)
        try:
            while not rospy.is_shutdown():
                if not self.is_recording:
                    rate.sleep()
                    continue

                if self.stop_requested:
                    self.finalize_current_episode()
                    rate.sleep()
                    continue

                result = self.get_synced_frame()
                if result is None:
                    rate.sleep()
                    continue

                should_finalize = self.append_recording_step(result)
                if should_finalize or self.stop_requested:
                    self.finalize_current_episode()
                rate.sleep()
        finally:
            self.controller.stop()


def get_arguments():
    parser = argparse.ArgumentParser(description="Pedal-controlled master-arm + camera collection")
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--task_name", type=str, default="aloha_master_cam")
    parser.add_argument("--episode_idx", type=int, default=-1)
    parser.add_argument("--max_timesteps", type=int, default=500)
    parser.add_argument(
        "--camera_names",
        nargs="+",
        type=str,
        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
    )
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--master_arm_left_topic", type=str, default="/master/joint_left")
    parser.add_argument("--master_arm_right_topic", type=str, default="/master/joint_right")
    parser.add_argument("--puppet_arm_left_topic", type=str, default="/puppet/joint_left")
    parser.add_argument("--puppet_arm_right_topic", type=str, default="/puppet/joint_right")
    parser.add_argument("--frame_rate", type=int, default=30)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--lamp_port", type=str, default=None)
    parser.add_argument(
        "--pedal_device",
        type=str,
        default=None,
    )
    parser.add_argument("--trigger_key", type=str, default="enter")
    return parser.parse_args()


def main():
    args = get_arguments()
    collector = PedalControlledCollector(args)
    collector.run()


if __name__ == "__main__":
    main()
