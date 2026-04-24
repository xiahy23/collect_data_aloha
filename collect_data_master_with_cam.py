# -- coding: UTF-8
"""
主臂 + 相机数据采集脚本（无从臂）

用法:
    python collect_data_master_with_cam.py \
        --dataset_dir ~/data --task_name my_task \
        --max_timesteps 500 --episode_idx 0

说明:
    - 采集双主臂关节数据 + 3 个相机 RGB 图像
    - 无从臂：主臂 qpos 同时作为 observation 和 action
    - 输出 HDF5 文件，图像以 JPEG 二进制保存以节省空间
    - 相机话题默认: /camera_f/color/image_raw, /camera_l/color/image_raw, /camera_r/color/image_raw
    - 主臂话题默认: /master/joint_left, /master/joint_right
"""
import os
import time
from collections import deque

import argparse
import cv2
import h5py
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState


def encode_jpeg(image, quality):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, encoded = cv2.imencode(".jpg", image, encode_params)
    if not ok:
        raise RuntimeError("failed to encode image to jpeg")
    return np.frombuffer(encoded.tobytes(), dtype=np.uint8)


def save_data(args, timesteps, actions, dataset_path):
    """保存数据为 HDF5 格式。图像字段保存为 JPEG 二进制。"""
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

    print(f"\033[32m\nSaving: {time.time() - t0:.1f} secs. {dataset_path}.hdf5\033[0m\n")


class RosOperator:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()

        self.img_front_deque = deque()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.master_arm_left_deque = deque()
        self.master_arm_right_deque = deque()

        rospy.init_node("record_episodes_master_cam", anonymous=True)

        rospy.Subscriber(
            args.img_front_topic, Image, self._img_front_cb, queue_size=1000, tcp_nodelay=True
        )
        rospy.Subscriber(
            args.img_left_topic, Image, self._img_left_cb, queue_size=1000, tcp_nodelay=True
        )
        rospy.Subscriber(
            args.img_right_topic, Image, self._img_right_cb, queue_size=1000, tcp_nodelay=True
        )

        rospy.Subscriber(
            args.master_arm_left_topic,
            JointState,
            self._master_left_cb,
            queue_size=1000,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            args.master_arm_right_topic,
            JointState,
            self._master_right_cb,
            queue_size=1000,
            tcp_nodelay=True,
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

    def get_synced_frame(self):
        """取所有队列的最近公共时间戳并拉齐数据。"""
        if (
            len(self.img_front_deque) == 0
            or len(self.img_left_deque) == 0
            or len(self.img_right_deque) == 0
            or len(self.master_arm_left_deque) == 0
            or len(self.master_arm_right_deque) == 0
        ):
            return None

        frame_time = min(
            self.img_front_deque[-1].header.stamp.to_sec(),
            self.img_left_deque[-1].header.stamp.to_sec(),
            self.img_right_deque[-1].header.stamp.to_sec(),
            self.master_arm_left_deque[-1].header.stamp.to_sec(),
            self.master_arm_right_deque[-1].header.stamp.to_sec(),
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

        img_front = cv2.resize(img_front, (640, 480))
        img_left = cv2.resize(img_left, (640, 480))
        img_right = cv2.resize(img_right, (640, 480))

        return img_front, img_left, img_right, master_left, master_right

    def process(self):
        print("等待相机和主臂数据...")
        rate = rospy.Rate(50)
        t0 = time.time()
        while not rospy.is_shutdown():
            if (
                len(self.img_front_deque) > 0
                and len(self.img_left_deque) > 0
                and len(self.img_right_deque) > 0
                and len(self.master_arm_left_deque) > 0
                and len(self.master_arm_right_deque) > 0
            ):
                print("\033[32m所有数据源就绪!\033[0m")
                break
            if time.time() - t0 > 15.0:
                print("\033[31m等待数据超时! 请检查相机和主臂节点是否已启动.\033[0m")
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
                print(f"  缺失话题: {missing}")
                return [], []
            rate.sleep()

        print("等待主臂 CAN 数据稳定...")
        t_can = time.time()
        while not rospy.is_shutdown():
            def _arm_valid(dq):
                if not dq:
                    return False
                pos = list(dq[-1].position)
                return any(abs(v) > 1e-4 for v in pos[:6])

            if _arm_valid(self.master_arm_left_deque) and _arm_valid(self.master_arm_right_deque):
                print("\033[32m主臂 CAN 数据已稳定（非零）!\033[0m")
                break
            if time.time() - t_can > 5.0:
                print("\033[33m[warn] 主臂关节数据仍为零，可能臂在机械零位，继续录制...\033[0m")
                break
            rate.sleep()

        input("\n\033[33m按 Enter 开始录制...\033[0m")
        print(f"开始录制，目标帧数: {self.args.max_timesteps}")

        timesteps = []
        actions = []
        count = 0
        rate = rospy.Rate(self.args.frame_rate)
        print_flag = True

        while count < self.args.max_timesteps + 1 and not rospy.is_shutdown():
            result = self.get_synced_frame()
            if result is None:
                if print_flag:
                    print("sync fail, waiting...")
                    print_flag = False
                rate.sleep()
                continue
            print_flag = True
            count += 1

            img_front, img_left, img_right, master_left, master_right = result

            qpos = np.concatenate(
                (np.array(master_left.position), np.array(master_right.position)), axis=0
            )
            qvel = np.concatenate(
                (np.array(master_left.velocity), np.array(master_right.velocity)), axis=0
            )
            effort = np.concatenate(
                (np.array(master_left.effort), np.array(master_right.effort)), axis=0
            )

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
            action = qpos.copy()

            if count == 1:
                timesteps.append(obs)
                continue

            actions.append(action)
            timesteps.append(obs)
            if count % 50 == 0 or count == self.args.max_timesteps + 1:
                print(f"  已录制: {count - 1}/{self.args.max_timesteps}")
            rate.sleep()

        print(f"\n录制完成! timesteps: {len(timesteps)}, actions: {len(actions)}")
        return timesteps, actions


def get_arguments():
    parser = argparse.ArgumentParser(description="主臂+相机数据采集(无从臂)")
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--task_name", type=str, default="aloha_master_cam")
    parser.add_argument("--episode_idx", type=int, default=0)
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
    parser.add_argument("--frame_rate", type=int, default=30)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    return parser.parse_args()


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    timesteps, actions = ros_operator.process()

    if len(actions) < args.max_timesteps:
        print(
            f"\033[31m\n录制失败，需要 {args.max_timesteps} 帧，只录到 {len(actions)} 帧\033[0m\n"
        )
        raise SystemExit(1)

    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, f"episode_{args.episode_idx}")
    save_data(args, timesteps, actions, dataset_path)


if __name__ == "__main__":
    main()
