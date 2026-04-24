# -- coding: UTF-8
"""
仅主臂数据采集脚本（无从臂、无摄像头）

用法:
    python collect_data_master_only.py --dataset_dir ~/data --task_name my_task --max_timesteps 500 --episode_idx 0

说明:
    - 仅采集双主臂的关节数据，无图像
    - observation 的 qpos/qvel/effort 来自主臂（因为没有从臂，主臂数据同时作为 obs 和 action）
    - action 也来自主臂
    - 输出 HDF5 文件，格式兼容 ACT 训练框架
"""
import os
import time
import numpy as np
import h5py
import argparse
import collections

import rospy
from sensor_msgs.msg import JointState


def save_data(args, timesteps, actions, dataset_path):
    """保存数据为 HDF5 格式"""
    data_size = len(actions)
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
    }

    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts['qpos'])
        data_dict['/observations/qvel'].append(ts['qvel'])
        data_dict['/observations/effort'].append(ts['effort'])
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(np.array([0.0, 0.0]))

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False
        obs = root.create_group('observations')
        # 不创建图像组（无摄像头）
        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = root.create_dataset('action', (data_size, 14))
        _ = root.create_dataset('base_action', (data_size, 2))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. {dataset_path}\033[0m\n')


class RosOperator:
    def __init__(self, args):
        self.args = args
        self.master_arm_left_msg = None
        self.master_arm_right_msg = None

        rospy.init_node('record_episodes_master_only', anonymous=True)
        rospy.Subscriber(args.master_arm_left_topic, JointState,
                         self._master_left_cb, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(args.master_arm_right_topic, JointState,
                         self._master_right_cb, queue_size=1000, tcp_nodelay=True)

    def _master_left_cb(self, msg):
        self.master_arm_left_msg = msg

    def _master_right_cb(self, msg):
        self.master_arm_right_msg = msg

    def wait_for_data(self, timeout=10.0):
        """等待双主臂数据就绪"""
        print("等待主臂数据...")
        t0 = time.time()
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.master_arm_left_msg is not None and self.master_arm_right_msg is not None:
                print("双主臂数据就绪!")
                return True
            if time.time() - t0 > timeout:
                print("\033[31m等待主臂数据超时!\033[0m")
                return False
            rate.sleep()
        return False

    def get_obs_and_action(self):
        """获取当前帧的观测和动作（主臂数据同时作为 obs 和 action）"""
        left = self.master_arm_left_msg
        right = self.master_arm_right_msg
        if left is None or right is None:
            return None

        qpos = np.concatenate((np.array(left.position), np.array(right.position)), axis=0)
        qvel = np.concatenate((np.array(left.velocity), np.array(right.velocity)), axis=0)
        effort = np.concatenate((np.array(left.effort), np.array(right.effort)), axis=0)

        obs = {
            'qpos': qpos,
            'qvel': qvel,
            'effort': effort,
        }
        action = qpos.copy()  # 仅主臂时，action = 主臂的 qpos
        return obs, action

    def process(self):
        """采集数据主循环"""
        if not self.wait_for_data():
            return [], []

        input("\n\033[33m按 Enter 开始录制...\033[0m")
        print(f"开始录制，目标帧数: {self.args.max_timesteps}")

        timesteps = []
        actions = []
        count = 0
        rate = rospy.Rate(self.args.frame_rate)

        while count < self.args.max_timesteps + 1 and not rospy.is_shutdown():
            result = self.get_obs_and_action()
            if result is None:
                rate.sleep()
                continue

            obs, action = result
            count += 1

            if count == 1:
                # 第一帧只保存观测，不保存动作
                timesteps.append(obs)
                continue

            actions.append(action)
            timesteps.append(obs)
            if count % 50 == 0 or count == self.args.max_timesteps + 1:
                print(f"  已录制: {count - 1}/{self.args.max_timesteps}")
            rate.sleep()

        print(f"录制完成! timesteps: {len(timesteps)}, actions: {len(actions)}")
        return timesteps, actions


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--task_name', type=str, default='aloha_master_only')
    parser.add_argument('--episode_idx', type=int, default=0)
    parser.add_argument('--max_timesteps', type=int, default=500)
    parser.add_argument('--master_arm_left_topic', type=str, default='/master/joint_left')
    parser.add_argument('--master_arm_right_topic', type=str, default='/master/joint_right')
    parser.add_argument('--frame_rate', type=int, default=30)
    return parser.parse_args()


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    timesteps, actions = ros_operator.process()

    if len(actions) < args.max_timesteps:
        print(f"\033[31m\n录制失败，需要 {args.max_timesteps} 帧，只录到 {len(actions)} 帧\033[0m\n")
        exit(1)

    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, f"episode_{args.episode_idx}")
    save_data(args, timesteps, actions, dataset_path)


if __name__ == '__main__':
    main()
