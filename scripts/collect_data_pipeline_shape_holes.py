# -- coding: UTF-8
"""
Shape-to-hole data-collection pipeline with Tkinter UI + pedal control.
Extended with deterministic per-episode layout and subtask sequencing.

Pedal behavior
--------------
  Press 1     : Start recording — lamp green, subtask[0] activated
  Press 2..N  : Advance to next subtask (still recording, still green)
  Press N+1   : No more subtasks → stop & save episode

Each timestep records the currently active subtask string in
/observations/subtask inside the HDF5 file.

Task layout
-----------
Each episode contains four top-row solids and three bottom-row holes:
  solids: hexagonal prism, cuboid, cube, triangular prism
  holes : three of the four matching holes

The object layout covers all 24 top-row permutations and the hole layout covers
all 24 ordered choices of three holes.  Their pairings cover the full 24 x 24
Cartesian product, so the exact same layout pair repeats every 576 episodes.
Within each 24-episode block, both top and hole layouts still each appear once.

Per task layout on disk
-----------------------
<dataset_dir>/<task_name>/
    pipeline_meta.json
    <instruction_slug>/
        episode_0.hdf5
        ...
"""
import sys
import os
import itertools
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------------------
# Shape-hole layout plan.
# ------------------------------------------------------------------

SHAPES = [
    {"key": "hexagonal_prism", "name": "hexagonal prism"},
    {"key": "cuboid", "name": "cuboid"},
    {"key": "cube", "name": "cube"},
    {"key": "triangular_prism", "name": "triangular prism"},
]

TOP_POSITIONS = ["left", "middle-left", "middle-right", "right"]
HOLE_POSITIONS = ["left", "middle", "right"]

# Shape indices by top object position.  Covers all 4! = 24 top-row layouts.
# Across one cycle, each shape appears 6 times in each top-row position.
TOP_SHAPE_CYCLE = list(itertools.permutations(range(len(SHAPES)), len(TOP_POSITIONS)))

# Shape indices by bottom hole position.  Covers all P(4, 3) = 24 hole layouts.
# Across one cycle, each shape appears 6 times in each hole position and is
# omitted 6 times.
HOLE_SHAPE_CYCLE = list(itertools.permutations(range(len(SHAPES)), len(HOLE_POSITIONS)))

DEFAULT_INSTRUCTIONS = ["put the shapes into the matching holes"]
DEFAULT_LAYOUT_PAIRING_SEED = 20260516

SHAPE_COLORS = {
    "hexagonal_prism": "#ebb30c",
    "cuboid": "#2f9ce0",
    "cube": "#ff0000",
    "triangular_prism": "#2f9ce0",
}

SHAPE_SHORT_LABELS = {
    "hexagonal_prism": "hex prism",
    "cuboid": "cuboid",
    "cube": "cube",
    "triangular_prism": "tri prism",
}

JOINT_TOPIC_SPECS = [
    ("arm_left", "master left", "master_arm_left_topic", "master_arm_left_deque"),
    ("arm_right", "master right", "master_arm_right_topic", "master_arm_right_deque"),
    ("puppet_left", "puppet left", "puppet_arm_left_topic", "puppet_arm_left_deque"),
    ("puppet_right", "puppet right", "puppet_arm_right_topic", "puppet_arm_right_deque"),
]



import argparse
import glob
import json
import re
import threading
import time
import queue
from collections import deque

import cv2
import h5py
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState

from dataarm_notifier import RecordingController
from go_home import go_home_and_wait


# ------------------------------------------------------------------
# Filesystem helpers
# ------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify(text: str) -> str:
    text = text.strip().lower().replace(" ", "_")
    text = _SLUG_RE.sub("_", text)
    text = text.strip("._-")
    return text or "instruction"


def encode_jpeg(image, quality):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, encoded = cv2.imencode(".jpg", image, encode_params)
    if not ok:
        raise RuntimeError("failed to encode image to jpeg")
    return np.frombuffer(encoded.tobytes(), dtype=np.uint8)


def list_episode_indices(folder: str):
    if not os.path.isdir(folder):
        return []
    indices = []
    for path in glob.glob(os.path.join(folder, "episode_*.hdf5")):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            indices.append(int(name.split("_")[-1]))
        except ValueError:
            continue
    return sorted(indices)


def next_episode_idx(folder: str):
    indices = list_episode_indices(folder)
    return (indices[-1] + 1) if indices else 0


def episode_paths(folder: str):
    paths = []
    for idx in list_episode_indices(folder):
        paths.append((idx, os.path.join(folder, f"episode_{idx}.hdf5")))
    return paths


def build_layout_pair_cycle(seed: int):
    """Return 576 deterministic balanced (top_idx, hole_idx) pairs."""
    rng = random.Random(seed)
    n = len(TOP_SHAPE_CYCLE)
    if n != len(HOLE_SHAPE_CYCLE):
        raise ValueError("top and hole cycles must have the same length")

    top_order = list(range(n))
    hole_order = list(range(n))
    rng.shuffle(top_order)
    rng.shuffle(hole_order)
    block_offsets = list(range(n))
    rng.shuffle(block_offsets)

    pairs = []
    for offset in block_offsets:
        for pos, top_idx in enumerate(top_order):
            hole_idx = hole_order[(pos + offset) % n]
            pairs.append((top_idx, hole_idx))
    return pairs


def _shape_name(shape_idx: int) -> str:
    return SHAPES[shape_idx]["name"]


def build_shape_hole_episode_plan(
    episode_idx: int,
    layout_pair_cycle=None,
    pairing_seed=DEFAULT_LAYOUT_PAIRING_SEED,
    pair_cycle_idx_override=None,
):
    """Return deterministic balanced layout and subtasks for one episode."""
    if layout_pair_cycle is None:
        layout_pair_cycle = build_layout_pair_cycle(pairing_seed)
    if pair_cycle_idx_override is None:
        pair_cycle_idx = episode_idx % len(layout_pair_cycle)
    else:
        pair_cycle_idx = int(pair_cycle_idx_override) % len(layout_pair_cycle)
    top_cycle_idx, hole_cycle_idx = layout_pair_cycle[pair_cycle_idx]
    top_shape_indices = TOP_SHAPE_CYCLE[top_cycle_idx]
    hole_shape_indices = HOLE_SHAPE_CYCLE[hole_cycle_idx]
    omitted_shape_idx = ({0, 1, 2, 3} - set(hole_shape_indices)).pop()

    top = [
        {
            "position": position,
            "shape": _shape_name(shape_idx),
            "shape_key": SHAPES[shape_idx]["key"],
        }
        for position, shape_idx in zip(TOP_POSITIONS, top_shape_indices)
    ]
    holes = [
        {
            "position": position,
            "shape": _shape_name(shape_idx),
            "shape_key": SHAPES[shape_idx]["key"],
        }
        for position, shape_idx in zip(HOLE_POSITIONS, hole_shape_indices)
    ]

    subtasks = [
        f"put the {_shape_name(shape_idx)} into the {position} hole"
        for position, shape_idx in zip(HOLE_POSITIONS, hole_shape_indices)
    ]

    return {
        "episode_idx": int(episode_idx),
        "pair_cycle_idx": int(pair_cycle_idx),
        "pair_cycle_length": len(layout_pair_cycle),
        "top_cycle_idx": int(top_cycle_idx),
        "hole_cycle_idx": int(hole_cycle_idx),
        "cycle_length": len(TOP_SHAPE_CYCLE),
        "layout_pairing_seed": int(pairing_seed),
        "top": top,
        "holes": holes,
        "omitted_hole_shape": _shape_name(omitted_shape_idx),
        "omitted_hole_shape_key": SHAPES[omitted_shape_idx]["key"],
        "subtasks": subtasks,
    }


def format_episode_plan(plan) -> str:
    top = " | ".join(
        f"{item['position']}: {item['shape']}" for item in plan["top"]
    )
    holes = " | ".join(
        f"{item['position']}: {item['shape']} hole" for item in plan["holes"]
    )
    subtasks = " -> ".join(plan["subtasks"])
    return (
        f"Episode {plan['episode_idx']}  "
        f"(top {plan['top_cycle_idx'] + 1}/{plan['cycle_length']} | "
        f"hole {plan['hole_cycle_idx'] + 1}/{plan['cycle_length']})\n"
        f"Objects(top): {top}\n"
        f"Holes(bottom): {holes}\n"
        f"No hole for: {plan['omitted_hole_shape']}\n"
        f"Subtasks: {subtasks}"
    )


def load_episode_layout_plan(path: str):
    try:
        with h5py.File(path, "r") as root:
            raw = root.attrs.get("shape_hole_layout")
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))
    except Exception as exc:
        print(f"[WARN] failed to load layout from {path}: {exc}")
        return None


def layout_pair_from_plan(plan):
    if not plan:
        return None
    if "top_cycle_idx" in plan and "hole_cycle_idx" in plan:
        return int(plan["top_cycle_idx"]), int(plan["hole_cycle_idx"])
    return None


def used_layout_pairs(folder: str):
    used = {}
    for episode_idx, path in episode_paths(folder):
        pair = layout_pair_from_plan(load_episode_layout_plan(path))
        if pair is None:
            continue
        used.setdefault(pair, []).append(episode_idx)
    return used


def find_first_unused_pair_cycle_idx(layout_pair_cycle, used_pairs):
    used_set = set(used_pairs)
    for pair_cycle_idx, pair in enumerate(layout_pair_cycle):
        if tuple(pair) not in used_set:
            return pair_cycle_idx
    return None


# ------------------------------------------------------------------
# HDF5 saving
# ------------------------------------------------------------------

def save_episode(args, timesteps, actions, subtask_sequence, dataset_path, instruction_text, layout_plan=None):
    """Save one episode to HDF5.  subtask_sequence is a list[str] aligned with actions."""
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

    subtask_list = []
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        subtask_list.append(subtask_sequence.pop(0) if subtask_sequence else "")
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
        root.attrs["instruction"] = instruction_text
        root.attrs["frame_rate"] = int(args.frame_rate)
        if layout_plan is not None:
            root.attrs["shape_hole_layout"] = json.dumps(layout_plan, ensure_ascii=False)

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

        # Per-timestep subtask string  (shape: data_size, dtype: variable-length UTF-8)
        obs.create_dataset(
            "subtask",
            data=np.array(subtask_list, dtype=object),
            dtype=h5py.string_dtype(),
        )

    print(f"\033[32mSaved {dataset_path}.hdf5 ({data_size} frames) in {time.time() - t0:.1f}s\033[0m")


# ------------------------------------------------------------------
# Pipeline metadata persistence
# ------------------------------------------------------------------

class PipelineMeta:
    """Tracks ordered instructions and their slug mapping on disk."""

    def __init__(self, task_dir: str):
        self.task_dir = task_dir
        os.makedirs(task_dir, exist_ok=True)
        self.path = os.path.join(task_dir, "pipeline_meta.json")
        self._lock = threading.Lock()
        self.instructions = []
        self.slugs = {}
        self.load()

    def load(self):
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            self.instructions = list(payload.get("instructions", []))
            self.slugs = dict(payload.get("slugs", {}))
        except Exception as exc:
            print(f"[WARN] failed to load pipeline meta: {exc}")

    def save(self):
        payload = {"instructions": self.instructions, "slugs": self.slugs}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def ensure_slug(self, instruction: str) -> str:
        with self._lock:
            if instruction in self.slugs:
                return self.slugs[instruction]
            base = slugify(instruction)
            slug = base
            existing = set(self.slugs.values())
            i = 1
            while slug in existing:
                i += 1
                slug = f"{base}_{i}"
            self.slugs[instruction] = slug
            if instruction not in self.instructions:
                self.instructions.append(instruction)
            self.save()
            return slug

    def add_instruction(self, instruction: str):
        instruction = instruction.strip()
        if not instruction:
            return False
        with self._lock:
            if instruction in self.instructions:
                return False
            self.instructions.append(instruction)
        self.ensure_slug(instruction)
        return True

    def remove_instruction(self, instruction: str):
        with self._lock:
            if instruction in self.instructions:
                self.instructions.remove(instruction)
            self.slugs.pop(instruction, None)
            self.save()

    def folder_for(self, instruction: str) -> str:
        slug = self.ensure_slug(instruction)
        folder = os.path.join(self.task_dir, slug)
        os.makedirs(folder, exist_ok=True)
        return folder


# ------------------------------------------------------------------
# Core ROS data pump
# ------------------------------------------------------------------

class RosDataPump:
    """Subscribes to ROS topics; provides synchronized frames on demand."""

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

        _now = time.time()
        self._last_recv = {
            "front":        _now,
            "left":         _now,
            "right":        _now,
            "arm_left":     _now,
            "arm_right":    _now,
            "puppet_left":  _now,
            "puppet_right": _now,
        }

        rospy.Subscriber(args.img_front_topic, Image, self._img_front_cb, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(args.img_left_topic, Image, self._img_left_cb, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(args.img_right_topic, Image, self._img_right_cb, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(args.master_arm_left_topic, JointState, self._master_left_cb,
                         queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(args.master_arm_right_topic, JointState, self._master_right_cb,
                         queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(args.puppet_arm_left_topic, JointState, self._puppet_left_cb,
                         queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(args.puppet_arm_right_topic, JointState, self._puppet_right_cb,
                         queue_size=1000, tcp_nodelay=True)

    def _img_front_cb(self, msg):
        self._last_recv["front"] = time.time()
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def _img_left_cb(self, msg):
        self._last_recv["left"] = time.time()
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def _img_right_cb(self, msg):
        self._last_recv["right"] = time.time()
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def _master_left_cb(self, msg):
        self._last_recv["arm_left"] = time.time()
        if len(self.master_arm_left_deque) >= 2000:
            self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)

    def _master_right_cb(self, msg):
        self._last_recv["arm_right"] = time.time()
        if len(self.master_arm_right_deque) >= 2000:
            self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)

    def _puppet_left_cb(self, msg):
        self._last_recv["puppet_left"] = time.time()
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def _puppet_right_cb(self, msg):
        self._last_recv["puppet_right"] = time.time()
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    @staticmethod
    def _arm_valid(dq):
        if not dq:
            return False
        pos = list(dq[-1].position)
        return any(abs(v) > 1e-4 for v in pos[:6])

    def lost_topics(self, timeout: float = 1.5):
        now = time.time()
        key_to_topic = {
            "front":        self.args.img_front_topic,
            "left":         self.args.img_left_topic,
            "right":        self.args.img_right_topic,
            "arm_left":     self.args.master_arm_left_topic,
            "arm_right":    self.args.master_arm_right_topic,
            "puppet_left":  self.args.puppet_arm_left_topic,
            "puppet_right": self.args.puppet_arm_right_topic,
        }
        return [key_to_topic[k] for k, t in self._last_recv.items() if now - t > timeout]

    def joint_status(self, timeout: float = 1.5, expected_joints: int = 7):
        now = time.time()
        rows = []
        for key, label, topic_attr, deque_attr in JOINT_TOPIC_SPECS:
            topic = getattr(self.args, topic_attr)
            dq = getattr(self, deque_attr)
            age = now - self._last_recv.get(key, 0.0)
            if not dq:
                rows.append({
                    "key": key,
                    "label": label,
                    "topic": topic,
                    "ok": False,
                    "message": "no JointState received",
                    "age": None,
                })
                continue

            msg = dq[-1]
            positions = list(msg.position)
            if age > timeout:
                rows.append({
                    "key": key,
                    "label": label,
                    "topic": topic,
                    "ok": False,
                    "message": f"stale {age:.1f}s",
                    "age": age,
                })
                continue
            if len(positions) < expected_joints:
                rows.append({
                    "key": key,
                    "label": label,
                    "topic": topic,
                    "ok": False,
                    "message": f"only {len(positions)}/{expected_joints} joint angles",
                    "age": age,
                })
                continue
            if not np.all(np.isfinite(np.array(positions[:expected_joints], dtype=np.float64))):
                rows.append({
                    "key": key,
                    "label": label,
                    "topic": topic,
                    "ok": False,
                    "message": "joint angle is NaN/Inf",
                    "age": age,
                })
                continue
            rows.append({
                "key": key,
                "label": label,
                "topic": topic,
                "ok": True,
                "message": f"ok {len(positions)} joints",
                "age": age,
            })
        return rows

    def all_streams_ready(self):
        return (
            len(self.img_front_deque) > 0
            and len(self.img_left_deque) > 0
            and len(self.img_right_deque) > 0
            and len(self.master_arm_left_deque) > 0
            and len(self.master_arm_right_deque) > 0
            and len(self.puppet_arm_left_deque) > 0
            and len(self.puppet_arm_right_deque) > 0
        )

    def wait_until_ready(self, timeout=15.0):
        rate = rospy.Rate(50)
        t0 = time.time()
        while not rospy.is_shutdown():
            if self.all_streams_ready():
                break
            if time.time() - t0 > timeout:
                missing = []
                if len(self.img_front_deque) == 0: missing.append(self.args.img_front_topic)
                if len(self.img_left_deque) == 0: missing.append(self.args.img_left_topic)
                if len(self.img_right_deque) == 0: missing.append(self.args.img_right_topic)
                if len(self.master_arm_left_deque) == 0: missing.append(self.args.master_arm_left_topic)
                if len(self.master_arm_right_deque) == 0: missing.append(self.args.master_arm_right_topic)
                if len(self.puppet_arm_left_deque) == 0: missing.append(self.args.puppet_arm_left_topic)
                if len(self.puppet_arm_right_deque) == 0: missing.append(self.args.puppet_arm_right_topic)
                raise RuntimeError(f"Timed out waiting for topics: {missing}")
            rate.sleep()

        t_can = time.time()
        while not rospy.is_shutdown():
            if (self._arm_valid(self.master_arm_left_deque)
                    and self._arm_valid(self.master_arm_right_deque)
                    and self._arm_valid(self.puppet_arm_left_deque)
                    and self._arm_valid(self.puppet_arm_right_deque)):
                return
            if time.time() - t_can > 5.0:
                print("[WARN] Joint data still near zero; continue anyway.")
                return
            rate.sleep()

    def get_synced_frame(self):
        if not self.all_streams_ready():
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


# ------------------------------------------------------------------
# Collector worker (background thread)
# ------------------------------------------------------------------

class CollectorWorker(threading.Thread):
    """Background thread that pulls frames and appends them while recording."""

    def __init__(self, args, pump, ui_queue):
        super().__init__(daemon=True)
        self.args = args
        self.pump = pump
        self.ui_queue = ui_queue

        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._is_recording = False
        self._stop_requested = False
        self._episode = None
        self._episode_meta = None
        self._stall_since = None
        self._last_front = None
        self._discard_requested = False

        # Subtask state (protected by _lock)
        self._subtasks = []    # list[str] for current episode
        self._subtask_idx = 0  # index into _subtasks

    # ---- subtask API (thread-safe) ----

    def has_more_subtasks(self) -> bool:
        with self._lock:
            return self._subtask_idx < len(self._subtasks) - 1

    def advance_subtask(self):
        """Advance to next subtask.  Returns (new_idx, total, new_text, has_more)."""
        with self._lock:
            if self._subtask_idx < len(self._subtasks) - 1:
                self._subtask_idx += 1
            idx = self._subtask_idx
            total = len(self._subtasks)
            text = self._subtasks[idx] if self._subtasks else ""
            has_more = idx < total - 1
            return idx, total, text, has_more

    def subtask_info(self):
        """Returns (current_idx, total, current_text)."""
        with self._lock:
            if not self._subtasks:
                return 0, 0, ""
            return self._subtask_idx, len(self._subtasks), self._subtasks[self._subtask_idx]

    # ---- control API (thread-safe, called from UI / pedal) ----

    def request_stop_thread(self):
        self._stop_evt.set()

    def request_start(self, instruction, folder, idx, subtasks=None, layout_plan=None):
        with self._lock:
            if self._is_recording:
                return False
            self._subtasks = list(subtasks) if subtasks else [""]
            self._subtask_idx = 0
            self._episode = {"timesteps": [], "actions": [], "subtasks": [], "count": 0}
            self._episode_meta = {
                "folder": folder,
                "idx": idx,
                "instruction": instruction,
                "layout_plan": layout_plan,
            }
            self._is_recording = True
            self._stop_requested = False
            self._discard_requested = False
        self.ui_queue.put(("status", f"Recording episode_{idx} :: {instruction}"))
        self.ui_queue.put(("recording", True))
        return True

    def request_stop(self):
        with self._lock:
            if not self._is_recording:
                return False
            self._stop_requested = True
        self.ui_queue.put(("status", "Stop requested. Saving..."))
        return True

    def request_discard_current(self):
        with self._lock:
            if not self._is_recording:
                return False
            self._discard_requested = True
            self._stop_requested = True
        self.ui_queue.put(("status", "Stop requested. Deleting current episode..."))
        return True

    def is_recording(self):
        with self._lock:
            return self._is_recording

    # ---- thread main loop ----

    def run(self):
        rate = rospy.Rate(self.args.frame_rate)
        while not self._stop_evt.is_set() and not rospy.is_shutdown():
            with self._lock:
                recording = self._is_recording
                stop_req = self._stop_requested
                discard_req = self._discard_requested
            if not recording:
                rate.sleep()
                continue
            if stop_req:
                if discard_req:
                    self._discard_episode("discarded by user")
                else:
                    self._finalize_episode()
                rate.sleep()
                continue

            result = self.pump.get_synced_frame()
            if result is None:
                if self._stall_since is None:
                    self._stall_since = time.time()
                elif time.time() - self._stall_since > 1.5:
                    lost = self.pump.lost_topics(timeout=1.5)
                    if lost:
                        reason = "Camera/topic lost: " + ", ".join(lost)
                        self.ui_queue.put(("status", f"[ERROR] {reason}"))
                        self._abort_episode(reason)
                        self._stall_since = None
                rate.sleep()
                continue

            self._stall_since = None
            self._append_step(result)
            rate.sleep()

    def _append_step(self, result):
        (img_front, img_left, img_right,
         master_left, master_right,
         puppet_left, puppet_right) = result

        qpos = np.concatenate((np.array(puppet_left.position), np.array(puppet_right.position)), axis=0)
        qvel = np.concatenate((np.array(puppet_left.velocity), np.array(puppet_right.velocity)), axis=0)
        effort = np.concatenate((np.array(puppet_left.effort), np.array(puppet_right.effort)), axis=0)
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

        with self._lock:
            ep = self._episode
            if ep is None:
                return
            ep["count"] += 1
            count = ep["count"]
            if count == 1:
                ep["timesteps"].append(obs)
                self.ui_queue.put(("episode_first_frame", img_front.copy()))
                return
            ep["actions"].append(action)
            ep["timesteps"].append(obs)
            ep["subtasks"].append(self._subtasks[self._subtask_idx])
            self._last_front = img_front
            if count % 50 == 0:
                self.ui_queue.put(("frames", count - 1))
            if len(ep["actions"]) >= self.args.max_timesteps:
                self.ui_queue.put(("status", "Reached max_timesteps. Auto-stopping."))
                self._stop_requested = True

    def _abort_episode(self, reason: str = "aborted"):
        with self._lock:
            self._episode = None
            self._episode_meta = None
            self._is_recording = False
            self._stop_requested = False
            self._discard_requested = False
        self.ui_queue.put(("episode_aborted", reason))
        self.ui_queue.put(("saving", False))
        self.ui_queue.put(("recording", False))

    def _discard_episode(self, reason: str = "discarded"):
        with self._lock:
            meta = self._episode_meta
            self._episode = None
            self._episode_meta = None
            self._is_recording = False
            self._stop_requested = False
            self._discard_requested = False
        idx_text = f"episode_{meta['idx']}" if meta else "current episode"
        self.ui_queue.put(("status", f"Deleted current recording: {idx_text} ({reason})"))
        self.ui_queue.put(("episode_discarded", meta))
        self.ui_queue.put(("saving", False))
        self.ui_queue.put(("recording", False))

    def _finalize_episode(self):
        with self._lock:
            episode = self._episode
            meta = self._episode_meta
            self._episode = None
            self._episode_meta = None
            self._is_recording = False
            self._stop_requested = False
            self._discard_requested = False

        if episode is None or meta is None:
            self.ui_queue.put(("recording", False))
            return

        actions = episode["actions"]
        timesteps = episode["timesteps"]
        subtasks = episode.get("subtasks", [])

        if len(actions) == 0:
            self.ui_queue.put(("status", "Empty episode. Discarded."))
            self.ui_queue.put(("recording", False))
            return

        if self._last_front is not None:
            self.ui_queue.put(("episode_last_frame", self._last_front.copy()))
        self.ui_queue.put(("saving", True))
        save_ok = False
        try:
            dataset_path = os.path.join(meta["folder"], f"episode_{meta['idx']}")
            save_episode(
                self.args,
                timesteps,
                actions,
                subtasks,
                dataset_path,
                meta["instruction"],
                meta.get("layout_plan"),
            )
            self.ui_queue.put(("status",
                f"Saved episode_{meta['idx']} ({len(actions)} frames) :: {meta['instruction']}"))
            save_ok = True
        except Exception as exc:
            self.ui_queue.put(("status", f"[ERROR] save failed: {exc}"))
        self.ui_queue.put(("saving", False))

        if save_ok and getattr(self.args, "auto_home", True):
            self.ui_queue.put(("homing", True))
            try:
                home_ok = go_home_and_wait(
                    pos_threshold=self.args.home_pos_threshold,
                    vel_threshold=self.args.home_vel_threshold,
                    stable_seconds=self.args.home_stable_seconds,
                    timeout=self.args.home_timeout,
                    log=lambda m: self.ui_queue.put(("status", m)),
                )
            except Exception as exc:
                self.ui_queue.put(("status", f"[ERROR] homing exception: {exc}"))
                home_ok = False
            self.ui_queue.put(("homing", False))
            self.ui_queue.put(("home_result", home_ok))

        if save_ok:
            self.ui_queue.put(("episode_saved", meta))
        self.ui_queue.put(("recording", False))


# ------------------------------------------------------------------
# Tk UI
# ------------------------------------------------------------------

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

try:
    from PIL import Image as PILImage
    from PIL import ImageTk
    _PIL_AVAILABLE = True
except Exception as _pil_exc:
    _PIL_AVAILABLE = False
    PILImage = None
    ImageTk = None
    print(f"[WARN] Pillow not available, live preview disabled: {_pil_exc}")


class CollectorApp:
    def __init__(self, args):
        self.args = args
        self.task_dir = os.path.join(args.dataset_dir, args.task_name)
        os.makedirs(self.task_dir, exist_ok=True)

        self.meta = PipelineMeta(self.task_dir)
        if not self.meta.instructions and args.instructions:
            for ins in args.instructions:
                self.meta.add_instruction(ins)
        if not self.meta.instructions:
            for ins in DEFAULT_INSTRUCTIONS:
                self.meta.add_instruction(ins)
        self.layout_pair_cycle = build_layout_pair_cycle(args.layout_pairing_seed)

        self.ui_queue = queue.Queue()

        rospy.init_node("collect_data_pipeline", anonymous=True)
        self.pump = RosDataPump(args)

        self.controller = RecordingController(
            port=args.lamp_port,
            pedal_device=args.pedal_device,
            trigger_key=args.trigger_key,
        )
        self.controller.on_recording_start(self._pedal_start)
        self.controller.on_recording_stop(self._pedal_stop)

        self.worker = CollectorWorker(args, self.pump, self.ui_queue)
        self._pending_recollect = None
        self._ros_ready = False

        self.root = tk.Tk()
        self.root.title("Data Collection Pipeline - Shape Hole Mode")
        self.root.geometry("1440x900")
        self._build_ui()
        self.root.focus_set()

        self.is_recording_ui = False

        self.root.after(100, self._async_init)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_queue)
        self.root.after(500, self._poll_joint_status)

    # ---- UI construction ----

    def _keep_shortcut_focus(self, widget):
        try:
            widget.configure(takefocus=0)
        except tk.TclError:
            pass
        widget.bind("<ButtonRelease-1>", lambda _e: self.root.after_idle(self.root.focus_set), add="+")
        return widget

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, **pad)
        ttk.Label(top, text="Task:").pack(side=tk.LEFT)
        ttk.Label(top, text=self.args.task_name, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(top, text="Dir:").pack(side=tk.LEFT)
        ttk.Label(top, text=self.task_dir).pack(side=tk.LEFT)

        joint_frame = ttk.LabelFrame(self.root, text="Joint Angle Topic Monitor")
        joint_frame.pack(fill=tk.X, **pad)
        self.joint_status_var = tk.StringVar(value="Joint topics: checking...")
        self.joint_status_label = ttk.Label(
            joint_frame,
            textvariable=self.joint_status_var,
            font=("TkDefaultFont", 11, "bold"),
            foreground="#666666",
            anchor="w",
        )
        self.joint_status_label.pack(fill=tk.X, padx=8, pady=5)
        self._joint_alarm_active = False

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, **pad)

        left = ttk.LabelFrame(body, text="Instructions")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.instr_list = self._keep_shortcut_focus(
            tk.Listbox(left, exportselection=False, takefocus=0)
        )
        self.instr_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.instr_list.bind("<<ListboxSelect>>", lambda _e: self._on_instruction_selected())

        instr_btns = ttk.Frame(left)
        instr_btns.pack(fill=tk.X, padx=4, pady=4)
        self._keep_shortcut_focus(
            ttk.Button(instr_btns, text="Add", command=self._add_instruction, takefocus=0)
        ).pack(side=tk.LEFT)
        self._keep_shortcut_focus(
            ttk.Button(instr_btns, text="Remove", command=self._remove_instruction, takefocus=0)
        ).pack(side=tk.LEFT, padx=4)

        middle = ttk.LabelFrame(body, text="Episodes for selected instruction")
        middle.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.episode_list = self._keep_shortcut_focus(
            tk.Listbox(middle, exportselection=False, takefocus=0)
        )
        self.episode_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        ep_btns = ttk.Frame(middle)
        ep_btns.pack(fill=tk.X, padx=4, pady=4)
        self._keep_shortcut_focus(
            ttk.Button(ep_btns, text="Delete selected episode", command=self._delete_episode, takefocus=0)
        ).pack(side=tk.LEFT)
        self._keep_shortcut_focus(
            ttk.Button(ep_btns, text="Refresh", command=self._refresh_episodes, takefocus=0)
        ).pack(side=tk.LEFT, padx=4)

        right = ttk.LabelFrame(body, text="Live cameras  /  Episode frames")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        live = ttk.LabelFrame(right, text="Live (front | left | right)")
        live.pack(fill=tk.X, padx=4, pady=4)
        self.preview_label = ttk.Label(live, text="(waiting for camera frames...)", anchor="center")
        self.preview_label.pack(fill=tk.X, padx=4, pady=4)
        self._preview_photo = None

        thumbs = ttk.Frame(right)
        thumbs.pack(fill=tk.X, padx=4, pady=4)
        first_box = ttk.LabelFrame(thumbs, text="Episode start (frame 0)")
        first_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.first_label = ttk.Label(first_box, text="(no episode yet)", anchor="center")
        self.first_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._first_photo = None
        last_box = ttk.LabelFrame(thumbs, text="Episode end (final frame)")
        last_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.last_label = ttk.Label(last_box, text="(no episode yet)", anchor="center")
        self.last_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._last_photo = None

        layout_frame = ttk.LabelFrame(right, text="Next Episode Layout")
        layout_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.layout_text_var = tk.StringVar(value="")
        self.layout_canvas = self._keep_shortcut_focus(tk.Canvas(
            layout_frame,
            height=360,
            bg="#f7f7f4",
            highlightthickness=0,
            takefocus=0,
        ))
        self.layout_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.layout_canvas.bind("<Configure>", lambda _e: self._draw_layout_canvas())
        self._current_layout_plan = None

        # Control buttons
        ctl = ttk.LabelFrame(self.root, text="Control")
        ctl.pack(fill=tk.X, **pad)
        self.btn_start = self._keep_shortcut_focus(
            ttk.Button(ctl, text="Start (Pedal)", command=self._on_start, takefocus=0)
        )
        self.btn_start.pack(side=tk.LEFT, padx=6, pady=6)
        self.btn_stop = self._keep_shortcut_focus(
            ttk.Button(ctl, text="Stop & Save (Pedal)", command=self._on_stop, state=tk.DISABLED, takefocus=0)
        )
        self.btn_stop.pack(side=tk.LEFT, padx=6, pady=6)
        self.btn_next = self._keep_shortcut_focus(
            ttk.Button(
                ctl,
                text="Stop & Delete Current",
                command=self._on_delete_current,
                state=tk.DISABLED,
                takefocus=0,
            )
        )
        self.btn_next.pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Separator(ctl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.btn_home = self._keep_shortcut_focus(
            ttk.Button(ctl, text="Go Home (主从臂归零)", command=self._on_go_home, takefocus=0)
        )
        self.btn_home.pack(side=tk.LEFT, padx=6, pady=6)

        # Subtask display - prominent, shows current active subtask
        subtask_frame = ttk.LabelFrame(self.root, text="Current Subtask  (pedal advances)")
        subtask_frame.pack(fill=tk.X, **pad)
        self.subtask_progress_var = tk.StringVar(value="")
        self.subtask_text_var = tk.StringVar(value="(no subtasks configured)")
        ttk.Label(
            subtask_frame,
            textvariable=self.subtask_progress_var,
            font=("TkDefaultFont", 10),
            foreground="#888888",
            width=8,
            anchor="e",
        ).pack(side=tk.LEFT, padx=(8, 2), pady=6)
        ttk.Label(
            subtask_frame,
            textvariable=self.subtask_text_var,
            font=("TkDefaultFont", 13, "bold"),
            foreground="#1a6e1a",
        ).pack(side=tk.LEFT, padx=6, pady=6)

        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        status = ttk.Frame(self.root)
        status.pack(fill=tk.X, **pad)
        self.lamp_canvas = self._keep_shortcut_focus(
            tk.Canvas(status, width=18, height=18, highlightthickness=0, takefocus=0)
        )
        self.lamp_canvas.pack(side=tk.LEFT, padx=4)
        self.lamp_id = self.lamp_canvas.create_oval(2, 2, 16, 16, fill="gray")
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, padx=6)

        self._refresh_instructions()
        self._refresh_next_episode_plan()

    # ---- subtask display ----

    def _update_subtask_display(self, idx: int, total: int, text: str):
        if total <= 0:
            self.subtask_progress_var.set("")
            self.subtask_text_var.set("(no subtask)")
            return
        self.subtask_progress_var.set(f"{idx + 1} / {total}")
        self.subtask_text_var.set(text if text else "(empty)")

    def _draw_layout_canvas(self):
        canvas = getattr(self, "layout_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        plan = getattr(self, "_current_layout_plan", None)
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 1:
            width = 640
        if height <= 1:
            height = 360

        if plan is None:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Select or add an instruction to show the next layout.",
                fill="#555555",
                font=("TkDefaultFont", 14, "bold"),
            )
            return

        canvas.create_text(
            18,
            18,
            text=(
                f"Episode {plan['episode_idx']}   "
                f"Pair {plan['pair_cycle_idx'] + 1}/{plan['pair_cycle_length']}   "
                f"Top {plan['top_cycle_idx'] + 1}/{plan['cycle_length']}   "
                f"Hole {plan['hole_cycle_idx'] + 1}/{plan['cycle_length']}"
            ),
            anchor="w",
            fill="#222222",
            font=("TkDefaultFont", 14, "bold"),
        )
        canvas.create_text(
            18,
            45,
            text=f"No hole for: {plan['omitted_hole_shape']}",
            anchor="w",
            fill="#8a3b00",
            font=("TkDefaultFont", 12, "bold"),
        )

        top_y = height * 0.34
        hole_y = height * 0.68
        self._draw_layout_row_label(canvas, 18, top_y, "OBJECTS")
        self._draw_layout_row_label(canvas, 18, hole_y, "HOLES")

        top_xs = self._layout_positions(width, len(plan["top"]))
        hole_xs = self._layout_positions(width, len(plan["holes"]))
        for x, item in zip(top_xs, plan["top"]):
            self._draw_shape_icon(
                canvas,
                x,
                top_y,
                item["shape_key"],
                filled=True,
                scale=1.0,
            )
            self._draw_position_label(canvas, x, top_y + 72, item["position"], item["shape_key"])

        for x, item in zip(hole_xs, plan["holes"]):
            self._draw_shape_icon(
                canvas,
                x,
                hole_y,
                item["shape_key"],
                filled=False,
                scale=1.08,
            )
            self._draw_position_label(canvas, x, hole_y + 72, item["position"], item["shape_key"])

    def _layout_positions(self, width, count):
        left_margin = 145
        right_margin = 45
        usable = max(width - left_margin - right_margin, count * 120)
        if count == 1:
            return [left_margin + usable / 2]
        return [
            left_margin + i * (usable / (count - 1))
            for i in range(count)
        ]

    def _draw_layout_row_label(self, canvas, x, y, text):
        canvas.create_text(
            x,
            y,
            text=text,
            anchor="w",
            fill="#555555",
            font=("TkDefaultFont", 11, "bold"),
        )

    def _draw_position_label(self, canvas, x, y, position, shape_key):
        label = f"{position}\n{SHAPE_SHORT_LABELS.get(shape_key, shape_key)}"
        canvas.create_text(
            x,
            y,
            text=label,
            anchor="n",
            justify=tk.CENTER,
            fill="#222222",
            font=("TkDefaultFont", 11, "bold"),
        )

    def _draw_shape_icon(self, canvas, x, y, shape_key, filled, scale=1.0):
        color = SHAPE_COLORS.get(shape_key, "#444444")
        fill = color if filled else "#ffffff"
        outline = "#222222" if filled else color
        stroke = 2 if filled else 6
        tag = "shape_layout"

        if shape_key == "hexagonal_prism":
            r = 42 * scale
            pts = []
            for i in range(6):
                angle = i * np.pi / 3
                pts.extend([x + r * np.cos(angle), y + r * np.sin(angle)])
            canvas.create_polygon(pts, fill=fill, outline=outline, width=stroke, tags=tag)
        elif shape_key == "cuboid":
            w = 48 * scale
            h = 92 * scale
            canvas.create_rectangle(
                x - w / 2, y - h / 2, x + w / 2, y + h / 2,
                fill=fill, outline=outline, width=stroke, tags=tag,
            )
        elif shape_key == "cube":
            s = 64 * scale
            canvas.create_rectangle(
                x - s / 2, y - s / 2, x + s / 2, y + s / 2,
                fill=fill, outline=outline, width=stroke, tags=tag,
            )
        elif shape_key == "triangular_prism":
            r = 48 * scale
            pts = [
                x, y - r,
                x - r * 0.92, y + r * 0.55,
                x + r * 0.92, y + r * 0.55,
            ]
            canvas.create_polygon(pts, fill=fill, outline=outline, width=stroke, tags=tag)
        else:
            r = 36 * scale
            canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=fill, outline=outline, width=stroke, tags=tag,
            )

    def _next_episode_context(self):
        ins = self._selected_instruction()
        if not ins:
            return None, None, None
        pending = getattr(self, "_pending_recollect", None)
        if pending and pending.get("instruction") == ins:
            return ins, pending["folder"], pending["idx"]
        folder = self.meta.folder_for(ins)
        idx = next_episode_idx(folder)
        return ins, folder, idx

    def _build_plan_for_context(self, ins, folder, idx):
        pending = getattr(self, "_pending_recollect", None)
        if pending and pending.get("instruction") == ins and pending.get("idx") == idx and pending.get("layout_plan"):
            return pending["layout_plan"]

        used_pairs = used_layout_pairs(folder)
        pair_cycle_idx = find_first_unused_pair_cycle_idx(self.layout_pair_cycle, used_pairs)
        if pair_cycle_idx is None:
            pair_cycle_idx = idx % len(self.layout_pair_cycle)
        return build_shape_hole_episode_plan(
            idx,
            self.layout_pair_cycle,
            pairing_seed=self.args.layout_pairing_seed,
            pair_cycle_idx_override=pair_cycle_idx,
        )

    def _layout_usage_summary(self, folder):
        used_pairs = used_layout_pairs(folder)
        total_with_layout = sum(len(v) for v in used_pairs.values())
        duplicate_count = sum(max(0, len(v) - 1) for v in used_pairs.values())
        return used_pairs, total_with_layout, duplicate_count

    def _refresh_next_episode_plan(self):
        ins, folder, idx = self._next_episode_context()
        if ins is None:
            self.layout_text_var.set("Select or add an instruction to show the layout.")
            self._current_layout_plan = None
            self._draw_layout_canvas()
            self._update_subtask_display(0, 0, "")
            return None
        plan = self._build_plan_for_context(ins, folder, idx)
        self._current_layout_plan = plan
        self.layout_text_var.set(format_episode_plan(plan))
        self._draw_layout_canvas()
        subtasks = plan["subtasks"]
        self._update_subtask_display(0, len(subtasks), subtasks[0])
        used_pairs, _total_with_layout, duplicate_count = self._layout_usage_summary(folder)
        self.status_var.set(
            f"Layout usage: {len(used_pairs)}/576 unique pairs, "
            f"{duplicate_count} duplicate saved episodes. "
            f"Current plan pair {plan['pair_cycle_idx'] + 1}/576."
        )
        return plan

    def _show_layout_plan(self, plan, status_prefix=None):
        self._current_layout_plan = plan
        self.layout_text_var.set(format_episode_plan(plan))
        self._draw_layout_canvas()
        subtasks = plan.get("subtasks", [])
        if subtasks:
            self._update_subtask_display(0, len(subtasks), subtasks[0])
        if status_prefix:
            self.status_var.set(f"{status_prefix}: episode_{plan.get('episode_idx')}")

    # ---- async init ----

    def _async_init(self):
        def _wait():
            try:
                self.ui_queue.put(("status", "Waiting for ROS topics..."))
                self.pump.wait_until_ready()
            except Exception as exc:
                self.ui_queue.put(("status", f"[ERROR] wait_until_ready: {exc}"))
                return
            self.worker.start()
            try:
                self._start_preview_thread()
            except Exception as exc:
                print(f"[WARN] preview thread unavailable: {exc}")
            try:
                self.controller.start()
            except Exception as exc:
                self.ui_queue.put(("status", f"[WARN] pedal/lamp unavailable: {exc}"))
            if getattr(self.args, "auto_home", True):
                self.ui_queue.put(("homing", True))
                try:
                    home_ok = go_home_and_wait(
                        pos_threshold=self.args.home_pos_threshold,
                        vel_threshold=self.args.home_vel_threshold,
                        stable_seconds=self.args.home_stable_seconds,
                        timeout=self.args.home_timeout,
                        log=lambda m: self.ui_queue.put(("status", m)),
                    )
                except Exception as exc:
                    self.ui_queue.put(("status", f"[ERROR] startup homing: {exc}"))
                    home_ok = False
                self.ui_queue.put(("homing", False))
                self.ui_queue.put(("home_result", home_ok))
            self.ui_queue.put(("ros_ready", True))
            self.ui_queue.put(("status", "Ready. Select an instruction and press Start (or pedal)."))
            self.ui_queue.put(("ready", True))
        threading.Thread(target=_wait, daemon=True).start()

    # ---- instruction management ----

    def _refresh_instructions(self):
        sel = self._selected_instruction()
        self.instr_list.delete(0, tk.END)
        for ins in self.meta.instructions:
            folder = self.meta.folder_for(ins)
            count = len(list_episode_indices(folder))
            self.instr_list.insert(tk.END, f"[{count}]  {ins}")
        if sel and sel in self.meta.instructions:
            i = self.meta.instructions.index(sel)
            self.instr_list.selection_clear(0, tk.END)
            self.instr_list.selection_set(i)
            self.instr_list.see(i)
        elif self.meta.instructions:
            self.instr_list.selection_set(0)
            self.instr_list.see(0)
        self._refresh_episodes()

    def _selected_instruction(self):
        sel = self.instr_list.curselection()
        if not sel:
            return None
        return self.meta.instructions[sel[0]]

    def _on_instruction_selected(self):
        self._refresh_episodes()
        self._refresh_next_episode_plan()

    def _add_instruction(self):
        text = simpledialog.askstring("Add instruction", "Instruction text:", parent=self.root)
        if text:
            if self.meta.add_instruction(text):
                self._refresh_instructions()

    def _remove_instruction(self):
        ins = self._selected_instruction()
        if not ins:
            return
        if not messagebox.askyesno("Remove instruction",
                                   f"Remove instruction '{ins}' from the list?\n"
                                   "(Existing hdf5 files will NOT be deleted.)"):
            return
        self.meta.remove_instruction(ins)
        self._refresh_instructions()

    # ---- episode list ----

    def _refresh_episodes(self):
        self.episode_list.delete(0, tk.END)
        ins = self._selected_instruction()
        if not ins:
            self._refresh_next_episode_plan()
            return
        folder = self.meta.folder_for(ins)
        for idx in list_episode_indices(folder):
            path = os.path.join(folder, f"episode_{idx}.hdf5")
            try:
                size_kb = os.path.getsize(path) // 1024
                self.episode_list.insert(tk.END, f"episode_{idx}.hdf5   ({size_kb} KB)")
            except OSError:
                self.episode_list.insert(tk.END, f"episode_{idx}.hdf5")
        self._refresh_next_episode_plan()

    def _delete_episode(self):
        ins = self._selected_instruction()
        sel = self.episode_list.curselection()
        if not ins or not sel:
            return
        folder = self.meta.folder_for(ins)
        indices = list_episode_indices(folder)
        if sel[0] >= len(indices):
            return
        idx = indices[sel[0]]
        path = os.path.join(folder, f"episode_{idx}.hdf5")
        if not messagebox.askyesno("Delete episode", f"Delete {path}?"):
            return
        deleted_plan = load_episode_layout_plan(path)
        try:
            os.remove(path)
            self.ui_queue.put(("status", f"Deleted {os.path.basename(path)}"))
        except OSError as exc:
            messagebox.showerror("Delete failed", str(exc))
        self._refresh_instructions()
        if deleted_plan:
            self._pending_recollect = {
                "instruction": ins,
                "folder": folder,
                "idx": idx,
                "layout_plan": deleted_plan,
            }
            self._show_layout_plan(deleted_plan, status_prefix="Deleted layout loaded for recollection")

    # ---- recording controls ----

    def _on_start(self):
        if self.worker.is_recording():
            return
        ins, folder, idx = self._next_episode_context()
        if not ins:
            messagebox.showwarning("No instruction", "Please select an instruction first.")
            return
        target_path = os.path.join(folder, f"episode_{idx}.hdf5")
        if os.path.exists(target_path):
            messagebox.showerror(
                "Episode exists",
                f"{target_path} already exists. Delete it before recollecting this episode.",
            )
            self._pending_recollect = None
            self._refresh_next_episode_plan()
            return
        pending = getattr(self, "_pending_recollect", None)
        plan = self._build_plan_for_context(ins, folder, idx)
        if not (pending and pending.get("instruction") == ins and pending.get("idx") == idx):
            pair = layout_pair_from_plan(plan)
            used_pairs = used_layout_pairs(folder)
            if pair in used_pairs:
                messagebox.showerror(
                    "Duplicate layout pair",
                    (
                        f"Layout pair top={pair[0]} hole={pair[1]} already exists in "
                        f"episodes {used_pairs[pair]}. Refusing to start duplicate collection."
                    ),
                )
                self._refresh_next_episode_plan()
                return
        subtasks = plan["subtasks"]
        self._current_layout_plan = plan
        self.layout_text_var.set(format_episode_plan(plan))
        self._draw_layout_canvas()
        if self.worker.request_start(ins, folder, idx, subtasks, layout_plan=plan):
            if pending and pending.get("instruction") == ins and pending.get("idx") == idx:
                self._pending_recollect = None
            self._update_subtask_display(0, len(subtasks), subtasks[0])
            try:
                self.controller._is_recording = True  # noqa: SLF001
                self.controller._notifier.teach()
            except Exception:
                pass

    def _on_stop(self):
        if self.worker.request_stop():
            try:
                self.controller._is_recording = False  # noqa: SLF001
                self.controller._notifier.saving()
            except Exception:
                pass

    def _on_delete_current(self):
        if not self.worker.is_recording():
            return
        if not messagebox.askyesno(
            "Delete current episode",
            "Stop recording and delete the current unsaved episode?",
        ):
            return
        if self.worker.request_discard_current():
            try:
                self.controller._is_recording = False  # noqa: SLF001
                self.controller._notifier.saving()
            except Exception:
                pass

    # ---- pedal callbacks ----

    def _pedal_start(self):
        # controller: OFF -> ON  (first press, not recording)
        self.root.after(0, self._on_start)

    def _pedal_stop(self):
        # controller: ON -> OFF  (press while recording)
        # Route to subtask-advance logic instead of immediately stopping.
        self.root.after(0, self._on_pedal_recording_press)

    def _on_pedal_recording_press(self):
        """UI-thread handler for a pedal press during recording."""
        if not self.worker.is_recording():
            # Shouldn't normally happen, but keep controller in sync
            self._reset_controller_toggle()
            return

        if self.worker.has_more_subtasks():
            # Still more subtasks: advance and keep recording
            idx, total, text, _has_more = self.worker.advance_subtask()
            # Flip controller back to "recording=True" so the next pedal press
            # triggers _pedal_stop again (not _pedal_start).
            # Also restore green lamp — controller internally turns it yellow on ON→OFF.
            try:
                self.controller._is_recording = True  # noqa: SLF001
                self.controller._notifier.teach()  # noqa: SLF001
            except Exception:
                pass
            self._update_subtask_display(idx, total, text)
            self.status_var.set(f"Subtask {idx + 1}/{total}: {text}")
        else:
            # Last subtask exhausted: stop and save
            self._on_stop()

    # ---- lamp & state mirror ----

    def _set_lamp(self, color):
        self.lamp_canvas.itemconfig(self.lamp_id, fill=color)

    def _poll_joint_status(self):
        try:
            if not getattr(self, "_ros_ready", False):
                self.joint_status_var.set("Joint topics: waiting for ROS streams...")
                self.joint_status_label.configure(foreground="#666666")
                self.root.after(int(self.args.joint_monitor_period * 1000), self._poll_joint_status)
                return
            statuses = self.pump.joint_status(
                timeout=self.args.joint_timeout,
                expected_joints=self.args.expected_joints_per_arm,
            )
            bad = [row for row in statuses if not row["ok"]]
            if bad:
                text = "JOINT ALERT: " + " | ".join(
                    f"{row['label']}: {row['message']}" for row in bad
                )
                self.joint_status_var.set(text)
                self.joint_status_label.configure(foreground="#b00020")
                self._set_lamp("#ff3333")
                if not self._joint_alarm_active:
                    try:
                        self.controller._notifier.error()  # noqa: SLF001
                    except Exception:
                        pass
                self._joint_alarm_active = True
            else:
                text = "Joint topics OK: " + " | ".join(
                    f"{row['label']} {row['age']:.1f}s" for row in statuses
                )
                self.joint_status_var.set(text)
                self.joint_status_label.configure(foreground="#1a6e1a")
                if self._joint_alarm_active:
                    try:
                        if self.worker.is_recording():
                            self.controller._notifier.teach()  # noqa: SLF001
                            self._set_lamp("#22cc22")
                        else:
                            self.controller._notifier.idle()  # noqa: SLF001
                            self._set_lamp("#00aaff")
                    except Exception:
                        pass
                self._joint_alarm_active = False
        except Exception as exc:
            self.joint_status_var.set(f"JOINT MONITOR ERROR: {exc}")
            self.joint_status_label.configure(foreground="#b00020")
        self.root.after(int(self.args.joint_monitor_period * 1000), self._poll_joint_status)

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "recording":
                    self.is_recording_ui = bool(payload)
                    if payload:
                        self.btn_start.config(state=tk.DISABLED)
                        self.btn_stop.config(state=tk.NORMAL)
                        self.btn_next.config(state=tk.NORMAL)
                        if not self._joint_alarm_active:
                            self._set_lamp("#22cc22")
                    else:
                        self.btn_start.config(state=tk.NORMAL)
                        self.btn_stop.config(state=tk.DISABLED)
                        self.btn_next.config(state=tk.DISABLED)
                        if not self._joint_alarm_active:
                            self._set_lamp("#00aaff")
                elif kind == "saving":
                    if payload and not self._joint_alarm_active:
                        self._set_lamp("#ffcc00")
                elif kind == "homing":
                    if payload:
                        if not self._joint_alarm_active:
                            self._set_lamp("#cc66ff")
                        try:
                            self.controller._notifier.homing()  # noqa: SLF001
                        except Exception as exc:
                            print(f"[WARN] could not set hardware lamp to homing: {exc}")
                        self.btn_start.config(state=tk.DISABLED)
                        self.btn_stop.config(state=tk.DISABLED)
                        self.btn_next.config(state=tk.DISABLED)
                        try:
                            self.btn_home.config(state=tk.DISABLED)
                        except Exception:
                            pass
                        self.status_var.set("Returning to home (master+slave)...")
                    else:
                        try:
                            self.btn_home.config(state=tk.NORMAL)
                        except Exception:
                            pass
                elif kind == "live_preview_image":
                    self._render_preview(payload)
                elif kind == "first_frame_image":
                    self._set_thumb(self.first_label, "_first_photo", payload)
                elif kind == "last_frame_image":
                    self._set_thumb(self.last_label, "_last_photo", payload)
                elif kind == "home_result":
                    if payload:
                        self.status_var.set("Home reached. Ready for next episode.")
                    else:
                        self.status_var.set("[WARN] Home timeout/failure. Please verify arms manually.")
                elif kind == "frames":
                    idx, total, text = self.worker.subtask_info()
                    self.status_var.set(
                        f"Recording... {payload}/{self.args.max_timesteps} frames  "
                        f"| subtask {idx + 1}/{total}: {text}"
                    )
                elif kind == "episode_first_frame":
                    self._publish_thumb_async("first_frame_image", payload)
                elif kind == "episode_last_frame":
                    self._publish_thumb_async("last_frame_image", payload)
                elif kind == "episode_aborted":
                    reason = payload
                    self.status_var.set(f"[ABORTED] {reason}")
                    self._set_lamp("#ff3333")
                    self._reset_controller_toggle()
                    self.root.after(3000, lambda: self._set_lamp("#00aaff"))
                elif kind == "episode_saved":
                    self._refresh_instructions()
                    self._refresh_next_episode_plan()
                    self._reset_controller_toggle()
                elif kind == "episode_discarded":
                    meta = payload or {}
                    self._refresh_instructions()
                    layout_plan = meta.get("layout_plan")
                    if layout_plan:
                        self._pending_recollect = {
                            "instruction": meta.get("instruction"),
                            "folder": meta.get("folder"),
                            "idx": meta.get("idx"),
                            "layout_plan": layout_plan,
                        }
                        self._show_layout_plan(layout_plan, status_prefix="Deleted current layout loaded for recollection")
                    self._reset_controller_toggle()
                elif kind == "ros_ready":
                    self._ros_ready = bool(payload)
                elif kind == "ready":
                    pass
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _reset_controller_toggle(self):
        try:
            self.controller._is_recording = False  # noqa: SLF001
            self.controller._notifier.idle()
        except Exception:
            pass

    # ---- Go Home button ----

    def _on_go_home(self):
        if self.worker.is_recording():
            messagebox.showwarning("Busy", "Cannot home while recording. Stop first.")
            return
        if getattr(self, "_homing_in_progress", False):
            return
        self._homing_in_progress = True
        self.ui_queue.put(("homing", True))

        def _do_home():
            try:
                ok = go_home_and_wait(
                    pos_threshold=self.args.home_pos_threshold,
                    vel_threshold=self.args.home_vel_threshold,
                    stable_seconds=self.args.home_stable_seconds,
                    timeout=self.args.home_timeout,
                    log=lambda m: self.ui_queue.put(("status", m)),
                )
            except Exception as exc:
                self.ui_queue.put(("status", f"[ERROR] manual homing: {exc}"))
                ok = False
            self.ui_queue.put(("homing", False))
            self.ui_queue.put(("home_result", ok))
            try:
                self.controller._notifier.idle()  # noqa: SLF001
            except Exception:
                pass
            self._homing_in_progress = False

        threading.Thread(target=_do_home, daemon=True).start()

    # ---- Live preview ----

    def _start_preview_thread(self):
        if not _PIL_AVAILABLE:
            return
        if getattr(self, "_preview_thread", None) is not None:
            return
        self._preview_stop = threading.Event()
        self._pending_preview = False
        self._preview_bridge = CvBridge()
        self._last_preview_msg_ids = (None, None, None)
        self._preview_thread = threading.Thread(
            target=self._preview_loop, daemon=True, name="preview"
        )
        self._preview_thread.start()

    def _stop_preview_thread(self):
        evt = getattr(self, "_preview_stop", None)
        if evt is not None:
            evt.set()

    def _preview_loop(self):
        idle_period = 0.2
        busy_period = 1.0
        while not self._preview_stop.is_set():
            t0 = time.time()
            recording = self.worker.is_recording() if hasattr(self, "worker") else False
            period = busy_period if recording else idle_period

            if self._pending_preview:
                time.sleep(period)
                continue

            try:
                tile = self._compose_live_tile_safe()
            except Exception as exc:
                print(f"[WARN] preview compose failed: {exc}")
                tile = None

            if tile is not None:
                self._pending_preview = True
                self.ui_queue.put(("live_preview_image", tile))

            elapsed = time.time() - t0
            sleep_for = period - elapsed
            if sleep_for > 0:
                self._preview_stop.wait(sleep_for)

    def _compose_live_tile_safe(self):
        front_msg = left_msg = right_msg = None
        try:
            if len(self.pump.img_front_deque):
                front_msg = self.pump.img_front_deque[-1]
            if len(self.pump.img_left_deque):
                left_msg = self.pump.img_left_deque[-1]
            if len(self.pump.img_right_deque):
                right_msg = self.pump.img_right_deque[-1]
        except IndexError:
            return None
        if front_msg is None or left_msg is None or right_msg is None:
            return None

        msg_ids = (id(front_msg), id(left_msg), id(right_msg))
        if msg_ids == self._last_preview_msg_ids:
            return None
        self._last_preview_msg_ids = msg_ids

        try:
            front = self._preview_bridge.imgmsg_to_cv2(front_msg, "bgr8")
            left = self._preview_bridge.imgmsg_to_cv2(left_msg, "bgr8")
            right = self._preview_bridge.imgmsg_to_cv2(right_msg, "bgr8")
        except Exception:
            return None
        target_w, target_h = 320, 240
        front = cv2.resize(front, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        left = cv2.resize(left, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(right, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        composite = np.hstack([front, left, right])
        rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)

    def _publish_thumb_async(self, kind, bgr_img):
        if not _PIL_AVAILABLE:
            return
        try:
            small = cv2.resize(bgr_img, (240, 180))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil)
        except Exception as exc:
            print(f"[WARN] thumb conversion failed: {exc}")
            return
        if kind == "first_frame_image":
            self.first_label.configure(image=photo, text="")
            self._first_photo = photo
        elif kind == "last_frame_image":
            self.last_label.configure(image=photo, text="")
            self._last_photo = photo

    def _set_thumb(self, label, attr, photo):
        pass

    def _render_preview(self, pil_image):
        try:
            if pil_image is None:
                return
            photo = ImageTk.PhotoImage(pil_image)
            self.preview_label.configure(image=photo, text="")
            self._preview_photo = photo
        except Exception as exc:
            print(f"[WARN] preview render failed: {exc}")
        finally:
            self._pending_preview = False

    # ---- shutdown ----

    def _on_close(self):
        if self.is_recording_ui:
            if not messagebox.askyesno("Quit", "An episode is being recorded. Quit anyway?"):
                return
        self.worker.request_stop_thread()
        try:
            self._stop_preview_thread()
        except Exception:
            pass
        try:
            self.controller.stop()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def get_arguments():
    parser = argparse.ArgumentParser(description="UI + pedal data collection pipeline (subtask mode)")
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--task_name", type=str, default="aloha_pipeline")
    parser.add_argument("--max_timesteps", type=int, default=9000)
    parser.add_argument(
        "--camera_names", nargs="+", type=str,
        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
    )
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--master_arm_left_topic", type=str, default="/master/joint_left")
    parser.add_argument("--master_arm_right_topic", type=str, default="/master/joint_right")
    parser.add_argument("--puppet_arm_left_topic", type=str, default="/puppet/joint_left")
    parser.add_argument("--puppet_arm_right_topic", type=str, default="/puppet/joint_right")
    parser.add_argument("--joint_timeout", type=float, default=1.5)
    parser.add_argument("--expected_joints_per_arm", type=int, default=7)
    parser.add_argument("--joint_monitor_period", type=float, default=0.5)
    parser.add_argument("--frame_rate", type=int, default=30)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--lamp_port", type=str, default=None)
    parser.add_argument("--pedal_device", type=str, default=None)
    parser.add_argument("--trigger_key", type=str, default="enter")
    parser.add_argument(
        "--instructions", nargs="*", default=None,
        help="Seed initial instruction list (only when meta is empty).",
    )
    parser.add_argument(
        "--layout_pairing_seed", type=int, default=DEFAULT_LAYOUT_PAIRING_SEED,
        help=(
            "Seed used to shuffle the 24 hole layouts against the 24 top layouts. "
            "The sequence is deterministic for the same seed."
        ),
    )
    parser.add_argument("--auto_home", dest="auto_home", action="store_true",
                        help="保存成功后自动主从臂回零 (默认开启)")
    parser.add_argument("--no_auto_home", dest="auto_home", action="store_false",
                        help="关闭保存后自动回零")
    parser.set_defaults(auto_home=True)
    parser.add_argument("--home_pos_threshold", type=float, default=0.1)
    parser.add_argument("--home_vel_threshold", type=float, default=0.1)
    parser.add_argument("--home_stable_seconds", type=float, default=0.2)
    parser.add_argument("--home_timeout", type=float, default=4.5)
    return parser.parse_args()


def main():
    args = get_arguments()
    app = CollectorApp(args)
    app.run()


if __name__ == "__main__":
    main()
