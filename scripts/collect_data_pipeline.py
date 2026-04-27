# -- coding: UTF-8
"""
Full data-collection pipeline with Tkinter UI + pedal control.

Features
--------
* Pre-set a list of instructions (language prompts) before/while collecting.
* For each instruction, collect multiple episodes with Start / Stop / Next
  controlled either from the UI buttons or from the pedal (mapped to Enter).
* USB lamp follows recording state (cyan = idle, green = recording,
  yellow = saving).
* Failed episodes can be deleted directly from the UI episode list.
* Switch instruction by selecting in the UI; new episodes are stored in a
  per-instruction subdirectory.

Per task layout on disk
-----------------------
<dataset_dir>/<task_name>/
    pipeline_meta.json                # instruction list + slug mapping
    <instruction_slug>/
        episode_0.hdf5
        episode_1.hdf5
        ...
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


# ------------------------------------------------------------------
# HDF5 saving
# ------------------------------------------------------------------

def save_episode(args, timesteps, actions, dataset_path, instruction_text):
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
        root.attrs["instruction"] = instruction_text
        root.attrs["frame_rate"] = int(args.frame_rate)

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
        self.instructions = []   # list[str]
        self.slugs = {}          # str -> str
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

        # Wall-clock time of last received message for each stream
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
        """Return list of stream keys whose last message is older than timeout seconds."""
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

        # warm up: wait for non-zero CAN values
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
        self.ui_queue = ui_queue   # queue for UI status updates

        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._is_recording = False
        self._stop_requested = False
        self._episode = None
        self._episode_meta = None  # dict(folder, idx, instruction)
        self._stall_since = None   # wall time when frames first stopped arriving
        self._last_front = None    # most recent front-cam frame for end-thumb

    # ---- control API (thread-safe, called from UI / pedal) ----
    def request_stop_thread(self):
        self._stop_evt.set()

    def request_start(self, instruction, folder, idx):
        with self._lock:
            if self._is_recording:
                return False
            self._episode = {"timesteps": [], "actions": [], "count": 0}
            self._episode_meta = {"folder": folder, "idx": idx, "instruction": instruction}
            self._is_recording = True
            self._stop_requested = False
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
            if not recording:
                rate.sleep()
                continue
            if stop_req:
                self._finalize_episode()
                rate.sleep()
                continue

            result = self.pump.get_synced_frame()
            if result is None:
                # Stall detection: if frames stop arriving while recording,
                # check whether a stream has gone silent.
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

            self._stall_since = None  # frames flowing normally
            self._append_step(result)
            rate.sleep()

    def _append_step(self, result):
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

        with self._lock:
            ep = self._episode
            if ep is None:
                return
            ep["count"] += 1
            count = ep["count"]
            if count == 1:
                ep["timesteps"].append(obs)
                # Push the very first frame as the "episode start" thumbnail
                self.ui_queue.put(("episode_first_frame", img_front.copy()))
                return
            ep["actions"].append(action)
            ep["timesteps"].append(obs)
            # Continuously update "last seen" frame; on stop we publish it
            # as the "episode end" thumbnail.
            self._last_front = img_front
            if count % 50 == 0:
                self.ui_queue.put(("frames", count - 1))
            if len(ep["actions"]) >= self.args.max_timesteps:
                self.ui_queue.put(("status", "Reached max_timesteps. Auto-stopping."))
                self._stop_requested = True

    def _abort_episode(self, reason: str = "aborted"):
        """Discard current episode without saving."""
        with self._lock:
            self._episode = None
            self._episode_meta = None
            self._is_recording = False
            self._stop_requested = False
        self.ui_queue.put(("episode_aborted", reason))
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

        if episode is None or meta is None:
            self.ui_queue.put(("recording", False))
            return

        actions = episode["actions"]
        timesteps = episode["timesteps"]

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
            save_episode(self.args, timesteps, actions, dataset_path, meta["instruction"])
            self.ui_queue.put(("status",
                f"Saved episode_{meta['idx']} ({len(actions)} frames) :: {meta['instruction']}"))
            save_ok = True
        except Exception as exc:
            self.ui_queue.put(("status", f"[ERROR] save failed: {exc}"))
        self.ui_queue.put(("saving", False))

        # 保存成功后, 自动主从臂回零; 回零真正完成才解锁下一次采集.
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
    # NOTE: do NOT do `from PIL import Image` — it would shadow
    # `sensor_msgs.msg.Image` imported above and break rospy.Subscriber.
    from PIL import Image as PILImage
    from PIL import ImageTk
    _PIL_AVAILABLE = True
except Exception as _pil_exc:  # pragma: no cover - graceful fallback
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
        # seed instructions from CLI if provided and meta empty
        if not self.meta.instructions and args.instructions:
            for ins in args.instructions:
                self.meta.add_instruction(ins)

        # UI<->worker queue
        self.ui_queue = queue.Queue()

        # ROS
        rospy.init_node("collect_data_pipeline", anonymous=True)
        self.pump = RosDataPump(args)

        # Pedal + lamp
        self.controller = RecordingController(
            port=args.lamp_port,
            pedal_device=args.pedal_device,
            trigger_key=args.trigger_key,
        )
        # Map controller events -> our toggle (so pedal acts like Start/Stop)
        self.controller.on_recording_start(self._pedal_start)
        self.controller.on_recording_stop(self._pedal_stop)

        # Worker
        self.worker = CollectorWorker(args, self.pump, self.ui_queue)

        # ----- Build UI -----
        self.root = tk.Tk()
        self.root.title("Data Collection Pipeline")
        self.root.geometry("1440x820")
        self._build_ui()

        # state
        self.is_recording_ui = False  # mirrored from worker via queue

        # Wait for streams in background to avoid blocking UI
        self.root.after(100, self._async_init)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_queue)
        # Live camera preview runs on a background thread so that UI
        # rendering and image decoding never block the collector worker.
        # Started after ROS topics are ready, see _async_init.

    # ---- UI construction ----
    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, **pad)
        ttk.Label(top, text="Task:").pack(side=tk.LEFT)
        ttk.Label(top, text=self.args.task_name, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(top, text="Dir:").pack(side=tk.LEFT)
        ttk.Label(top, text=self.task_dir).pack(side=tk.LEFT)

        # Instructions panel (left) + Episodes panel (middle) + Camera preview (right)
        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, **pad)

        left = ttk.LabelFrame(body, text="Instructions")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.instr_list = tk.Listbox(left, exportselection=False)
        self.instr_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.instr_list.bind("<<ListboxSelect>>", lambda _e: self._refresh_episodes())

        instr_btns = ttk.Frame(left)
        instr_btns.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(instr_btns, text="Add", command=self._add_instruction).pack(side=tk.LEFT)
        ttk.Button(instr_btns, text="Remove", command=self._remove_instruction).pack(side=tk.LEFT, padx=4)

        middle = ttk.LabelFrame(body, text="Episodes for selected instruction")
        middle.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.episode_list = tk.Listbox(middle, exportselection=False)
        self.episode_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        ep_btns = ttk.Frame(middle)
        ep_btns.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(ep_btns, text="Delete selected episode", command=self._delete_episode).pack(side=tk.LEFT)
        ttk.Button(ep_btns, text="Refresh", command=self._refresh_episodes).pack(side=tk.LEFT, padx=4)

        # ---- Camera preview + start/end frame thumbnails ----
        right = ttk.LabelFrame(body, text="Live cameras  /  Episode frames")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        live = ttk.LabelFrame(right, text="Live (front | left | right)")
        live.pack(fill=tk.X, padx=4, pady=4)
        self.preview_label = ttk.Label(live, text="(waiting for camera frames...)",
                                       anchor="center")
        self.preview_label.pack(fill=tk.X, padx=4, pady=4)
        self._preview_photo = None  # keep ref so PhotoImage isn'\''t GCd

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

        # Control buttons
        ctl = ttk.LabelFrame(self.root, text="Control")
        ctl.pack(fill=tk.X, **pad)
        self.btn_start = ttk.Button(ctl, text="Start (Pedal)", command=self._on_start)
        self.btn_start.pack(side=tk.LEFT, padx=6, pady=6)
        self.btn_stop = ttk.Button(ctl, text="Stop & Save (Pedal)", command=self._on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=6, pady=6)
        self.btn_next = ttk.Button(ctl, text="Stop & Start Next", command=self._on_next, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Separator(ctl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.btn_home = ttk.Button(ctl, text="Go Home (主从臂归零)", command=self._on_go_home)
        self.btn_home.pack(side=tk.LEFT, padx=6, pady=6)

        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        status = ttk.Frame(self.root)
        status.pack(fill=tk.X, **pad)
        self.lamp_canvas = tk.Canvas(status, width=18, height=18, highlightthickness=0)
        self.lamp_canvas.pack(side=tk.LEFT, padx=4)
        self.lamp_id = self.lamp_canvas.create_oval(2, 2, 16, 16, fill="gray")
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT, padx=6)

        self._refresh_instructions()

    def _async_init(self):
        def _wait():
            try:
                self.ui_queue.put(("status", "Waiting for ROS topics..."))
                self.pump.wait_until_ready()
            except Exception as exc:
                self.ui_queue.put(("status", f"[ERROR] wait_until_ready: {exc}"))
                return
            # Start collector worker first so buttons always work
            self.worker.start()
            # Live preview runs in its own thread (zero work on UI thread or
            # collector worker thread), so safe to start now.
            try:
                self._start_preview_thread()
            except Exception as exc:
                print(f"[WARN] preview thread unavailable: {exc}")
            # Controller (lamp + pedal) may fail gracefully if hardware absent
            try:
                self.controller.start()
            except Exception as exc:
                self.ui_queue.put(("status", f"[WARN] pedal/lamp unavailable: {exc}"))
            # 启动时先回零并恢复示教模式，确保主臂处于可拖拽状态
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
        # restore selection
        if sel and sel in self.meta.instructions:
            i = self.meta.instructions.index(sel)
            self.instr_list.selection_clear(0, tk.END)
            self.instr_list.selection_set(i)
            self.instr_list.see(i)
        self._refresh_episodes()

    def _selected_instruction(self):
        sel = self.instr_list.curselection()
        if not sel:
            return None
        return self.meta.instructions[sel[0]]

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
            return
        folder = self.meta.folder_for(ins)
        for idx in list_episode_indices(folder):
            path = os.path.join(folder, f"episode_{idx}.hdf5")
            try:
                size_kb = os.path.getsize(path) // 1024
                self.episode_list.insert(tk.END, f"episode_{idx}.hdf5   ({size_kb} KB)")
            except OSError:
                self.episode_list.insert(tk.END, f"episode_{idx}.hdf5")

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
        try:
            os.remove(path)
            self.ui_queue.put(("status", f"Deleted {os.path.basename(path)}"))
        except OSError as exc:
            messagebox.showerror("Delete failed", str(exc))
        self._refresh_instructions()

    # ---- recording controls ----
    def _on_start(self):
        if self.worker.is_recording():
            return
        ins = self._selected_instruction()
        if not ins:
            messagebox.showwarning("No instruction", "Please select an instruction first.")
            return
        folder = self.meta.folder_for(ins)
        idx = next_episode_idx(folder)
        if self.worker.request_start(ins, folder, idx):
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

    def _on_next(self):
        # Stop current episode and arm a flag to auto-start next when save done
        self._auto_start_next = True
        self.worker.request_stop()

    # ---- pedal -> toggle ----
    # The RecordingController toggles its internal flag; we use both callbacks
    # as a single "pedal pressed" signal and dispatch by current worker state.
    def _pedal_start(self):
        # called when controller transitions OFF -> ON (1st press)
        self.root.after(0, self._on_start)

    def _pedal_stop(self):
        # called when controller transitions ON -> OFF (2nd press)
        self.root.after(0, self._on_stop)

    # ---- lamp & state mirror ----
    def _set_lamp(self, color):
        self.lamp_canvas.itemconfig(self.lamp_id, fill=color)

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
                        self._set_lamp("#22cc22")
                    else:
                        self.btn_start.config(state=tk.NORMAL)
                        self.btn_stop.config(state=tk.DISABLED)
                        self.btn_next.config(state=tk.DISABLED)
                        self._set_lamp("#00aaff")
                elif kind == "saving":
                    if payload:
                        self._set_lamp("#ffcc00")
                elif kind == "homing":
                    if payload:
                        self._set_lamp("#cc66ff")  # purple
                        # Drive the *hardware* USB lamp purple as well.  The
                        # previous code only updated the on-screen indicator
                        # so the physical lamp stayed yellow during homing.
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
                    self.status_var.set(f"Recording... {payload}/{self.args.max_timesteps} frames")
                elif kind == "episode_first_frame":
                    self._publish_thumb_async("first_frame_image", payload)
                elif kind == "episode_last_frame":
                    self._publish_thumb_async("last_frame_image", payload)
                elif kind == "episode_aborted":
                    reason = payload
                    self.status_var.set(f"[ABORTED] {reason}")
                    self._set_lamp("#ff3333")
                    self._reset_controller_toggle()
                    # Schedule lamp back to idle after 3 s so user notices the error
                    self.root.after(3000, lambda: self._set_lamp("#00aaff"))
                elif kind == "episode_saved":
                    self._refresh_instructions()
                    if getattr(self, "_auto_start_next", False):
                        self._auto_start_next = False
                        # Reset controller internal toggle so pedal stays in sync
                        self._reset_controller_toggle()
                        self.root.after(150, self._on_start)
                    else:
                        self._reset_controller_toggle()
                elif kind == "ready":
                    pass
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _reset_controller_toggle(self):
        # The RecordingController keeps an internal _is_recording flag; after a
        # save we ensure it matches "not recording" so the next pedal press
        # triggers _pedal_start (not _pedal_stop).
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
            # Restore idle lamp on hardware once homing finishes
            try:
                self.controller._notifier.idle()  # noqa: SLF001
            except Exception:
                pass
            self._homing_in_progress = False

        threading.Thread(target=_do_home, daemon=True).start()

    # ---- Live preview ----
    #
    # Design notes (避免影响数据采集):
    #   * 全部 cv_bridge 解码 / resize / hstack / cvtColor / PIL 转换都在
    #     独立后台线程里做, UI 主线程只负责把 PIL.Image 包成 PhotoImage 并
    #     贴到 Label 上, 因此 UI 卡顿不会拖累采集 worker.
    #   * 录制中刷新率降到 1 FPS, 空闲时 5 FPS, 给采集线程让出 CPU/GIL.
    #   * 仅 peek (`deque[-1]`) **不** popleft, 不会影响采集 worker 从
    #     队首消费帧的逻辑; 也不修改任何已入队的 ROS 消息.
    #   * 使用专属 CvBridge, 与 RosDataPump.bridge 隔离.
    #   * 使用 `_pending_preview` 标志避免向 UI 队列堆积积压.
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
        idle_period = 0.2     # 5 FPS when not recording
        busy_period = 1.0     # 1 FPS during recording, minimal interference
        while not self._preview_stop.is_set():
            t0 = time.time()
            recording = self.worker.is_recording() if hasattr(self, "worker") else False
            period = busy_period if recording else idle_period

            # Skip work entirely if UI hasn't yet drained the previous frame
            # (prevents queue buildup if the UI thread stalls).
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
                # Use Event.wait so stop is responsive
                self._preview_stop.wait(sleep_for)

    def _compose_live_tile_safe(self):
        # Snapshot latest message refs without popping the deques. CPython
        # deque indexing is atomic at the C level, but the deque can briefly
        # be empty between the len() check and the [-1] read, so we guard
        # with try/except.
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

        # De-duplicate: if all three msg refs are the same as last cycle,
        # there is no new content to display, skip the decode entirely.
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
        target_w = 320
        target_h = 240
        front = cv2.resize(front, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        left = cv2.resize(left, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(right, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        composite = np.hstack([front, left, right])
        rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)

    def _publish_thumb_async(self, kind, bgr_img):
        """Convert a captured BGR frame to a PhotoImage on the UI thread."""
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
        # placeholder kept for symmetry; thumbs are pushed via
        # _publish_thumb_async directly
        pass

    def _render_preview(self, pil_image):
        """Called on UI thread. Cheap: PIL.Image -> PhotoImage -> label."""
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
    parser = argparse.ArgumentParser(description="UI + pedal data collection pipeline")
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
    parser.add_argument("--frame_rate", type=int, default=30)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--lamp_port", type=str, default=None)
    parser.add_argument("--pedal_device", type=str, default=None)
    parser.add_argument("--trigger_key", type=str, default="enter")
    parser.add_argument(
        "--instructions", nargs="*", default=None,
        help="Optional: seed initial instruction list (only when meta is empty).",
    )
    # 主从臂自动回零相关
    parser.add_argument("--auto_home", dest="auto_home", action="store_true",
                        help="保存成功后自动主从臂回零 (默认开启)")
    parser.add_argument("--no_auto_home", dest="auto_home", action="store_false",
                        help="关闭保存后自动回零")
    parser.set_defaults(auto_home=True)
    parser.add_argument("--home_pos_threshold", type=float, default=0.1,
                        help="回零完成位置阈值 (rad)")
    parser.add_argument("--home_vel_threshold", type=float, default=0.1,
                        help="回零完成速度阈值")
    parser.add_argument("--home_stable_seconds", type=float, default=0.2,
                        help="满足阈值的持续时间")
    parser.add_argument("--home_timeout", type=float, default=4.5,
                        help="回零等待超时 (s)")
    return parser.parse_args()


def main():
    args = get_arguments()
    app = CollectorApp(args)
    app.run()


if __name__ == "__main__":
    main()
