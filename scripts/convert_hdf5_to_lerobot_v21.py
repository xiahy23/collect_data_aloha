# -- coding: UTF-8
"""
Convert collected HDF5 episodes into a simple LeRobot v2.1-style local dataset layout.

Output layout:
    <out_dir>/
        meta/
            info.json
            episodes.jsonl
            tasks.jsonl
            episodes_stats.jsonl
        data/
            chunk-000/
                episode-000000.parquet
                episode-000001.parquet
                ...
        videos/
            chunk-000/
                observation.images.cam_high/
                    episode-000000.mp4
                observation.images.cam_left_wrist/
                    episode-000000.mp4
                observation.images.cam_right_wrist/
                    episode-000000.mp4
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pandas is required. Original error: {exc}")


def discover_episodes(src_dir):
    src = Path(src_dir)
    meta_path = src / "pipeline_meta.json"

    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        slug_to_text = {slug: text for text, slug in meta.get("slugs", {}).items()}
        ordered_instructions = list(meta.get("instructions", []))
        grouped = {}

        for slug_dir in sorted(src.iterdir()):
            if not slug_dir.is_dir():
                continue
            files = sorted(
                slug_dir.glob("episode_*.hdf5"),
                key=lambda p: int(p.stem.split("_")[-1]),
            )
            if files:
                grouped[slug_to_text.get(slug_dir.name, slug_dir.name)] = files

        if not ordered_instructions:
            ordered_instructions = sorted(grouped.keys())

        for instruction in ordered_instructions:
            for path in grouped.get(instruction, []):
                yield instruction, path
        return

    flat_files = sorted(
        src.glob("episode_*.hdf5"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for path in flat_files:
        yield "default_task", path


def discover_single_task_episodes(task_dir, task_name):
    task_path = Path(task_dir)
    files = sorted(
        task_path.glob("episode_*.hdf5"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for path in files:
        yield task_name, path


def infer_task_name_from_dir(task_dir):
    return Path(task_dir).name.replace("_", " ").strip()


def decode_jpeg(buf):
    arr = np.frombuffer(bytes(buf), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("failed to decode jpeg frame")
    return img


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_video(video_path, frames_bgr, fps):
    if not frames_bgr:
        return
    video_path = Path(video_path)
    ensure_dir(video_path.parent)

    with tempfile.TemporaryDirectory(prefix="lerobot_v21_frames_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for idx, frame in enumerate(frames_bgr):
            frame_path = tmp_dir / f"frame_{idx:06d}.png"
            ok = cv2.imwrite(str(frame_path), frame)
            if not ok:
                raise RuntimeError(f"failed to write temporary frame: {frame_path}")

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg libx264 encode failed for {video_path}:\n{result.stderr.strip()}"
            )


def summarize_array(name, array):
    return {
        "name": name,
        "shape": list(array.shape),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def convert_episode(episode_idx, instruction, h5_path, out_dir):
    with h5py.File(h5_path, "r") as f:
        qpos = np.asarray(f["/observations/qpos"], dtype=np.float32)
        qvel = np.asarray(f["/observations/qvel"], dtype=np.float32)
        effort = np.asarray(f["/observations/effort"], dtype=np.float32)
        action = np.asarray(f["/action"], dtype=np.float32)
        base_action = np.asarray(f["/base_action"], dtype=np.float32)
        fps = int(f.attrs.get("frame_rate", 30))
        cam_names = list(f["/observations/images"].keys())
        image_shape = [
            int(x) for x in f.attrs.get("image_shape", np.array([480, 640, 3], dtype=np.int32))
        ]

        decoded_frames = {cam: [] for cam in cam_names}
        for cam in cam_names:
            for item in f[f"/observations/images/{cam}"][...]:
                decoded_frames[cam].append(decode_jpeg(item))

    num_frames = qpos.shape[0]
    chunk_dir = out_dir / "data" / "chunk-000"
    ensure_dir(chunk_dir)

    rows = []
    for frame_idx in range(num_frames):
        row = {
            "episode_index": episode_idx,
            "frame_index": frame_idx,
            "timestamp": frame_idx / max(fps, 1),
            "task": instruction,
            "observation.state": qpos[frame_idx].tolist(),
            "observation.qvel": qvel[frame_idx].tolist(),
            "observation.effort": effort[frame_idx].tolist(),
            "action": action[frame_idx].tolist(),
            "base_action": base_action[frame_idx].tolist(),
        }
        for cam in cam_names:
            row[f"observation.images.{cam}"] = (
                f"videos/chunk-000/observation.images.{cam}/episode-{episode_idx:06d}.mp4"
            )
        rows.append(row)

    parquet_path = chunk_dir / f"episode-{episode_idx:06d}.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    for cam in cam_names:
        video_dir = out_dir / "videos" / "chunk-000" / f"observation.images.{cam}"
        ensure_dir(video_dir)
        video_path = video_dir / f"episode-{episode_idx:06d}.mp4"
        write_video(video_path, decoded_frames[cam], fps)

    episode_meta = {
        "episode_index": episode_idx,
        "task": instruction,
        "num_frames": num_frames,
        "fps": fps,
    }
    episode_stats = {
        "episode_index": episode_idx,
        "task": instruction,
        "num_frames": num_frames,
        "observation.state": summarize_array("observation.state", qpos),
        "action": summarize_array("action", action),
    }
    return {
        "episode_meta": episode_meta,
        "episode_stats": episode_stats,
        "cam_names": cam_names,
        "fps": fps,
        "image_shape": image_shape,
    }


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="HDF5 -> LeRobot v2.1-style local converter")
    parser.add_argument("--src_dir", required=True, help="Source dataset directory.")
    parser.add_argument("--out_dir", required=True, help="Output dataset directory.")
    parser.add_argument("--repo_id", default="local/aloha_dataset", help="Dataset id written into meta/info.json")
    parser.add_argument("--robot_type", default="aloha-piper")
    parser.add_argument(
        "--task_name",
        default=None,
        help=(
            "Optional fixed task name. Use together with a single task folder containing episode_*.hdf5. "
            "If omitted in single-task mode, the folder name is used with '_' replaced by spaces."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.src_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)

    ensure_dir(out_dir / "meta")
    ensure_dir(out_dir / "data" / "chunk-000")
    ensure_dir(out_dir / "videos" / "chunk-000")

    if args.task_name:
        episodes = list(discover_single_task_episodes(src_dir, args.task_name))
    elif src_dir.is_dir() and (src_dir / "pipeline_meta.json").is_file() is False and list(src_dir.glob("episode_*.hdf5")):
        inferred_task_name = infer_task_name_from_dir(src_dir)
        episodes = list(discover_single_task_episodes(src_dir, inferred_task_name))
    else:
        episodes = list(discover_episodes(src_dir))
    if not episodes:
        raise SystemExit(f"No episodes found under {src_dir}")

    tasks_written = set()
    episodes_jsonl = out_dir / "meta" / "episodes.jsonl"
    tasks_jsonl = out_dir / "meta" / "tasks.jsonl"
    stats_jsonl = out_dir / "meta" / "episodes_stats.jsonl"

    first_info = None
    for episode_idx, (instruction, h5_path) in enumerate(episodes):
        print(f"[{episode_idx + 1}/{len(episodes)}] {instruction} <- {h5_path}")
        result = convert_episode(episode_idx, instruction, h5_path, out_dir)
        append_jsonl(episodes_jsonl, result["episode_meta"])
        append_jsonl(stats_jsonl, result["episode_stats"])
        if instruction not in tasks_written:
            append_jsonl(tasks_jsonl, {"task": instruction})
            tasks_written.add(instruction)
        if first_info is None:
            first_info = result

    info = {
        "repo_id": args.repo_id,
        "robot_type": args.robot_type,
        "num_episodes": int(len(episodes)),
        "camera_names": first_info["cam_names"],
        "fps": int(first_info["fps"]),
        "image_shape": first_info["image_shape"],
        "format": "lerobot_v2.1_style_local",
    }
    with open(out_dir / "meta" / "info.json", "w", encoding="utf-8") as fp:
        json.dump(info, fp, ensure_ascii=False, indent=2)

    print(f"[DONE] saved v2.1-style dataset to: {out_dir}")


if __name__ == "__main__":
    main()
