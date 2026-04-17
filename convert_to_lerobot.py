"""
Convert G1Nav pkl episodes to LeRobot v2.1 format for GR00T fine-tuning.

Output structure:
  dataset/lerobot/
    meta/{modality.json, info.json, tasks.jsonl, episodes.jsonl}
    data/chunk-000/episode_XXXXXX.parquet
    videos/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4
"""

import glob
import json
import os
import pickle
import sys
sys.path.insert(0, "/home/ludo-us/work/g1nav")

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Config ────────────────────────────────────────────────────────────────────

EPISODES_DIR  = "dataset/episodes"
OUTPUT_DIR    = "dataset/lerobot"
SIM_HZ        = 200
TARGET_FPS    = 10           # subsample to 10 Hz
STRIDE        = SIM_HZ // TARGET_FPS   # keep every 20th step
STATE_DIM     = 30           # 15 joint pos + 15 joint vel
ACTION_DIM    = 15           # 15 target joint positions
IMG_H, IMG_W  = 224, 224

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_episodes():
    paths = sorted(glob.glob(os.path.join(EPISODES_DIR, "ep_*.pkl")))
    episodes = []
    for p in paths:
        with open(p, "rb") as f:
            episodes.append(pickle.load(f))
    return episodes


def write_video(frames_rgb, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, TARGET_FPS, (IMG_W, IMG_H))
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_parquet(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path)

# ── Main conversion ───────────────────────────────────────────────────────────

def convert():
    episodes = load_episodes()
    print(f"Found {len(episodes)} episodes")

    # Collect unique task strings
    task_set  = sorted({ep.instruction for ep in episodes})
    task2idx  = {t: i for i, t in enumerate(task_set)}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/meta", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/data/chunk-000", exist_ok=True)

    global_frame_idx = 0
    episode_meta     = []
    total_frames     = 0

    for ep_idx, ep in enumerate(episodes):
        # Subsample steps
        steps = [s for i, s in enumerate(ep.steps) if i % STRIDE == 0]
        n     = len(steps)

        # Build video frames and parquet rows
        video_frames = []
        rows         = []
        task_idx     = task2idx[ep.instruction]

        for frame_idx, step in enumerate(steps):
            state  = np.concatenate([step.joints, step.joint_vels]).astype(np.float32)
            action = step.action.astype(np.float32)

            rows.append({
                "observation.state":                          state.tolist(),
                "action":                                     action.tolist(),
                "timestamp":                                  float(frame_idx) / TARGET_FPS,
                "frame_index":                                frame_idx,
                "episode_index":                              ep_idx,
                "index":                                      global_frame_idx + frame_idx,
                "task_index":                                 task_idx,
                "annotation.human.action.task_description":   task_idx,
                "next.done":                                  frame_idx == n - 1,
            })
            video_frames.append(step.rgb_ego)

        # Write parquet
        parquet_path = f"{OUTPUT_DIR}/data/chunk-000/episode_{ep_idx:06d}.parquet"
        write_parquet(rows, parquet_path)

        # Write video
        video_path = (
            f"{OUTPUT_DIR}/videos/chunk-000/"
            f"observation.images.ego_view/episode_{ep_idx:06d}.mp4"
        )
        write_video(video_frames, video_path)

        episode_meta.append({
            "episode_index": ep_idx,
            "tasks": [ep.instruction],
            "length": n,
        })

        global_frame_idx += n
        total_frames     += n
        print(f"  [{ep_idx:03d}] {ep.instruction:<45}  frames={n}")

    # ── meta/tasks.jsonl ──────────────────────────────────────────────────────
    with open(f"{OUTPUT_DIR}/meta/tasks.jsonl", "w") as f:
        for task, idx in task2idx.items():
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # ── meta/episodes.jsonl ───────────────────────────────────────────────────
    with open(f"{OUTPUT_DIR}/meta/episodes.jsonl", "w") as f:
        for em in episode_meta:
            f.write(json.dumps(em) + "\n")

    # ── meta/modality.json ────────────────────────────────────────────────────
    modality = {
        "state": {
            "lower_body_joints":     {"start": 0,  "end": 15},
            "lower_body_joint_vels": {"start": 15, "end": 30},
        },
        "action": {
            "lower_body_joints": {"start": 0, "end": 15},
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
        },
        "annotation": {
            "human.action.task_description": {
                "original_key": "annotation.human.action.task_description"
            }
        },
    }
    with open(f"{OUTPUT_DIR}/meta/modality.json", "w") as f:
        json.dump(modality, f, indent=4)

    # ── meta/info.json ────────────────────────────────────────────────────────
    joint_names = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
        "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
        "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
    ]
    state_names  = [f"{j}.pos" for j in joint_names] + [f"{j}.vel" for j in joint_names]
    action_names = [f"{j}.pos" for j in joint_names]

    info = {
        "codebase_version": "v2.1",
        "robot_type": "g1_29dof",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(task_set),
        "chunks_size": 1000,
        "fps": TARGET_FPS,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32", "shape": [STATE_DIM], "names": state_names,
            },
            "action": {
                "dtype": "float32", "shape": [ACTION_DIM], "names": action_names,
            },
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [IMG_H, IMG_W, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": IMG_H, "video.width": IMG_W,
                    "video.codec": "mp4v", "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": TARGET_FPS, "video.channels": 3,
                    "has_audio": False,
                },
            },
            "timestamp":      {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":    {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index":  {"dtype": "int64",   "shape": [1], "names": None},
            "index":          {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":     {"dtype": "int64",   "shape": [1], "names": None},
        },
        "total_chunks": 1,
        "total_videos": len(episodes),
    }
    with open(f"{OUTPUT_DIR}/meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"\nDone. {len(episodes)} episodes, {total_frames} frames → {OUTPUT_DIR}")
    print(f"Next: run stats generation")
    print(f"  cd /home/ludo-us/work/Isaac-GR00T")
    print(f"  python gr00t/data/stats.py /home/ludo-us/work/g1nav/{OUTPUT_DIR} G1NAV")


if __name__ == "__main__":
    convert()
