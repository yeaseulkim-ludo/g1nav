"""Visualize a collected episode: replays qpos and saves third-person MP4."""
import os
import pickle
import sys
sys.path.insert(0, "/home/ludo-us/work/g1nav")

import cv2
import mujoco

from code.env.arena import G1NavArena

EPISODES_DIR = "dataset/episodes"

ep_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
path = os.path.join(EPISODES_DIR, f"ep_{ep_idx:04d}.pkl")

with open(path, "rb") as f:
    ep = pickle.load(f)

print(f"Instruction : {ep.instruction}")
print(f"Target obj  : {ep.target_obj}")
print(f"Steps       : {len(ep.steps)}")
print(f"Success     : {ep.success}")

os.makedirs("images", exist_ok=True)
out = f"images/episode_{ep_idx:04d}.mp4"

arena = G1NavArena()

# Restore object positions from the episode so they render in the right place
for obj_name, pos in ep.object_positions.items():
    body_id  = mujoco.mj_name2id(arena.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    mocap_id = arena.model.body_mocapid[body_id]
    arena.data.mocap_pos[mocap_id] = pos

# Sim runs at 200 Hz; subsample to 25 fps → keep every 8th frame (real-time speed)
SIM_HZ = 200
TARGET_FPS = 25
STRIDE = SIM_HZ // TARGET_FPS  # 8

h, w = arena.TP_H, arena.TP_W
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out, fourcc, TARGET_FPS, (w, h))

for i, step in enumerate(ep.steps):
    if i % STRIDE != 0:
        continue
    arena.data.qpos[:] = step.qpos
    mujoco.mj_forward(arena.model, arena.data)
    frame = arena.render_third_person()
    label = f"{ep.instruction}  |  step {i}/{len(ep.steps)}"
    cv2.putText(frame, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1, cv2.LINE_AA)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

writer.release()
arena.close()
print(f"\nSaved {out}  ({len(ep.steps)} frames @ {TARGET_FPS} fps)")
