"""Visualize a collected episode: saves ego-view MP4 + prints stats."""
import os
import pickle
import sys
sys.path.insert(0, "/home/ludo-us/work/g1nav")

import cv2
import numpy as np
from PIL import Image, ImageDraw

EPISODES_DIR = "dataset/episodes"

ep_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
path = os.path.join(EPISODES_DIR, f"ep_{ep_idx:04d}.pkl")

with open(path, "rb") as f:
    ep = pickle.load(f)

print(f"Instruction : {ep.instruction}")
print(f"Target obj  : {ep.target_obj}")
print(f"Steps       : {len(ep.steps)}")
print(f"Success     : {ep.success}")
print(f"Action range: [{ep.steps[0].action.min():.3f}, {ep.steps[0].action.max():.3f}]")

os.makedirs("images", exist_ok=True)
out = f"images/episode_{ep_idx:04d}.mp4"

h, w = ep.steps[0].rgb.shape[:2]
fps = 25
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out, fourcc, fps, (w, h))

for i, step in enumerate(ep.steps):
    frame = step.rgb.copy()
    label = f"{ep.instruction}  |  step {i}/{len(ep.steps)}"
    cv2.putText(frame, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

writer.release()
print(f"\nSaved {out}  ({len(ep.steps)} frames @ {fps} fps)")
