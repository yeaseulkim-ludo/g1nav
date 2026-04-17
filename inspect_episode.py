"""Visualize a collected episode: saves ego-view GIF + prints stats."""
import os
import pickle
import sys
sys.path.insert(0, "/home/ludo-us/work/g1nav")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EPISODES_DIR = "dataset/episodes"

# Pick episode to inspect (default: first one)
ep_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
path = os.path.join(EPISODES_DIR, f"ep_{ep_idx:04d}.pkl")

with open(path, "rb") as f:
    ep = pickle.load(f)

print(f"Instruction : {ep.instruction}")
print(f"Target obj  : {ep.target_obj}")
print(f"Steps       : {len(ep.steps)}")
print(f"Success     : {ep.success}")
print(f"Action range: [{ep.steps[0].action.min():.3f}, {ep.steps[0].action.max():.3f}]")
print(f"Joint names : 15 lower-body joints")

# Sample every 10th frame to keep GIF small
frames = ep.steps[::10]

def add_text(img_array, text):
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), text, fill=(255, 255, 255))
    return np.array(img)

gif_frames = []
for i, step in enumerate(frames):
    label = f"{ep.instruction}  |  step {i*10}/{len(ep.steps)}"
    gif_frames.append(Image.fromarray(add_text(step.rgb, label)))

os.makedirs("images", exist_ok=True)
out = f"images/episode_{ep_idx:04d}.gif"
gif_frames[0].save(
    out,
    save_all=True,
    append_images=gif_frames[1:],
    duration=80,   # ms per frame
    loop=0,
)
print(f"\nSaved {out}  ({len(gif_frames)} frames)")
