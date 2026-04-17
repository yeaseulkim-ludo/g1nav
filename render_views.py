"""Render ego + third-person views and save as PNGs."""
import os
import sys
sys.path.insert(0, "/home/ludo-us/work/g1nav")

from PIL import Image
from code.env.arena import G1NavArena

os.makedirs("images", exist_ok=True)

arena = G1NavArena()
arena.reset()

Image.fromarray(arena.render_ego()).save("images/ego_view.png")
Image.fromarray(arena.render_third_person()).save("images/third_person_view.png")
print("Saved images/ego_view.png and images/third_person_view.png")
arena.close()
