"""Render ego + third-person views and save as PNGs."""
import sys
sys.path.insert(0, "/home/ludo-us/work/g1nav")

from PIL import Image
import numpy as np
from code.env.arena import G1NavArena

arena = G1NavArena()
arena.reset()

ego = arena.render_ego()
tp  = arena.render_third_person()

Image.fromarray(ego).save("ego_view.png")
Image.fromarray(tp).save("third_person_view.png")
print("Saved ego_view.png and third_person_view.png")
arena.close()
