"""
Episode data collector for G1Nav.

Each episode: robot steers toward a target object using the WBC locomotion policy.
Records (rgb, depth, joints, joint_vels, instruction) → action (lower-body joint targets).
"""

import math
import os
import pickle
from dataclasses import dataclass, field
from typing import List

import numpy as np

from code.env.arena import G1NavArena, ObjectConfig, default_objects
from code.env.wbc_controller import CFG, WBCController

DATASET_DIR = os.path.join(os.path.dirname(__file__), "../../dataset/episodes")

INSTRUCTION_TEMPLATES = {
    "red_ball":      ["walk to the red ball", "go to the red ball", "navigate to the red ball"],
    "yellow_square": ["walk to the yellow square", "go to the yellow square", "approach the yellow cube"],
    "blue_cylinder": ["walk to the blue cylinder", "go to the blue cylinder", "move toward the blue cylinder"],
    "green_cone":    ["walk to the green cone", "go to the green cone", "approach the green cone"],
}

# Episode parameters
WARMUP_STEPS  = 200       # zero-command stabilization before walking (~1 s)
EPISODE_STEPS = 2000      # max sim steps after warmup (~10 s at 5 ms/step)
SUCCESS_DIST  = 0.4       # metres — episode ends when robot is this close to target
VX_CMD        = 0.4       # forward speed command (m/s)
WZ_GAIN       = 1.5       # proportional gain for yaw command from heading error
MAX_WZ        = 1.0       # rad/s cap


@dataclass
class Step:
    rgb_tp:     np.ndarray   # (240, 320, 3) uint8  third-person view for visualization
    joints:     np.ndarray   # (15,) float32
    joint_vels: np.ndarray   # (15,) float32
    action:     np.ndarray   # (15,) float32  ← WBC joint targets


@dataclass
class Episode:
    instruction: str
    target_obj:  str
    steps:       List[Step] = field(default_factory=list)
    success:     bool = False


def _heading_error(mj_data, target_pos: np.ndarray) -> float:
    """Signed angle from robot's forward direction to target (radians)."""
    base_pos = mj_data.qpos[:3]
    dx, dy = target_pos[0] - base_pos[0], target_pos[1] - base_pos[1]
    target_yaw = math.atan2(dy, dx)

    # Robot yaw from quaternion [w, x, y, z]
    qw, qx, qy, qz = mj_data.qpos[3:7]
    robot_yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    err = target_yaw - robot_yaw
    # Wrap to [-pi, pi]
    return (err + math.pi) % (2 * math.pi) - math.pi


def _dist_to_target(mj_data, target_pos: np.ndarray) -> float:
    base_xy = mj_data.qpos[:2]
    return float(np.linalg.norm(base_xy - target_pos[:2]))


def collect_episode(
    arena: G1NavArena,
    wbc: WBCController,
    target_obj: str,
    instruction: str,
    rng: np.random.Generator,
) -> Episode:
    obs = arena.reset(init_joint_angles=wbc.default_angles)
    wbc.reset()

    target_pos = np.array(arena.object_positions()[target_obj])
    episode = Episode(instruction=instruction, target_obj=target_obj)

    # Stabilization warmup — zero command, don't record
    zero_cmd = np.zeros(3, dtype=np.float32)
    for _ in range(WARMUP_STEPS):
        _, torques = wbc.step(arena.data, zero_cmd)
        arena.step_torque(torques)
    obs = arena._get_obs()

    for _ in range(EPISODE_STEPS):
        dist = _dist_to_target(arena.data, target_pos)
        if dist < SUCCESS_DIST:
            episode.success = True
            break

        # Steering: proportional heading controller → [vx, vy, wz]
        herr = _heading_error(arena.data, target_pos)
        wz   = float(np.clip(WZ_GAIN * herr, -MAX_WZ, MAX_WZ))
        vx   = VX_CMD if abs(herr) < math.pi / 3 else 0.0   # slow turn first
        command = np.array([vx, 0.0, wz], dtype=np.float32)

        target_q, torques = wbc.step(arena.data, command)
        arena.step_torque(torques)

        episode.steps.append(Step(
            rgb_tp=arena.render_third_person(),
            joints=obs["joints"].astype(np.float32),
            joint_vels=obs["joint_vels"].astype(np.float32),
            action=target_q,
        ))
        obs = arena._get_obs()

    return episode


def collect_dataset(n_episodes: int = 100, seed: int = 42):
    os.makedirs(DATASET_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    arena = G1NavArena(seed=seed)
    wbc   = WBCController(arena.model)

    objects = default_objects()
    obj_names = [o.name for o in objects]

    saved = 0
    for ep_idx in range(n_episodes):
        target_name = rng.choice(obj_names)
        instruction  = rng.choice(INSTRUCTION_TEMPLATES[target_name])

        episode = collect_episode(arena, wbc, target_name, instruction, rng)

        if len(episode.steps) == 0:
            continue

        path = os.path.join(DATASET_DIR, f"ep_{ep_idx:04d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(episode, f)

        saved += 1
        status = "OK" if episode.success else "timeout"
        print(f"[{ep_idx:04d}] {instruction:<45} steps={len(episode.steps):4d}  {status}")

    print(f"\nSaved {saved}/{n_episodes} episodes to {DATASET_DIR}")
    arena.close()
