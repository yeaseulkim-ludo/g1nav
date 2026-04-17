"""
MuJoCo arena for G1Nav.

The arena is a 4m x 4m flat room. A handful of colored objects are scattered
on the floor. The G1 robot starts near the center facing +X.

Coordinate system (MuJoCo default):
  X = forward, Y = left, Z = up

Objects:
  - red ball       (sphere)
  - yellow square  (box)
  - blue cylinder  (cylinder)
  - green cone     (tapered cylinder)
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Tuple

import mujoco
import numpy as np


# Use the WBC's sim2mujoco XML — it has the correct physics parameters (low joint
# damping) that match the WBC policy's training environment.
G1_MODEL_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../..",
        "Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl"
        "/gr00t_wbc/sim2mujoco/resources/robots/g1",
    )
)
G1_SCENE_XML = os.path.join(G1_MODEL_DIR, "g1_gear_wbc.xml")

# Lower-body actuated joint names (15 joints total)
LOWER_BODY_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


@dataclass
class ObjectConfig:
    name: str
    shape: str                               # "sphere", "box", "cylinder"
    rgba: Tuple[float, float, float, float]
    size: List[float]                        # MuJoCo geom size params
    pos: Tuple[float, float, float]


def default_objects() -> List[ObjectConfig]:
    return [
        ObjectConfig("red_ball",      "sphere",   (0.85, 0.15, 0.15, 1), [0.08],           (2.0,  0.5, 0.08)),
        ObjectConfig("yellow_square", "box",      (0.95, 0.80, 0.10, 1), [0.10, 0.10, 0.10],(1.5, -0.8, 0.10)),
        ObjectConfig("blue_cylinder", "cylinder", (0.10, 0.25, 0.85, 1), [0.07, 0.15],     (2.5, -0.3, 0.15)),
        ObjectConfig("green_cone",    "cylinder", (0.10, 0.75, 0.20, 1), [0.01, 0.18],     (1.0,  1.0, 0.18)),
    ]


def build_arena_xml(objects: List[ObjectConfig] | None = None) -> str:
    """
    Take the existing G1 scene XML and inject arena walls + colored objects.
    Returns the modified XML string with all paths made absolute.
    """
    if objects is None:
        objects = default_objects()

    tree = ET.parse(G1_SCENE_XML)
    root = tree.getroot()

    # Make mesh/texture paths absolute so the temp file can live anywhere
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", os.path.join(G1_MODEL_DIR, "meshes"))
    compiler.set("texturedir", G1_MODEL_DIR)

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # Arena walls (thin boxes, 4 m half-width, 1 m tall)
    walls = [
        ("wall_north", "4 0 0.5",  "0.05 4 0.5"),
        ("wall_south", "-4 0 0.5", "0.05 4 0.5"),
        ("wall_east",  "0 4 0.5",  "4 0.05 0.5"),
        ("wall_west",  "0 -4 0.5", "4 0.05 0.5"),
    ]
    for wname, wpos, wsize in walls:
        ET.SubElement(worldbody, "geom",
                      name=wname, type="box",
                      pos=wpos, size=wsize,
                      rgba="0.7 0.7 0.7 1",
                      contype="1", conaffinity="1")

    # Colored objects
    for obj in objects:
        size_str = " ".join(str(s) for s in obj.size)
        rgba_str = " ".join(str(v) for v in obj.rgba)
        pos_str  = " ".join(str(v) for v in obj.pos)
        body = ET.SubElement(worldbody, "body", name=obj.name, pos=pos_str)
        ET.SubElement(body, "geom",
                      name=obj.name + "_geom",
                      type=obj.shape, size=size_str, rgba=rgba_str,
                      contype="1", conaffinity="1")

    # Egocentric camera on torso_link (head position)
    torso = root.find(".//body[@name='torso_link']")
    if torso is not None:
        ET.SubElement(torso, "camera",
                      name="head_camera",
                      pos="0.1 0 0.3",
                      xyaxes="0 -1 0 0 0 1",
                      fovy="90")

    # Third-person camera — behind and elevated, looking forward-down toward objects
    ET.SubElement(worldbody, "camera",
                  name="third_person",
                  pos="-2 0 3",
                  xyaxes="0 -1 0 0.53 0 0.85",
                  fovy="60")

    return ET.tostring(root, encoding="unicode")


class G1NavArena:
    """
    Wraps a MuJoCo model+data for the G1Nav task.

    Usage:
        arena = G1NavArena()
        obs = arena.reset()
        for _ in range(500):
            action = np.zeros(15)   # lower-body joint targets
            obs, done = arena.step(action)
    """

    IMG_W, IMG_H = 640, 480
    TP_W,  TP_H  = 320, 240
    EGO_W, EGO_H = 224, 224

    def __init__(self, objects: List[ObjectConfig] | None = None, seed: int = 0):
        self._rng     = np.random.default_rng(seed)
        self._objects = objects or default_objects()

        xml_str = build_arena_xml(self._objects)

        # Write to temp file next to the G1 model so relative mesh paths resolve
        self._tmpfile = tempfile.NamedTemporaryFile(
            suffix=".xml", dir=G1_MODEL_DIR, delete=False, mode="w"
        )
        self._tmpfile.write(xml_str)
        self._tmpfile.flush()

        self.model = mujoco.MjModel.from_xml_path(self._tmpfile.name)
        self.data  = mujoco.MjData(self.model)

        # Resolve lower-body joint qpos/qvel addresses
        self._qpos_idx = [
            self.model.joint(name).qposadr[0] for name in LOWER_BODY_JOINTS
        ]
        self._qvel_idx = [
            self.model.joint(name).dofadr[0] for name in LOWER_BODY_JOINTS
        ]

        # Camera IDs
        self._ego_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera"
        )
        self._tp_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "third_person"
        )

        self._renderer     = mujoco.Renderer(self.model, height=self.IMG_H,  width=self.IMG_W)
        self._tp_renderer  = mujoco.Renderer(self.model, height=self.TP_H,   width=self.TP_W)
        self._ego_renderer = mujoco.Renderer(self.model, height=self.EGO_H,  width=self.EGO_W)

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self, init_joint_angles: np.ndarray | None = None) -> dict:
        """Reset simulation. Optionally set initial lower-body joint angles."""
        mujoco.mj_resetData(self.model, self.data)
        if init_joint_angles is not None:
            assert init_joint_angles.shape == (15,)
            for i, idx in enumerate(self._qpos_idx):
                self.data.qpos[idx] = init_joint_angles[i]
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step_torque(self, torques: np.ndarray) -> Tuple[dict, bool]:
        """Advance one sim step (5 ms) applying raw torques to data.ctrl[0:15]."""
        assert torques.shape == (15,)
        self.data.ctrl[:15] = torques
        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), False

    def render_ego(self) -> np.ndarray:
        """Egocentric RGB image (224x224) uint8 — for training."""
        self._ego_renderer.update_scene(self.data, camera=self._ego_cam_id)
        return self._ego_renderer.render().copy()

    def render_depth(self) -> np.ndarray:
        """Egocentric depth map (H, W) float32 in metres."""
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=self._ego_cam_id)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return depth

    def render_third_person(self) -> np.ndarray:
        """Third-person RGB image (320x240) uint8 — for visualization."""
        self._tp_renderer.update_scene(self.data, camera=self._tp_cam_id)
        return self._tp_renderer.render().copy()

    def get_joint_angles(self) -> np.ndarray:
        """Current lower-body joint angles (15,) radians."""
        return np.array([self.data.qpos[i] for i in self._qpos_idx])

    def get_joint_velocities(self) -> np.ndarray:
        """Current lower-body joint velocities (15,) rad/s."""
        return np.array([self.data.qvel[i] for i in self._qvel_idx])

    def object_positions(self) -> dict:
        """Returns {name: (x, y, z)} for every arena object."""
        return {
            obj.name: tuple(self.data.body(obj.name).xpos.copy())
            for obj in self._objects
        }

    def close(self):
        self._renderer.close()
        self._tp_renderer.close()
        self._ego_renderer.close()
        os.unlink(self._tmpfile.name)

    # ── internals ─────────────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        return {
            "joints":     self.get_joint_angles(),
            "joint_vels": self.get_joint_velocities(),
        }
