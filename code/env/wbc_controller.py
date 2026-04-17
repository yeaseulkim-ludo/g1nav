"""
WBC locomotion controller for data collection.

Wraps the GR00T-WholeBodyControl ONNX policies.
Replicates the logic from gr00t_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py
without ROS/viewer dependencies.
"""

import collections
import os

import mujoco
import numpy as np
import onnxruntime as ort
import torch

_WBC_RESOURCE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../..",
        "Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl"
        "/gr00t_wbc/sim2mujoco/resources/robots/g1",
    )
)

BALANCE_ONNX = os.path.join(_WBC_RESOURCE_DIR, "policy/GR00T-WholeBodyControl-Balance.onnx")
WALK_ONNX    = os.path.join(_WBC_RESOURCE_DIR, "policy/GR00T-WholeBodyControl-Walk.onnx")

# From g1_gear_wbc.yaml (sim2mujoco version)
CFG = {
    "kps":            np.array([150, 150, 150, 200, 40, 40,
                                150, 150, 150, 200, 40, 40,
                                250, 250, 250], dtype=np.float32),
    "kds":            np.array([2, 2, 2, 4, 2, 2,
                                2, 2, 2, 4, 2, 2,
                                5, 5, 5], dtype=np.float32),
    "default_angles": np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                 -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                  0.0, 0.0, 0.0], dtype=np.float32),
    "action_scale":   0.25,
    "ang_vel_scale":  0.5,
    "dof_pos_scale":  1.0,
    "dof_vel_scale":  0.05,
    "cmd_scale":      np.array([2.0, 2.0, 0.5], dtype=np.float32),
    "height_cmd":     0.74,
    "obs_history_len": 6,
    "num_obs":        516,   # 86 * 6
    "num_actions":    15,
    "control_decimation": 4,  # policy runs every 4 sim steps (4 * 5ms = 20ms)
}

SINGLE_OBS_DIM = 86


def _quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q=[w,x,y,z].

    Mirrors the reference implementation in run_mujoco_gear_wbc.py which uses
    conjugate q_conj=[w,-x,-y,-z] then applies the standard sandwich formula.
    """
    w, x, y, z = q
    # Expanding with q_conj = [w, -x, -y, -z]:
    return np.array([
        v[0] * (w*w + x*x - y*y - z*z) + v[1] * 2*(x*y + w*z) + v[2] * 2*(x*z - w*y),
        v[0] * 2*(x*y - w*z) + v[1] * (w*w - x*x + y*y - z*z) + v[2] * 2*(y*z + w*x),
        v[0] * 2*(x*z + w*y) + v[1] * 2*(y*z - w*x) + v[2] * (w*w - x*x - y*y + z*z),
    ], dtype=np.float32)


def _gravity_in_body(quat_wxyz):
    return _quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0]))


def _load_onnx(path):
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    def run(x: np.ndarray) -> np.ndarray:
        return sess.run(None, {input_name: x})[0]

    return run


class WBCController:
    """
    Wraps Balance + Walk ONNX policies.

    Call reset() at episode start, then step() every sim step.
    The policy only re-infers every control_decimation sim steps.
    """

    def __init__(self, mj_model: mujoco.MjModel):
        self._balance_policy = _load_onnx(BALANCE_ONNX)
        self._walk_policy    = _load_onnx(WALK_ONNX)

        # WBC PD gains and policy cadence are tuned for 5 ms sim steps
        mj_model.opt.timestep = 0.005

        self._n_joints = mj_model.nq - 7  # total DOF excluding freejoint
        self._torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

        self._action = np.zeros(CFG["num_actions"], dtype=np.float32)
        self._target_q = CFG["default_angles"].copy()
        self._obs_history = collections.deque(
            [np.zeros(SINGLE_OBS_DIM, dtype=np.float32)] * CFG["obs_history_len"],
            maxlen=CFG["obs_history_len"],
        )
        self._obs = np.zeros(CFG["num_obs"], dtype=np.float32)
        self._counter = 0

    def reset(self):
        self._action[:] = 0.0
        self._target_q = CFG["default_angles"].copy()
        self._obs_history = collections.deque(
            [np.zeros(SINGLE_OBS_DIM, dtype=np.float32)] * CFG["obs_history_len"],
            maxlen=CFG["obs_history_len"],
        )
        self._obs[:] = 0.0
        self._counter = 0

    def step(self, mj_data: mujoco.MjData, command_vxvywz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run one sim step of the WBC.

        Args:
            mj_data: current MuJoCo data (already stepped or just reset)
            command_vxvywz: [vx, vy, wz] velocity command in robot frame

        Returns:
            target_q: (15,) target joint positions — use as action label
            torques:  (15,) torques to set in data.ctrl[0:15]
        """
        if self._counter % CFG["control_decimation"] == 0:
            single_obs = self._compute_obs(mj_data, command_vxvywz)
            self._obs_history.append(single_obs)
            for i, h in enumerate(self._obs_history):
                self._obs[i * SINGLE_OBS_DIM:(i + 1) * SINGLE_OBS_DIM] = h

            obs_input = self._obs[np.newaxis].astype(np.float32)
            walking = np.linalg.norm(command_vxvywz) > 0.05
            policy = self._walk_policy if walking else self._balance_policy
            self._action = policy(obs_input)[0].astype(np.float32)
            self._target_q = self._action * CFG["action_scale"] + CFG["default_angles"]

        torques = self._pd_torques(mj_data)
        self._counter += 1
        return self._target_q.copy(), torques

    # ── internals ─────────────────────────────────────────────────────────────

    def _compute_obs(self, d: mujoco.MjData, cmd: np.ndarray) -> np.ndarray:
        n = self._n_joints
        qj   = d.qpos[7:7 + n].copy()
        dqj  = d.qvel[6:6 + n].copy()
        quat  = d.qpos[3:7].copy()   # [w, x, y, z]
        omega = d.qvel[3:6].copy()

        defaults = np.zeros(n, dtype=np.float32)
        L = min(len(CFG["default_angles"]), n)
        defaults[:L] = CFG["default_angles"][:L]

        obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
        obs[0:3] = cmd[:3] * CFG["cmd_scale"]
        obs[3]   = CFG["height_cmd"]
        obs[4:7] = 0.0                            # rpy cmd (keep level)
        obs[7:10]  = omega * CFG["ang_vel_scale"]
        obs[10:13] = _gravity_in_body(quat)
        obs[13:13 + n]      = (qj - defaults) * CFG["dof_pos_scale"]
        obs[13 + n:13 + 2*n] = dqj * CFG["dof_vel_scale"]
        obs[13 + 2*n:13 + 2*n + 15] = self._action
        return obs

    def _pd_torques(self, d: mujoco.MjData) -> np.ndarray:
        current_q  = d.qpos[7:7 + CFG["num_actions"]].copy()
        current_dq = d.qvel[6:6 + CFG["num_actions"]].copy()
        return (self._target_q - current_q) * CFG["kps"] + (0.0 - current_dq) * CFG["kds"]

    @property
    def default_angles(self) -> np.ndarray:
        return CFG["default_angles"].copy()
