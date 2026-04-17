# G1Nav — CLAUDE.md

## Python Environment

Do NOT use the `.venv` inside this repo. The working Python environment is the GR00T venv:

```
/home/ludo-us/work/Isaac-GR00T/.venv/bin/python
```

Run scripts with:
```bash
/home/ludo-us/work/Isaac-GR00T/.venv/bin/python <script.py>
```

Or activate it:
```bash
source /home/ludo-us/work/Isaac-GR00T/.venv/bin/activate
```

## Installed Packages (in GR00T venv)

- Python 3.11.15
- torch 2.7.1+cu128
- mujoco 3.7.0
- gr00t (Isaac GR00T N1.6, from /home/ludo-us/work/Isaac-GR00T)
- git-lfs binary at ~/.local/bin/git-lfs (installed manually, no sudo needed)

## G1 Robot Model

Using GR00T's own G1 model (not Menagerie):
- XML: `Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/robot_model/model_data/g1/scene_29dof.xml`
- Meshes pulled via git-lfs (65 binary STL files)
- Already has `head_camera` on torso_link — used as egocentric camera

## GR00T Repo

- Location: `/home/ludo-us/work/Isaac-GR00T`
- Version: N1.6 (checked out at tag `n1.6-release`)
- Checkpoint to use: `nvidia/GR00T-N1.6-3B` (downloads from HuggingFace on first run)

## MuJoCo Rendering

Always set `MUJOCO_GL=egl` for headless rendering (no display attached):
```bash
MUJOCO_GL=egl /home/ludo-us/work/Isaac-GR00T/.venv/bin/python <script.py>
```

## GPU

- RTX 5080 Laptop GPU, 16 GB VRAM
- Enough for inference (~12 GB) but NOT for full fine-tuning (48 GB needed)
- Strategy: freeze VLM (Cosmos-Reason-2B), only fine-tune the Action Head (DiT)
