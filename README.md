# Pick and Place with PPO

A Franka Panda arm learns to pick up a red cube and place it on a yellow target pad using reinforcement learning (PPO from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)) in a [PyBullet](https://pybullet.org/) simulation.

The task comes in two versions:

| Env | File | What the policy sees | Status |
|---|---|---|---|
| `state` | `robot_env.py` | Exact joint angles, gripper position, cube position and target position (a 23-dim vector) | Trained successfully (`runs/ppo_pick_place`) |
| `visual` | `robot_env_visual.py` | A 128×128 RGB camera image plus proprioception (the robot's own joint/gripper state) | Not solved yet: the policy doesn't get past the reach stage |

The long-term goal is to do pick and place from camera images only, and later stack several cubes.

---

## Setup

Tested with Python in a conda env (`rl`), using gymnasium 1.3, pybullet 3.2.7, torch 2.11 (CUDA 12.8), numpy 2.x and tensorboard.

```bash
conda create -n rl python=3.11
conda activate rl
pip install gymnasium pybullet stable-baselines3 tensorboard numpy
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU build
```

PPO uses the GPU automatically when `torch.cuda.is_available()`. Physics runs on the CPU, one PyBullet process per parallel env.

---

## Usage

### Train

```bash
python train.py --env state                          # default: 2M steps
python train.py --env visual --steps 500000
```

Outputs are written to `runs/ppo_pick_place_<env>/`:

```
runs/ppo_pick_place_<env>/
├── PPO_1/                 # tensorboard logs (a new PPO_n per run)
├── best/best_model.zip    # best model according to EvalCallback
├── checkpoints/           # model + VecNormalize stats every 100k steps
├── final_model.zip        # written when training finishes
└── vecnormalize.pkl       # observation normalization stats (needed for play)
```

> **Note:** training the same `--env` again writes into the same folder. `final_model.zip`, `vecnormalize.pkl` and `best/` are overwritten, while tensorboard gets a new `PPO_n`. Rename the old folder first if you want to keep it.

Watch progress live:

```bash
tensorboard --logdir runs/
```

Useful curves to watch are `rollout/ep_rew_mean`, `eval/mean_reward`, `eval/success_rate` and `train/std` (the policy's action noise; if this collapses towards 0, the policy has stopped exploring).

#### Long runs

Visual training is memory-heavy: SB3 stores image observations as float32 in the rollout buffer, which comes to about 1.6 GB for 8 envs × 1024 steps. If the desktop runs low on memory, `systemd-oomd` may kill the whole VS Code process group, and a run started in VS Code's terminal dies with it. For overnight runs, start training in `tmux` instead:

```bash
tmux new -s train
python train.py --env visual
# detach: Ctrl+B then D    reattach: tmux attach -t train
```

If memory is still tight, reduce `N_ENVS` or `n_steps` in `train.py`.

### Play / evaluate

```bash
python play.py --env state                       # opens the PyBullet GUI, 5 episodes
python play.py --env visual --episodes 10
python play.py --env state --no-render           # headless, prints results only
python play.py --env state --model runs/ppo_pick_place/checkpoints/ppo_1000000_steps.zip \
               --vecnormalize runs/ppo_pick_place/checkpoints/ppo_vecnormalize_1000000_steps.pkl
```

By default this loads `best/best_model.zip` together with `vecnormalize.pkl` from the env's run folder, and prints success, step count and total reward for each episode.

---

## The task

```
          y
          ▲
          │        ● cube (random spawn, 0.33–0.65 m from base)
   base ■─┼──────────────■ target pad (0.5, 0, 0)
          │
          └────────────► x
```

- **Robot:** a Franka Panda (`franka_panda/panda.urdf` from `pybullet_data`) with a fixed base at the origin. It has 7 revolute arm joints (0–6) and 2 prismatic finger joints (9, 10). The end effector is link 11.
- **Cube:** 7 cm across (half extent 0.035), 0.5 kg. The fingers open to 8 cm, so the cube must stay smaller than that. On each reset it spawns at a random point in a half ring in front of the robot, 0.33–0.65 m from the base (the comfortable top-down reach) and at least 15 cm from the target.
- **Target:** a 20×20 cm yellow pad at (0.5, 0, 0). It is visual only, with no collision.
- **Episode:** at most 400 steps. It ends early on success.

### Action space: `Box(-1, 1, shape=(4,))`

| Index | Meaning |
|---|---|
| 0–2 | Move the gripper in x/y/z by up to `action × 3 cm` per step |
| 3 | Gripper: `> 0` closes, `≤ 0` opens |

The env turns the Cartesian move into joint angles with `p.calculateInverseKinematics`. The gripper always points straight down, and IK uses joint limits plus the home pose as rest pose so the arm stays in a sensible configuration. The target is clipped to the workspace `x∈[0, 0.75], y∈[-0.7, 0.7], z∈[0.02, 0.3]`. Each step then runs 30 physics substeps at 240 Hz (one action = 0.125 s of sim time).

### Observation space

**State env** (`Box`, 23 floats): 9 joint positions, gripper xyz, cube xyz, cube − gripper, target − cube, `is_grasped`, `has_lifted`.

**Visual env** (`Dict`):

| Key | Shape | Contents |
|---|---|---|
| `camera_image` | `(128, 128, 3)` uint8 | RGB from a fixed camera behind the target looking back at the robot (yaw 90°, pitch −50°) |
| `proprio` | `(14,)` float32 | 7 arm joint angles, 2 finger positions, gripper xyz, `is_grasped`, `has_lifted` |

### Grasp detection

The cube counts as **grasped** when *both* fingers touch it and each pushes with at least 5 N of normal force. This uses `p.getContactPoints` filtered per finger link, and sums contact field 9 (`normalForce`). Requiring force from both fingers stops "touching the cube with one finger" from counting as a grasp.

### Success

An episode succeeds when the cube has been lifted at least once (`has_lifted`), and is now released (not grasped), resting on the ground and within 7 cm (xy) of the target centre.

---

## Reward design

The reward is **staged**: exactly one stage is active per step, picked by checking the furthest stage first. Each stage has its own constant base value, and inside a stage the reward is shaped by distance so the policy can tell "closer" from "further":

| Stage | Condition | Reward |
|---|---|---|
| 5. Success | `is_success` | `+100` (episode ends) |
| 4. Place | lifted and cube within 7 cm of target | `-1 - 8 × (cube height above the ground)` |
| 3. Carry | lifted and cube above the lift height (12 cm) | `-2 - distance(cube, target)` |
| 2. Lift | grasped | `-3 - 15 × (lift height - cube height)` |
| 1. Reach | otherwise | `-4 - distance(cube, gripper)` |

Why it's shaped like this:

- **Each stage's band of values doesn't overlap the next one.** Every step in a later stage is better than any step in an earlier stage, so progress is always rewarded and the policy can't do better by staying in an easy stage.
- **Every non-terminal reward is negative.** Dawdling costs reward on every step, so finishing quickly (and collecting the +100) is the best strategy.
- **`has_lifted` is sticky.** Once the cube has been lifted, the episode stays in the carry/place stages, so lowering the cube onto the target isn't punished as "un-lifting" it. The flag resets only if the cube ends up back on the ground away from the target without being held.

---

## Training setup (`train.py`)

| Setting | Value | Why |
|---|---|---|
| Parallel envs | 8 × `SubprocVecEnv` | Each env runs its own PyBullet instance in its own process |
| `n_steps` / `batch_size` / `n_epochs` | 1024 / 256 / 10 | 8192 samples per update |
| `learning_rate`, `gamma`, `gae_lambda`, `clip_range` | 3e-4, 0.99, 0.95, 0.2 | Standard PPO defaults |
| Network | 256×256 for both policy and value | |
| Policy | `MlpPolicy` (state) / `MultiInputPolicy` (visual; CNN for the image and MLP for proprio) | |
| `ent_coef` | 0.0 (state) / 0.01 (visual) | An entropy bonus to stop the visual policy's action noise from collapsing |
| `VecNormalize` | Normalizes observations and rewards; for visual envs only `proprio` is normalized, never the image | |
| Eval | Every 50k steps, 5 deterministic episodes; saves `best/` | |
| Checkpoints | Every 100k steps, with VecNormalize stats | |

For visual envs, SB3 wraps the training env as `VecTransposeImage(VecNormalize(...))` (HWC → CHW for PyTorch). The eval env and `play.py` apply the same wrappers manually so the model sees identical inputs.

---

## Repository layout

| File | Purpose |
|---|---|
| `robot_env.py` | State-based gymnasium env (Panda on a table) |
| `robot_env_visual.py` | Camera-based gymnasium env (Panda on the ground plane) |
| `train.py` | PPO training; `--env {state, visual}`, `--steps N` |
| `play.py` | Load a trained model and run episodes with the GUI |
| `handle_gripper.py` | Early experiment with gripper control in PyBullet |
| `runs/ppo_pick_place/` | Trained state-env run (tensorboard logs, checkpoints, best/final model, VecNormalize stats) |

## Possible next steps

- Stacking: extend the env to 2–3 cubes.
- Domain randomization (lighting, colours, camera pose) for the visual env.
