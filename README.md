# Provably Efficient Policy-Reward Co-Pretraining for Adversarial Imitation  Learning

Chinese documentation: `README.zh-CN.md`

This repository provides the author's official implementation of ICML 2026 paper "Provably Efficient Policy-Reward Co-Pretraining for Adversarial Imitation Learning".

## Project layout

```text
.
|-- agent/                 # BC, AIL, DAC, FOIL agents
|-- conf/                  # Hydra config: env, agent, method
|-- dataset/               # Expert dataset loader and replay memory
|-- dmc2gym/               # Local dm_control -> gym wrapper
|-- envs/                  # Custom MuJoCo env definitions
|-- experts/               # Place expert .pkl data here (empty by default)
|-- module/                # Networks (actor, critic, discriminator)
|-- pretrain/              # Suggested folder for BC checkpoints
|-- scripts/               # Batch scripts for pretrain and training
|-- pretrain.py            # Offline BC pretraining entry
|-- train.py               # Online AIL/DAC training entry
`-- off2on.py              # Off-to-on finetuning entry (FOIL/AIL)
```

## Requirements

- Python `3.7.16`
- MuJoCo + dm_control compatible environment
- Optional CUDA GPU (device is auto-selected in code)
- Dependencies in `requirements.txt`

Install:

```bash
pip install -r requirements.txt
pip install -e ./dmc2gym
```

## Prepare expert data

1. Download expert trajectories from:

   `https://osf.io/ceh6q/?view_only=41aa006b2fd149f7815294034a4792b0`

2. Put `.pkl` files under `experts/`.

Expected filenames (from `conf/env/*.yaml`):

- `acrobot_swingup_100.pkl`
- `cartpole_swingup_100.pkl`
- `cheetah_run_100.pkl`
- `finger_spin_100.pkl`
- `hopper_hop_100.pkl`
- `hopper_stand_100.pkl`
- `humanoid_stand_100.pkl`
- `quadruped_run_100.pkl`
- `quadruped_walk_100.pkl`
- `walker_run_100.pkl`
- `walker_stand_100.pkl`
- `walker_walk_100.pkl`

## Configure checkpoint output

Before running pretraining, update `conf/config.yaml`:

```yaml
model_folder: /absolute/path/to/CoPT-AIL/pretrain/
```

Use an **absolute path** and keep the trailing `/`.

BC checkpoints are saved as:

`<model_folder><env.name>_10_bc`

For example:

`/abs/path/CoPT-AIL/pretrain/dmc_finger_spin_10_bc`

## Quick start

### Step 1: BC pretraining (single task)

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py env=finger_spin agent=bc expert.demos=50 env.learn_steps=1e5 seed=2
```

### Step 2: Off-to-on CoPT-AIL / FOIL (single task)

```bash
CUDA_VISIBLE_DEVICES=0 python off2on.py env=finger_spin agent=foil expert.demos=50 method=il seed=2 agent.bc_transit=false project.name=run_off2on
```

`off2on.py` automatically loads the BC checkpoint from `model_folder`.

## Batch scripts

- `scripts/pretrain_all.sh`: BC pretraining for 8 DMC tasks.
- `scripts/run_all_off2on.sh`: FOIL off-to-on runs across tasks.

These scripts assume:

- Conda env named `ail`
- Multi-GPU setup (`CUDA_VISIBLE_DEVICES=0/1`)

Edit GPU ids, seeds, and demo counts before running.

## Hydra usage notes

- Environment configs are selected by short name, for example `env=finger_spin`.
- Always set `agent=bc|ail|foil|dac` in command line overrides.
- Useful overrides:
  - `expert.demos=<n>`
  - `env.learn_steps=<n>`
  - `method.lambda_gp=<float>`
  - `seed=<int>`
  - `wandb=true`


### Bibtex

If you find this code is helpful, please cite our paper in the following format.

```
@inproceedings{xu2026provably,
title={Provably Efficient Policy-Reward Co-Pretraining for Adversarial Imitation Learning},
author={Tian Xu and Zexuan Chen and Zhilong Zhang and Yi-Chen Li and Yang Yu},
booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
year={2026},
}
```

