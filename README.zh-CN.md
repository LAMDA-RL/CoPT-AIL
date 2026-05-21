# Provably Efficient Policy-Reward Co-Pretraining for Adversarial Imitation  Learning

英文文档：`README.md`

本仓库提供 ICML 2026 paper "Provably Efficient Policy-Reward Co-Pretraining for Adversarial Imitation Learning" 的作者官方实现。

## 项目结构

```text
.
|-- agent/                 # BC, AIL, DAC, FOIL agents
|-- conf/                  # Hydra 配置：env, agent, method
|-- dataset/               # Expert dataset loader 与 replay memory
|-- dmc2gym/               # 本地 dm_control -> gym wrapper
|-- envs/                  # 自定义 MuJoCo env definitions
|-- experts/               # 放置 expert .pkl data（默认空目录）
|-- module/                # Networks (actor, critic, discriminator)
|-- pretrain/              # 建议的 BC checkpoints 输出目录
|-- scripts/               # Pretrain 与 training 批量脚本
|-- pretrain.py            # Offline BC pretraining 入口
|-- train.py               # Online AIL/DAC training 入口
`-- off2on.py              # Off-to-on finetuning 入口 (FOIL/AIL)
```

## 环境要求

- Python `3.7.16`
- MuJoCo + dm_control 兼容环境
- Optional CUDA GPU（代码会自动选择 device）
- `requirements.txt` 中的依赖

安装：

```bash
pip install -r requirements.txt
pip install -e ./dmc2gym
```

## 准备 expert data

1. 从以下地址下载 expert trajectories：

   `https://osf.io/ceh6q/?view_only=41aa006b2fd149f7815294034a4792b0`

2. 将 `.pkl` 文件放到 `experts/` 目录下。

预期文件名（来自 `conf/env/*.yaml`）：

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

## 配置 checkpoint 输出

运行 pretraining 前，更新 `conf/config.yaml`：

```yaml
model_folder: /absolute/path/to/CoPT-AIL/pretrain/
```

使用 **absolute path**，并保留末尾的 `/`。

BC checkpoints 会保存为：

`<model_folder><env.name>_10_bc`

例如：

`/abs/path/CoPT-AIL/pretrain/dmc_finger_spin_10_bc`

## 快速开始

### Step 1: BC pretraining (single task)

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py env=finger_spin agent=bc expert.demos=50 env.learn_steps=1e5 seed=2
```

### Step 2: Off-to-on CoPT-AIL / FOIL (single task)

```bash
CUDA_VISIBLE_DEVICES=0 python off2on.py env=finger_spin agent=foil expert.demos=50 method=il seed=2 agent.bc_transit=false project.name=run_off2on
```

`off2on.py` 会从 `model_folder` 自动加载 BC checkpoint。

## 批量脚本

- `scripts/pretrain_all.sh`: 对 8 个 DMC tasks 进行 BC pretraining。
- `scripts/run_all_off2on.sh`: 跨 tasks 运行 FOIL off-to-on。

这些脚本假设：

- Conda env named `ail`
- Multi-GPU setup（`CUDA_VISIBLE_DEVICES=0/1`）

运行前请按需修改 GPU ids、seeds 和 demo counts。

## Hydra 使用说明

- Environment configs 使用 short name 选择，例如 `env=finger_spin`。
- 始终在 command line overrides 中设置 `agent=bc|ail|foil|dac`。
- 常用 overrides：
  - `expert.demos=<n>`
  - `env.learn_steps=<n>`
  - `method.lambda_gp=<float>`
  - `seed=<int>`
  - `wandb=true`


### Bibtex

如果你觉得本代码有帮助，请按以下格式引用我们的 paper。

```
@inproceedings{xu2026provably,
title={Provably Efficient Policy-Reward Co-Pretraining for Adversarial Imitation Learning},
author={Tian Xu and Zexuan Chen and Zhilong Zhang and Yi-Chen Li and Yang Yu},
booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
year={2026},
}
```
