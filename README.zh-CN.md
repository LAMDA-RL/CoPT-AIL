# CoPT-AIL 使用说明（中文）

本仓库实现了基于 DeepMind Control Suite 的模仿学习流程，包含 BC 预训练、AIL/DAC 基线和 FOIL（off-to-on）训练。

## 1. 当前仓库状态（先看）

- 代码中引用了 `utils/` 和 `wrappers/`（如日志、评估、Atari wrapper、动作归一化），但当前目录中未包含这两个文件夹。
- `conf/config.yaml` 默认是 `agent: mb_ail`，而 `conf/agent/` 中并没有 `mb_ail.yaml`。
- `experts/` 和 `pretrain/` 目录默认是空目录，需要你手动准备数据和权重。

如果你直接运行报 `ModuleNotFoundError`，优先检查是否缺少 `utils/` / `wrappers/`。

## 2. 项目结构

```text
.
|-- agent/                 # BC / AIL / DAC / FOIL
|-- conf/                  # Hydra 配置（env, agent, method）
|-- dataset/               # 专家数据读取与 replay memory
|-- dmc2gym/               # 本地 dm_control -> gym 封装
|-- envs/                  # 自定义 MuJoCo 环境
|-- experts/               # 放专家轨迹 .pkl（需自行下载）
|-- module/                # actor / critic / discriminator
|-- pretrain/              # 建议的 BC checkpoint 输出目录
|-- scripts/               # 批量脚本
|-- pretrain.py            # BC 预训练入口
|-- train.py               # AIL / DAC 在线训练入口
`-- off2on.py              # off-to-on 训练入口（FOIL / AIL）
```

## 3. 环境安装

建议环境：

- Python `3.7.16`
- MuJoCo + dm_control 可用
- 可选 CUDA GPU

安装命令：

```bash
pip install -r requirements.txt
pip install -e ./dmc2gym
```

说明：`requirements.txt` 中 PyTorch 版本被固定为 `1.13.1+cu117`，如果你的 CUDA/平台不匹配，请按本机环境调整。

## 4. 专家数据准备

1. 从以下地址下载专家数据：

   `https://osf.io/ceh6q/?view_only=41aa006b2fd149f7815294034a4792b0`

2. 将 `.pkl` 文件放到 `experts/` 目录。

环境配置中预期的文件名（来自 `conf/env/*.yaml`）：

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

## 5. 训练前关键配置

编辑 `conf/config.yaml`：

```yaml
model_folder: /absolute/path/to/CoPT-AIL/pretrain/
```

要求：

- 使用绝对路径。
- 建议保留末尾 `/`。
- 运行命令时始终显式传 `agent=bc|ail|foil|dac`，避免默认 `mb_ail` 导致配置错误。

BC 权重命名规则（由代码固定）：

`<model_folder><env.name>_10_bc`

例如：`/abs/path/CoPT-AIL/pretrain/dmc_finger_spin_10_bc`

## 6. 快速开始（单任务）

### 6.1 BC 预训练

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py env=finger_spin agent=bc expert.demos=50 env.learn_steps=1e5 seed=2
```

### 6.2 FOIL（CoPT-AIL）off-to-on

```bash
CUDA_VISIBLE_DEVICES=0 python off2on.py env=finger_spin agent=foil expert.demos=50 method=il seed=2 agent.bc_transit=false project.name=run_off2on
```

`off2on.py` 会根据 `model_folder` 自动尝试加载对应 BC 权重。

### 6.3 AIL（从零开始）

```bash
CUDA_VISIBLE_DEVICES=0 python train.py env=finger_spin agent=ail expert.demos=50 method=il method.lambda_gp=10 seed=2 project.name=run
```

### 6.4 AIL（加载预训练 actor）

```bash
CUDA_VISIBLE_DEVICES=0 python train.py env=finger_spin agent=ail expert.demos=50 method=il method.lambda_gp=10 seed=2 pretrain=/abs/path/CoPT-AIL/pretrain/dmc_finger_spin_10_bc
```

## 7. 批量脚本说明

- `scripts/pretrain_all.sh`：8 个任务批量 BC 预训练
- `scripts/run_all_off2on.sh`：8 个任务批量 FOIL off-to-on
- `scripts/run_all_ail.sh`：8 个任务批量 AIL（from scratch）
- `scripts/run_all_pre_ail.sh`：8 个任务批量 AIL（通过 `off2on.py`）

脚本默认假设：

- Conda 环境名是 `ail`
- 使用 `CUDA_VISIBLE_DEVICES=0/1`

请按你的机器修改 GPU、seed、demo 数量。

## 8. 常用 Hydra 覆盖项

- `env=<task>`，如 `env=finger_spin`
- `agent=bc|ail|foil|dac`
- `expert.demos=<n>`
- `env.learn_steps=<n>`
- `method.lambda_gp=<float>`
- `seed=<int>`
- `wandb=true`

## 9. 最小自检清单

先做这 4 步，可以快速定位环境问题：

1) 检查专家数据是否在位

```bash
ls experts
```

2) 检查关键包能否导入

```bash
python -c "import torch, hydra, dm_control"
```

3) 先跑一个短 BC 任务（几千步）

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py env=finger_spin agent=bc expert.demos=10 env.learn_steps=2000 seed=2
```

4) 再跑一个短 off-to-on 任务

```bash
CUDA_VISIBLE_DEVICES=0 python off2on.py env=finger_spin agent=foil expert.demos=10 env.learn_steps=2000 seed=2
```

## 10. 常见问题

- `ModuleNotFoundError: utils` 或 `wrappers`：当前仓库缺少对应目录，需要补齐代码或调整 `PYTHONPATH`。
- 找不到预训练权重：检查 `model_folder` 是否绝对路径、是否有结尾 `/`、文件名是否为 `dmc_<task>_10_bc`。
- 专家数据读取失败：确认 `.pkl` 文件在 `experts/`，且文件名与 `conf/env/*.yaml` 中 `env.demo` 一致。
- 训练命令报 agent 配置错误：命令里显式加 `agent=bc|ail|foil|dac`。

