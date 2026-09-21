import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import numpy as np
if TYPE_CHECKING:
    from dataset.memory import Memory


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def evaluate(actor, env, num_episodes=10, vis=True):
    """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward, total timesteps, and normalized score (if D4RL env).
    """
    total_timesteps = []
    total_returns = []
    env_name = getattr(env, 'spec', None) and getattr(env.spec, 'id', None)

    while len(total_returns) < num_episodes:
        state = env.reset()
        done = False

        with eval_mode(actor):
            while not done:
                action = actor.choose_action(state, sample=False)
                next_state, reward, done, info = env.step(action)
                state = next_state

                if 'episode' in info.keys():
                    total_returns.append(info['episode']['r'])
                    total_timesteps.append(info['episode']['l'])

    # 计算 D4RL 归一化分数
    normalized_score = None
    if env_name and ('d4rl' in env_name or env_name.startswith('pen-') or env_name.startswith('hammer-')
                     or env_name.startswith('door-') or env_name.startswith('relocate-')
                     or 'kitchen' in env_name or 'halfcheetah' in env_name
                     or 'hopper' in env_name or 'walker' in env_name):
        try:
            import d4rl
            normalized_score = d4rl.get_normalized_score(env_name, np.mean(total_returns))
        except Exception:
            normalized_score = None

    return total_returns, total_timesteps, normalized_score

  
def get_concat_samples(policy_batch, expert_batch):
    online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = policy_batch
    expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = expert_batch

    if isinstance(policy_batch[0], np.ndarray):
        batch_state = np.concatenate([online_batch_state, expert_batch_state], axis=0)
        batch_next_state = np.concatenate(
            [online_batch_next_state, expert_batch_next_state], axis=0)
        batch_action = np.concatenate([online_batch_action, expert_batch_action], axis=0)
        batch_reward = np.concatenate([online_batch_reward, expert_batch_reward], axis=0)
        batch_done = np.concatenate([online_batch_done, expert_batch_done], axis=0)
        is_expert = np.concatenate([np.zeros_like(online_batch_reward),
                            np.ones_like(expert_batch_reward)], axis=0)
    else:
        batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
        batch_next_state = torch.cat(
            [online_batch_next_state, expert_batch_next_state], dim=0)
        batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
        batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
        batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)
        is_expert = torch.cat([torch.zeros_like(online_batch_reward, dtype=torch.bool),
                            torch.ones_like(expert_batch_reward, dtype=torch.bool)], dim=0)

    return batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert


def save_state(tensor, path, num_states=5):
    """Show stack framed of images consisting the state"""
    from torchvision.utils import save_image

    tensor = tensor[:num_states]
    B, C, H, W = tensor.shape
    images = tensor.reshape(-1, 1, H, W).cpu()
    save_image(images, path, nrow=num_states)


def average_dicts(dict1, dict2):
    return {key: 1/2 * (dict1.get(key, 0) + dict2.get(key, 0))
                     for key in set(dict1) | set(dict2)}

def _get_hydra_run_dir() -> Path:
    """尽量兼容 Hydra 1.sh.1.sh~1.sh.3 的输出目录字段。"""
    from hydra.core.hydra_config import HydraConfig
    cfg = HydraConfig.get()
    # Hydra 1.sh.2+ 推荐 runtime.output_dir
    if hasattr(cfg, "runtime") and hasattr(cfg.runtime, "output_dir"):
        return Path(cfg.runtime.output_dir)
    # 旧字段
    if hasattr(cfg, "run") and hasattr(cfg.run, "dir"):
        return Path(cfg.run.dir)
    # 兜底：当前目录
    return Path.cwd()

def wd_param_groups(model, weight_decay=1e-4):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = name.endswith(".bias")
        is_norm = ("norm" in name.lower()) or ("bn" in name.lower()) or ("layernorm" in name.lower())
        (no_decay if (is_bias or is_norm) else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

class LinearSchedule:
    """
    线性衰减调度器
    - init_value: 初始值
    - final_value: 结束值
    - max_steps: 衰减总步数
    """
    def __init__(self, init_value: float, final_value: float, max_steps: int):
        self.init_value = init_value
        self.final_value = final_value
        self.max_steps = max_steps

    def __call__(self, step: int) -> float:
        """
        根据当前 steps 返回衰减后的值
        step: 当前步数，>=0
        """
        fraction = min(float(step) / self.max_steps, 1.0)
        return self.init_value + fraction * (self.final_value - self.init_value)


def save_memory(memory: "Memory", path: str):
    """
    保存 online_memory_replay 到指定路径（使用 pickle）
    
    Args:
        memory: 要保存的 Memory 对象
        path: 保存路径（如果路径没有 .pkl 扩展名，会自动添加）
    """
    if not path.endswith('.pkl'):
        path += '.pkl'

    # 确保目录存在
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    # 将 buffer 转换为列表并保存（因为 deque 不能直接 pickle）
    buffer_list = list(memory.buffer)
    with open(path, 'wb') as f:
        pickle.dump(buffer_list, f)
    
    print(f'--> Online memory saved to {path}, size: {memory.size()}')


def load_memory(memory: "Memory", path: str):
    """
    从指定路径加载数据到 online_memory_replay（使用 pickle）
    
    Args:
        memory: 要加载数据的 Memory 对象
        path: 加载路径（如果路径没有 .pkl 扩展名，会自动添加）
    """
    if not path.endswith('.pkl'):
        path += '.pkl'

    if not os.path.isfile(path):
        print(f"[Warning]: Did not find emory file {path}")
        return False

    # 加载 pickle 文件
    with open(path, 'rb') as f:
        buffer_list = pickle.load(f)

    # 将数据添加到 memory
    for experience in buffer_list:
        memory.add(experience)

    print(f'--> memory loaded from {path}, size: {memory.size()}')
    return True

