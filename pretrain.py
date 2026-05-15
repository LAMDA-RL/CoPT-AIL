import os
import random
import time
from collections import deque
from itertools import count

from utils.easylogger import logger
from utils.wandb import WandbLoggerHandler

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from utils.utils import evaluate

torch.set_num_threads(2)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_args = args.env

    env = make_env(args)
    eval_env = make_env(args)

    REPLAY_MEMORY = int(env_args.replay_mem)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)

    agent = make_agent(env, args)

    # Load expert data
    expert_memory_replay = Memory(REPLAY_MEMORY // 2, args.seed)
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    online_memory_replay = None

    if args.wandb:
        logger.add_handler(WandbLoggerHandler(cfg))

    # track mean reward and scores
    rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards

    learn_steps = 0

    for epoch in count():
        episode_reward = 0

        start_time = time.time()
        for episode_step in range(EPISODE_STEPS):

            # evaluate
            if learn_steps % args.env.eval_interval == 0:
                eval_returns, eval_timesteps = evaluate(agent, eval_env, num_episodes=args.eval.eps)
                returns = np.mean(eval_returns)
                logger.logkv('eval/episode_reward', returns)
                logger.logkv('eval/episode', epoch)
                logger.dump(learn_steps)
                # print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))


            for _ in range(args.agent.update_per_step):
                agent.update(online_memory_replay, expert_memory_replay, logger, learn_steps)
            learn_steps += 1

            if learn_steps > args.env.learn_steps:
                break

        rewards_window.append(episode_reward)
        logger.logkv('train/episode', epoch)
        logger.logkv('train/episode_reward', episode_reward)
        logger.logkv('train/duration', time.time() - start_time)
        logger.dump(learn_steps)

        if learn_steps > args.env.learn_steps:
            break

    agent.save(args.model_folder, args.env.name + "_10")

if __name__ == "__main__":
    main()
