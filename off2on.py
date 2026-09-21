import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types

from utils.easylogger import logger

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ.setdefault('MUJOCO_GL', 'glfw' if os.name == 'nt' else 'osmesa')

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from wrappers.atari_wrapper import LazyFrames
from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from utils.utils import eval_mode, evaluate

torch.set_num_threads(2)


def get_args(cfg: DictConfig):
    if cfg.device == "auto":
        cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.model_folder = hydra.utils.to_absolute_path(cfg.model_folder)
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
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
    INITIAL_MEMORY = int(env_args.initial_mem)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)
    LEARN_STEPS = int(env_args.learn_steps)

    agent = make_agent(env, args)

    pretrain_path = os.path.join(args.model_folder, args.env.name + '_10_bc')
    if os.path.isfile(pretrain_path):
        print("=> loading pretrain '{}'".format(pretrain_path))
        agent.load_state(pretrain_path)
    else:
        raise FileNotFoundError(f"BC checkpoint not found: {pretrain_path}. Run pretrain.py with agent=bc first.")

    # Load expert data
    expert_memory_replay = Memory(REPLAY_MEMORY // 2, args.seed)
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    online_memory_replay = Memory(REPLAY_MEMORY // 2, args.seed + 1)

    if args.wandb:
        from utils.wandb import WandbLoggerHandler
        logger.add_handler(WandbLoggerHandler(cfg))

    steps = 0

    # track mean reward and scores
    rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
    best_eval_returns = -np.inf

    learn_steps = 0
    begin_learn = False
    episode_reward = 0

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        done = False

        start_time = time.time()
        for episode_step in range(EPISODE_STEPS):

            if steps < args.num_seed_steps:
                action = env.action_space.sample()
            else:
                with eval_mode(agent):
                    action = agent.choose_action(state, sample=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1

            # evaluate
            if steps % args.env.eval_interval == 0 or steps == 1:
                eval_returns, eval_timesteps, _ = evaluate(agent, eval_env, num_episodes=args.eval.eps)
                returns = np.mean(eval_returns)
                logger.logkv('eval/episode_reward', returns)
                logger.logkv('eval/episode', epoch)
                logger.dump(steps)
                # print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))

            # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
            done_no_lim = done and not info.get('TimeLimit.truncated', False)
            online_memory_replay.add((state, next_state, action, reward, done_no_lim))

            # Start learning
            if online_memory_replay.size() > INITIAL_MEMORY:
                if begin_learn is False:
                    print('Learn begins!')
                    begin_learn = True

                if learn_steps == LEARN_STEPS:
                    os.makedirs('checkpoints', exist_ok=True)
                    agent.save('checkpoints', f'_{args.env.name}')
                    logger.dump(steps)
                    env.close()
                    eval_env.close()
                    print('Finished!')
                    return

                for _ in range(args.agent.update_per_step):
                    agent.update(online_memory_replay, expert_memory_replay, logger, learn_steps)
                learn_steps += 1

            if done:
                break
            state = next_state

        rewards_window.append(episode_reward)
        logger.logkv('train/episode', epoch)
        logger.logkv('train/episode_reward', episode_reward)
        logger.logkv('train/duration', time.time() - start_time)
        logger.dump(steps)

if __name__ == "__main__":
    main()
