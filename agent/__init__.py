import gym

from agent.ail import AIL
from agent.dac import DAC
from agent.foil import FOIL
from agent.bc import BC


def make_agent(env, args):
    obs_dim = env.observation_space.shape[0]
    if args.agent.name == "ail":
        print('--> Using AIL agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = AIL(obs_dim, action_dim, action_range, args.train.batch, args)
    elif args.agent.name == "FOIL":
        print('--> Using FOIL agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = FOIL(obs_dim, action_dim, action_range, args.train.batch, args)
    elif args.agent.name == "bc":
        print('--> Using BC agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = BC(obs_dim, action_dim, action_range, args.train.batch, args)
    else:
        print('--> Using DAC agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = DAC(obs_dim, action_dim, action_range, args.train.batch, args)

    return agent
