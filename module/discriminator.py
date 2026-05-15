import torch
import torch.nn as nn
from torch.autograd import Variable, grad
from torch import distributions as pyd

from .actor import SquashedNormal
from .net import *
from .rot import TruncatedNormal


class Discriminator(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(Discriminator, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q architecture
        self.norm = nn.LayerNorm(obs_dim + action_dim)
        self.trunk = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        obs_action = self.norm(obs_action)
        r = self.trunk(obs_action)
        r = torch.tanh(r)

        return r

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = torch.cat([obs1, action1], 1)
        policy_data = torch.cat([obs2, action2], 1)

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        r = self.forward(interpolated_state, interpolated_action)
        ones = torch.ones(r.size()).to(policy_data.device)
        gradient = grad(
            outputs=r,
            inputs=interpolated,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


class ShapedDiscriminator(nn.Module):
    def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim, scale=1):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.utils.spectral_norm(nn.Linear(hidden_dim, 2 * action_dim)),)
        self.apply(weight_init)
        self.scale = scale
        self.std = None

    def forward(self, obs, action, std=0.3, exp=False):
        h = self.trunk(obs)

        mu, log_std = self.policy(h).chunk(2, dim=-1)
        mu = torch.tanh(mu)

        self.std = std
        dist = TruncatedNormal(mu, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        if exp:
            return torch.exp(log_prob) * self.scale
        else:
            return log_prob * self.scale

    def grad_pen(self, obs_exp, act_exp, obs_pol, act_pol, lambda_=1.0):
        device = obs_exp.device
        x_exp = torch.cat([obs_exp, act_exp], dim=-1)
        x_pol = torch.cat([obs_pol, act_pol], dim=-1)

        alpha = torch.rand(x_exp.size(0), 1, device=device)
        alpha = alpha.expand_as(x_exp)

        x_hat = alpha * x_exp + (1.0 - alpha) * x_pol
        x_hat.requires_grad_(True)

        obs_dim = obs_exp.shape[1]
        act_dim = act_exp.shape[1]
        s_hat, a_hat = torch.split(x_hat, [obs_dim, act_dim], dim=-1)

        d_hat = self.forward(s_hat, a_hat)

        ones = torch.ones_like(d_hat, device=device)
        grads = torch.autograd.grad(
            outputs=d_hat,
            inputs=x_hat,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_norm = grads.view(grads.size(0), -1).norm(2, dim=1)
        gp = ((grad_norm - 1.0) ** 2).mean()
        return lambda_ * gp

class ShapedStdDiscriminator(nn.Module):
    def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim, scale=1, log_std_bounds=(-5, 2)):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 2 * action_dim))
        self.apply(weight_init)
        self.scale = scale
        self.log_std_bounds = log_std_bounds
        self.std = None

    def forward(self, obs, action, exp=False):
        h = self.trunk(obs)

        mu, log_std = self.policy(h).chunk(2, dim=-1)
        mu = torch.tanh(mu)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()
        self.std = std.mean().item()
        dist = TruncatedNormal(mu, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        if exp:
            return torch.exp(log_prob) * self.scale
        else:
            return log_prob * self.scale

    def grad_pen(self, obs_exp, act_exp, obs_pol, act_pol, lambda_=1.0):
        device = obs_exp.device
        x_exp = torch.cat([obs_exp, act_exp], dim=-1)
        x_pol = torch.cat([obs_pol, act_pol], dim=-1)

        alpha = torch.rand(x_exp.size(0), 1, device=device).expand_as(x_exp)
        x_hat = (alpha * x_exp + (1.0 - alpha) * x_pol).requires_grad_(True)

        obs_dim = obs_exp.shape[1]
        act_dim = act_exp.shape[1]
        s_hat, a_hat = torch.split(x_hat, [obs_dim, act_dim], dim=-1)

        d_hat = self.forward(s_hat, a_hat)
        ones = torch.ones_like(d_hat, device=device)
        grads = torch.autograd.grad(d_hat, x_hat, ones, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_norm = grads.view(grads.size(0), -1).norm(2, dim=1)
        gp = ((grad_norm - 1.0) ** 2).mean()
        return lambda_ * gp

class ReshapedDiscriminator(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds, args):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.trunk = mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)
        self.outputs = dict()
        self.apply(orthogonal_init_)
        self.std = None

    def forward(self, obs, action):
        EPS = (torch.finfo(action.dtype).eps * 4) if torch.is_floating_point(action) else 1e-6
        action = action.clamp(-1.0 + EPS, 1.0 - EPS)
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        self.std = std = 0.3
        mu = torch.tanh(mu)
        dist = SquashedNormal(mu, std)
        #print(mu.mean().item())
        return dist.log_prob(action).sum(-1, keepdim=True)

    @torch.no_grad()
    def reward(self, obs, action, tau=3.0, center=True, clip=10.0, per_dim_mean=True):
        """
        用作 RL 奖励：忽略 Jacobian，只用 base Normal 的 log_prob(atanh(a)).
        tau: 温度缩放 (越大 => 奖励幅度越小)
        center: 是否按 batch 去中心
        clip: 奖励裁剪范围
        per_dim_mean: 按动作维度平均，避免随维数线性变大
        """
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        mu = torch.clamp(mu, -20.0, 20.0)
        lmin, lmax = self.log_std_bounds
        log_std = torch.tanh(log_std)
        log_std = lmin + 0.5 * (lmax - lmin) * (log_std + 1.0)
        std = log_std.exp().clamp(min=1e-3)  # 用 1e-3 比 1e-6 更稳，避免极尖
        # 预反变换后的动作（稳定 atanh）
        std = 0.3
        eps = torch.finfo(action.dtype).eps * 4
        a = action.clamp(-1.0 + eps, 1.0 - eps)
        x = 0.5 * (a.log1p() - (-a).log1p())  # atanh(a)

        base = pyd.Normal(mu, std)
        lp = base.log_prob(x).sum(-1, keepdim=True)  # **不加 Jacobian**

        if per_dim_mean:
            lp = lp / action.size(-1)

        if center:
            lp = lp - lp.mean(dim=0, keepdim=True)  # batch 去中心

        r = (lp / tau).clamp(-clip, clip)
        return r

    def grad_pen(self,
           obs: torch.Tensor,
           act: torch.Tensor,
           gamma: float = 10.0,
           include_obs: bool = False,   # 默认只惩罚动作梯度
           include_action: bool = True):
        """
        Zero-centered R1: 0.5 * gamma * E[ ||∇_{(obs,act)} D(obs,act)||_2^2 ]
        返回 (penalty_scalar, stats_dict)
        """
        assert include_obs or include_action, "至少对 obs 或 action 之一施加惩罚"

        B = obs.shape[0]
        # 断开上游计算图后再打开 requires_grad，避免把梯度传回 actor/policy
        obs_r = obs.detach().requires_grad_(include_obs)
        act_r = act.detach().requires_grad_(include_action)

        score = self.forward(obs_r, act_r)  # [B, 1]

        inputs = []
        if include_obs:    inputs.append(obs_r)
        if include_action: inputs.append(act_r)

        grads = torch.autograd.grad(
            outputs=score.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )

        if include_obs and include_action:
            g_obs, g_act = grads
            g = torch.cat([g_obs.reshape(B, -1), g_act.reshape(B, -1)], dim=1)
        else:
            g = grads[0].reshape(B, -1)

        grad_norm2 = (g * g).sum(dim=1)                      # ||grad||^2 per-sample
        penalty = 0.5 * gamma * grad_norm2.mean()            # 标量

        return penalty