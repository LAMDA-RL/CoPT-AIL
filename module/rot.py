import math
import re

import numpy as np
import torch
from torch import autograd
import torch.nn.functional as F
from torch import nn
from torch import distributions as pyd
from torch.autograd import Variable, grad
from torch.distributions.utils import _standard_normal
from torch.distributions import Distribution, constraints

from utils import utils


class TruncatedNormal(pyd.Normal):
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, sample_shape=torch.Size()):
		with torch.no_grad():
			return self.rsample(sample_shape)

	@staticmethod
	def _phi(z):  # 标准正态 CDF
		return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


	@staticmethod
	def _phi_inv(u):  # 标准正态 逆CDF
		return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


	def rsample(self, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		loc = self.loc.expand(shape)
		scale = self.scale.expand(shape)
		alpha = (self.low - loc) / scale
		beta = (self.high - loc) / scale
		ua, ub = self._phi(alpha), self._phi(beta)
		u = torch.rand_like(loc).clamp_(self.eps, 1 - self.eps)
		u = ua + (ub - ua) * u
		z = self._phi_inv(u)
		return loc + scale * z  # 可重参数化


	def log_prob(self, value):
		value = value.clamp(self.low + self.eps, self.high - self.eps)
		base = super().log_prob(value)
		alpha = (self.low - self.loc) / self.scale
		beta = (self.high - self.loc) / self.scale
		Z = self._phi(beta) - self._phi(alpha)
		return base - torch.log(Z.clamp_min(self.eps))

class TanhTransform(torch.distributions.transforms.Transform):
	domain = constraints.real
	codomain = constraints.interval(-1.0, 1.0)
	bijective = True
	sign = +1

	def __init__(self, cache_size=1):
		super().__init__(cache_size=cache_size)

	@staticmethod
	def atanh(y):
		return 0.5 * (torch.log1p(y) - torch.log1p(-y))

	def _call(self, x):
		return x.tanh()

	def _inverse(self, y):
		return self.atanh(y)

	def log_abs_det_jacobian(self, x, y):
		return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))

class TanhSquashedNormal(Distribution):
	arg_constraints = {
		"loc": constraints.real,
		"scale": constraints.positive
	}
	support = constraints.interval(-1.0, 1.0)
	has_rsample = True

	def __init__(self, loc, scale, eps=1e-6, validate_args=False):
		super().__init__(batch_shape=torch.Size(), validate_args=validate_args)
		self.loc   = loc
		self.scale = scale
		self.eps   = eps
		self._normal = torch.distributions.Normal(loc, scale)
		self._tanh = TanhTransform(cache_size=1)

	def _extended_shape(self, sample_shape=torch.Size()):
		return sample_shape + self.loc.shape

	def rsample(self, sample_shape=torch.Size()):
		x = self._normal.rsample(sample_shape)     # Z
		return torch.tanh(x)                       # Y

	@torch.no_grad()
	def sample(self, sample_shape=torch.Size()):
		x = self._normal.sample(sample_shape)
		return torch.tanh(x)

	def log_prob(self, value, sum_last_dim=False):
		v_clamped = torch.clamp(value, -1.0 + self.eps, 1.0 - self.eps)
		v = value + (v_clamped - value).detach()

		x = TanhTransform.atanh(v)  # z = atanh(y)

		base_log_prob = self._normal.log_prob(x)

		log_abs_det = 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
		lp = base_log_prob - log_abs_det

		if sum_last_dim:
			return lp.sum(dim=-1)
		return lp

	@property
	def mean(self):
		return torch.tanh(self.loc)

	@property
	def mode(self):
		return torch.tanh(self.loc)


class Actor(nn.Module):
	def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim):
		super().__init__()

		self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, action_dim))
		self.apply(weight_init)

	def forward(self, obs, std=0.2):
		h = self.trunk(obs)

		mu = self.policy(h)
		mu = torch.tanh(mu)
		std = torch.ones_like(mu) * std

		dist = TruncatedNormal(mu, std)
		return dist

	def act(self, obs, std=0.2):
		dist = self.forward(obs, std)
		return dist.sample()

	def sample(self, obs, std=0.2):
		dist = self.forward(obs, std)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		return action, log_prob, dist.mean

	def log_prob(self, obs, action, std=0.2):
		dist = self.forward(obs, std)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		return log_prob

class ActorStd(nn.Module):
	def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim, log_std_bounds=(-5, 2)):
		super().__init__()

		self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, 2 * action_dim))
		self.apply(weight_init)
		self.log_std_bounds = log_std_bounds
		self.std = None

	def forward(self, obs):
		h = self.trunk(obs)

		mu, log_std = self.policy(h).chunk(2, dim=-1)
		mu = torch.tanh(mu)
		log_std = torch.tanh(log_std)
		log_std_min, log_std_max = self.log_std_bounds
		log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

		std = log_std.exp()
		self.std = std.mean().item()

		dist = TruncatedNormal(mu, std)
		return dist

	def act(self, obs):
		dist = self.forward(obs)
		return dist.sample()

	def sample(self, obs):
		dist = self.forward(obs)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		return action, log_prob, dist.mean

	def log_prob(self, obs, action):
		dist = self.forward(obs)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		return log_prob

	@staticmethod
	def atanh(x, eps=1e-6):
		x = x.clamp(-1+eps, 1-eps)
		return 0.5 * (x.log1p() - (-x).log1p())

	def pre_tanh_params(self, obs):
		h = self.trunk(obs)
		mu, log_std = self.policy(h).chunk(2, dim=-1)
		log_std_min, log_std_max = self.log_std_bounds
		log_std = torch.clamp(log_std, log_std_min, log_std_max)
		return mu, log_std


def weight_init(m):
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)


class Critic(nn.Module):
	def __init__(self, repr_dim, action_dim, feature_dim, hidden_dim):
		super().__init__()

		self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		self.Q1 = nn.Sequential(
			nn.Linear(feature_dim + action_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.Q2 = nn.Sequential(
			nn.Linear(feature_dim + action_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.apply(weight_init)

	def forward(self, obs, action, min=True):
		h = self.trunk(obs)
		h_action = torch.cat([h, action], dim=-1)
		q1 = self.Q1(h_action)
		q2 = self.Q2(h_action)
		if min:
			return torch.min(q1, q2)
		return q1, q2


