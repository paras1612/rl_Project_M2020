import math
import numpy as np
import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, max_size=int(1e6)):
        self.iter = 0
        self.size = max_size
        self.cur_state = np.zeros((self.size, obs_dim))
        self.action = np.zeros((self.size, action_dim))
        self.reward = np.zeros((self.size, 1))
        self.next_state = np.zeros((self.size, obs_dim))
        self.done = np.zeros((self.size, 1))
        self.filled_buffer = False

    def insert(self, state, action, reward, next_state, done):
        self.cur_state[self.iter] = state
        self.action[self.iter] = action
        self.reward[self.iter] = reward
        self.next_state[self.iter] = next_state
        self.done[self.iter] = done
        self.iter += 1
        self.filled_buffer = (self.iter == self.size or self.filled_buffer)
        self.iter %= self.size

    def get_batch(self, batch_size):
        start = 0
        end = self.iter if not self.filled_buffer else self.size
        size = self.iter if (self.iter <= batch_size and not self.filled_buffer) else batch_size
        indices = np.random.randint(start, end, size)
        return self.cur_state[indices], self.action[indices], self.reward[indices], self.next_state[indices], \
               self.done[indices]


# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is unstable numerically
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(), AffineTransform(loc=-1, scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))