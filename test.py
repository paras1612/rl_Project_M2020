import gym
from gym.wrappers import RescaleAction
import torch
from sac import Policy
import numpy as np

def get_action(state, policy, deterministic=False):
    # if state_filter:
    #     state = state_filter(state)
    with torch.no_grad():
        action, _, mean = policy(torch.Tensor(state).view(1, -1))
    if deterministic:
        return mean.squeeze().cpu().numpy()
    return np.atleast_1d(action.squeeze().cpu().numpy())

def evaluate_agent(env, policy, state_filter, n_starts=1):
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            action = get_action(state, policy, deterministic=True)
            nextstate, reward, done, _ = env.step(action)
            reward_sum += reward
            state = nextstate
            env.render()
    return reward_sum / n_starts

envs = {0: ['Walker2d-v2', 5], 1: ['Hopper-v2', 5], 2: ['HalfCheetah-v2', 1]}
ind = 1

env_name = envs[ind][0]
env = gym.make(env_name)
env = RescaleAction(env, -1, 1)

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print(action_dim, env.action_space.low, env.action_space.high)

save_path = "Hopper/Policy200000pt"


# policy  = Policy(obs_dim, action_dim)
policy = torch.load(save_path)

evaluate_agent(env, policy, False, n_starts=10000)
