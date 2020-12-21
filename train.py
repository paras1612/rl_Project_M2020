import random
import uuid
from argparse import ArgumentParser
from collections import deque

import gym
from gym.wrappers import RescaleAction
import numpy as np
import torch

from networks import DoubleQFunc, Policy
from utils import ReplayBuffer
import copy


seed = 100
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def get_action(state, policy):
    with torch.no_grad():
        action, _, mean = policy(torch.Tensor(state).view(1, -1))
    return np.atleast_1d(action.squeeze().cpu().numpy())


def train(env, critic_net, target_net, policy, total_steps=10**6, lr = 3e-4, gamma = 0.99, polyak = 0.995, batch_size = 256):
    update_interval = 1
    target_entropy = -env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    log_alpha = torch.zeros(1, requires_grad=True)
    alpha = log_alpha.exp()
    temp_optimizer = torch.optim.Adam([log_alpha], lr=lr)

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    freq_print = 1000
    start_after = 10000
    update_after = 1000
    update_times = 1

    episode_rewards = []
    state = env.reset()
    episode_reward = 0
    q1_loss, q2_loss, policy_loss_ = 0., 0., 0.
    for step in range(total_steps):
        if step%10**4==0 and step>1:
            torch.save(policy, "Policy"+str(step)+"pt")
        with torch.no_grad():
            if step < start_after:
                action = env.action_space.sample()
            else:
                action = get_action(state, policy)
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            replay_buffer.insert(state, action, reward, next_state, done)
            
            state = next_state
            if done:
                state = env.reset()
                episode_rewards.append(episode_reward)
                episode_reward = 0

            if step % freq_print == 0 and step>1:
                running_reward = np.mean(episode_rewards)
                print(step, np.min(episode_rewards), running_reward, np.max(episode_rewards), q1_loss + q2_loss, policy_loss_, alpha)
                log = [step, np.min(episode_rewards), running_reward, np.max(episode_rewards), q1_loss + q2_loss, policy_loss_]
                log = [str(x) for x in log]
                with open("plt_file", "a") as s:
                    s.write(" ".join(log) + "\n")
                episode_rewards = []

        if step > update_after: # and step % update_interval == 0:
            for update_count in range(update_times):
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.get_batch(
                    batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = torch.from_numpy(
                    batch_states).float(), torch.from_numpy(batch_actions).float(), torch.from_numpy(
                    batch_rewards).float(), torch.from_numpy(batch_next_states).float(), torch.from_numpy(
                    batch_dones).float()

                with torch.no_grad():
                    next_actions, logprob_next_actions, _ = policy(batch_next_states)
                    q_t1, q_t2 = target_net(batch_next_states, next_actions)
                    q_target = torch.min(q_t1, q_t2)
                    critic_target = batch_rewards + (1.0 - batch_dones) * gamma * (
                                q_target - alpha * logprob_next_actions)

                q_1, q_2 = critic_net(batch_states, batch_actions)
                loss_1 = torch.nn.MSELoss()(q_1, critic_target)
                loss_2 = torch.nn.MSELoss()(q_2, critic_target)

                q_loss_step = loss_1 + loss_2
                critic_net.optimizer.zero_grad()
                q_loss_step.backward()
                critic_net.optimizer.step()
                q1_loss = loss_1.detach().item()
                q2_loss = loss_2.detach().item()

                for p in critic_net.parameters():
                    p.requires_grad = False
                policy_action, log_prob_policy_action, _ = policy(batch_states)
                p1, p2 = critic_net(batch_states, policy_action)
                target = torch.min(p1, p2)
                policy_loss = (alpha * log_prob_policy_action - target).mean()
                policy.optimizer.zero_grad()
                policy_loss.backward()
                policy.optimizer.step()

                temp_loss = -log_alpha * (log_prob_policy_action.detach() + target_entropy).mean()
                temp_optimizer.zero_grad()
                temp_loss.backward()
                temp_optimizer.step()
                for p in critic_net.parameters():
                    p.requires_grad = True

                alpha = log_alpha.exp()

                policy_loss_ = policy_loss.detach().item()
                with torch.no_grad():
                    for target_q_param, q_param in zip(target_net.parameters(), critic_net.parameters()):
                        target_q_param.data.copy_((1-polyak) * q_param.data + polyak * target_q_param.data)




def main():
    envs = {0: ['Walker2d-v2', 5], 1: ['Hopper-v2', 5], 2: ['HalfCheetah-v2', 1]}
    ind = 1
    
    env_name = envs[ind][0]
    env = gym.make(env_name)
    env = RescaleAction(env, -1, 1)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(action_dim, env.action_space.low, env.action_space.high)

    critic_net = DoubleQFunc(obs_dim, action_dim)
    target_net = copy.deepcopy(critic_net)
    target_net.eval()
    policy = Policy(obs_dim, action_dim)

    train(env, critic_net, target_net, policy)


if __name__ == '__main__':
    main()