import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import json
import requests

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        
        return action

    def reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
    
#https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
def plot(frame_idx, rewards):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


class PolicyNetwork(nn.Module): # actor
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0]


def post_data(state, action, reward, next_state, done):
    data = {
        'state':state,
        'action':action,
        'reward':reward,
        'next_state':next_state,
        'done':done,
    }
    data = json.dumps(data)
    print(data)
    # requests.post()

def ddpg_update():
    '''
    更新模型，获取新的 PolicyNetword的参数
    '''
    pass
def init_policy_net():
    pass


if __name__ == '__main__':
    env = NormalizedActions(gym.make("Pendulum-v0"))
    ou_noise = OUNoise(env.action_space)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
    max_frames  = 12000
    max_steps   = 500
    frame_idx   = 0
    rewards     = []
    batch_size  = 128 # mean: 每次更新的动作 数
    post_t      = 0
    while frame_idx < max_frames:
        
        state = env.reset()
        ou_noise.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = policy_net.get_action(state) # 修改 cheng
            action = ou_noise.get_action(action, step)
            print('actions:',action)
            next_state, reward, done, _ = env.step(action)
            
            # post to trainer:
            post_data(state, action, reward, next_state, done)
            post_t += 1
            if post_t > batch_size:
                # ddpg_update(batch_size) # updata
                pass
            
            state = next_state
            episode_reward += reward
            frame_idx += 1
            
            if frame_idx % max(1000, max_steps + 1) == 0:
                plot(frame_idx, rewards)
            
            if done:
                break
        
        rewards.append(episode_reward)

