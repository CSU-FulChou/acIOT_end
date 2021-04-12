# -*- coding: UTF-8 -*-
from http_client import send_status_recv_parameter
from multiprocessing_env import SubprocVecEnv
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
import requests
import json
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env
    return _thunk


# 从服务端获取参数：
a2c_env_param = json.loads(requests.get(
    'http://192.168.199.128:5000/get_a2c_env_param').text)
num_envs = a2c_env_param['num_envs']
env_name = a2c_env_param['env_name']
envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)
# envs_test = SubprocVecEnv([make_env() for i in range(num_envs)])
env = gym.make(env_name)


def test_env(model, vis=False):
    state = env.reset()
    if vis:
        env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis:
            env.render()
        total_reward += reward
    return total_reward


def plot(frame_idx, rewards):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    # plt.savefig('frame %s. reward: %s.jpg' % (frame_idx, rewards[-1]))
    plt.show()


# actor model
class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, reduce_factor, std=0.0):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size // reduce_factor),
            nn.ReLU(),
            nn.Linear(hidden_size // reduce_factor,
                      hidden_size // reduce_factor),
            nn.ReLU(),
            nn.Linear(hidden_size // reduce_factor, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist


num_inputs = envs.observation_space.shape[0]
num_outputs = envs.action_space.n

# Hyper params:
hidden_size = a2c_env_param['hidden_size']
num_steps = a2c_env_param['num_steps']
reduce_factor = a2c_env_param['reduce_factor']
stop_t = 0  # 大于195 3次即认为模型 训练ok
stop_max = a2c_env_param['stop_max']
target = a2c_env_param['target']
model = Actor(num_inputs, num_outputs, hidden_size, reduce_factor).to(device)
summary(model,input_size=((1,4)))

max_frames = 80000
frame_idx = 0
test_rewards = []

start = time.time()
state = envs.reset()
while frame_idx < max_frames:
    states = []
    next_states = []
    actions = []
    rewards = []
    masks = []
    dist_probs = []
    log_probs = []
    values = []
    entropy = 0
    for _ in range(num_steps):
        states.append(state.tolist())  # add 2 buffer
        state = torch.FloatTensor(state).to(device)
        dist = model(state)
        dist_probs.append(dist.probs.data.tolist())
        action = dist.sample()
        actions.append(action.tolist())
        next_state, reward, done, _ = envs.step(action.cpu().numpy())
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        log_probs.append(log_prob.tolist())
        rewards.append(reward.tolist())
        masks.append((1 - done).tolist())
        next_states.append(next_state.tolist())
        state = next_state
        frame_idx += 1
        if frame_idx % 100 == 0:
            test_rewards.append(np.mean([test_env(model) for _ in range(10)]))
            print('frame_idx: ', frame_idx, test_rewards)
            if test_rewards[-1] >= target:
                stop_t += 1
            else:
                stop_t = 0
    # train end:
    if stop_t >= stop_max:
        torch.save(model, 'model.pth')
        break
    states.append(state.tolist())
    entropy = entropy.item()
    parameters = send_status_recv_parameter(
        states, actions, rewards, masks, dist_probs)
    # update parameters:
    model_parameters = model.state_dict()
    parameters = {k: torch.FloatTensor(v) for k, v in parameters.items()}
    model_parameters.update(parameters)
    model.load_state_dict(model_parameters)
    end = time.time()
    print('time: ', end - start, '\n')
