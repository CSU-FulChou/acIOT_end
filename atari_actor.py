'''
Author: Ful Chou
Date: 2021-04-08 17:31:08
LastEditors: Ful Chou
LastEditTime: 2021-04-10 14:27:15
FilePath: /RL-Adventure-2/atari.py
Description: What this document does
'''
import json
import gym
import numpy as np
from itertools import count
from collections import namedtuple
from PIL import Image

import matplotlib.pyplot as plt
from numpy.core.einsumfunc import _einsum_path_dispatcher

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchsummary import summary
import requests
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from http_client import url_head

n_frames = 1
gamma = 0.99
learning_rate = 3e-2
log_interval = 10
N_episodes = 100
size = (82, 82)



env = gym.make('Breakout-v0') # Breakout-v0 Atlantis-v0
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        
        self.conv1 = nn.Conv2d(3 * n_frames, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(8)
        self.actor_head = nn.Linear(392, env.action_space.n)
        self.critic_head = nn.Linear(392, 1)

        self.episode_rewards = []
        self.episode_actions = []

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) # 给了 2张（3，82，82）的图片？， 应为 summary 给了一帧图片！
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        action_prob = F.softmax(self.actor_head(x.view(x.size(0), -1)), dim=-1)
        critic_value = self.critic_head(x.view(x.size(0), -1))
        return action_prob, critic_value

model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
eps = np.finfo(np.float32).eps.item() # 非负最小值
# summary(model,input_size=(3,82,82))

def select_action(actor_critic, state):
    state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).to(device)
    probs, critic_value = actor_critic(state) # 0.2988, 0.2688, 0.2370, 0.1954
    m = Categorical(probs) # -1.2081, -1.3137, -1.4398, -1.6326
    action = m.sample() # [2]
    # print(m)
    # print(action)
    actor_critic.episode_actions.append(SavedAction(m.log_prob(action), critic_value))
    return action.item(), probs

def finish_episode():
    ''' compute loss, update model:
    '''
    R = 0
    episode_actions = model.episode_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.episode_rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards).to(device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(episode_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.mse_loss(value[0], torch.tensor(r).to(device)))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.episode_rewards[:]
    del model.episode_actions[:]

def collect_frames(n_frames, action, frame_list=[]):
    ''' 输入动作， 收集之后的 n frame 图像：
    '''
    reward = 0
    for _ in range(n_frames):
        state, temp_reward, done, _ = env.step(action)
        reward += temp_reward
        state = Image.fromarray(state)
        state = np.array(state.resize(size, Image.ANTIALIAS), dtype=np.float32) # 图像滤波， 转换成 （82， 82， 3） 的图像
        frame_list.append(state)
    return np.concatenate(frame_list, axis=2), reward, done



def main():
    render = False
    running_time = 0
    states = []
    probs_list = []
    actions = []
    episode_reward = []
    for i_episode in range(1, N_episodes + 1):
        start = time.time()
        state = env.reset()
        state = Image.fromarray(state)
        state = np.array(state.resize(size, Image.ANTIALIAS), dtype=np.float32) # 图像转换:
        states.append(state.tolist()) 
        if n_frames > 1: # 帧数 > 1
            state, reward, done = collect_frames(n_frames - 1, np.random.randint(
                env.action_space.n), frame_list=[state])
        else:
            done = False
        t = 0
        while not done  and t < 10000:
            t += 1
            action, probs = select_action(model, state)
            probs_list.append(probs.data.tolist())
            actions.append(action) # 离散动作：
            state, reward, done = collect_frames(n_frames, action, frame_list=[])
            model.episode_rewards.append(reward)
            states.append(state.tolist())
            if reward == 1: render = True
            if render:
                env.render()
                render = False
        
        action_value_pairs = [ {
            'log_prob':action_value_pair.log_prob.data.item(),
            'value':action_value_pair.value.data.item(),
        }  for action_value_pair in model.episode_actions]
        action_rewards = model.episode_rewards
        # TODO : 整理网络传输 rewards:
        states.pop() # 去掉最后一个多余的状态！
        end1 = time.time()
        print('time about interactive: ', end1-start)
        update_model(states, probs_list, actions, action_value_pairs, action_rewards, model)
        episode_reward.append(sum(action_rewards))
        print('{} rewards is {}'.format(i_episode, episode_reward[-1]), episode_reward)
        end2 = time.time()
        # finish_episode()# 改成通过网络通信 得到模型参数：
        # running_time = running_time * 0.99 + t * 0.01
        # if i_episode % log_interval == 0:
        #     print('Episode {}\tLast Length: {:5d}\tAverage length: {:.2f}'.format(
        #         i_episode, t, running_time
        #     ))
        print('time about update model', end2-end1)


def update_model(states, probs_list, actions, action_value_pairs, action_rewards, model):
    ''' 网络传输 action_value_pairs, action_reward 并且 接受model params update model
    '''
    data = {
        'states':states,
        'actions':actions,
        'probs_list':probs_list,
        'action_value_pairs':action_value_pairs,
        'action_rewards': action_rewards,
    }
    data = json.dumps(data) # 序列化
    req = requests.post(url_head+'atari/ac', data=data)
    params = json.loads(req.text)
    # print(params)
    params = {k:torch.Tensor(v) for k,v in params.items()}
    old_params = model.state_dict()
    old_params.update(params)
    



if __name__ == '__main__':
    main()