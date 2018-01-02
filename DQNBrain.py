#!/bin/env python3
#coding:utf-8
#Author:shawn_wang123@163.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import collections
import random
from collections import deque
from torch.autograd import Variable

class DQNBrain(nn.Module):
    empty_frame = np.zeros((80, 80), dtype=np.float32)
    empty_state = np.stack((empty_frame, empty_frame, empty_frame, empty_frame), axis=0)

    def __init__(self, cfg):
        super(DQNBrain, self).__init__()
        #init all hyper parameters
        self.time_step = 0
        self.actions = cfg.actions
        self.epsilon = cfg.epsilon
        self.mem_size = cfg.mem_size 
        self.use_cuda = cfg.use_cuda 
        self.actions = cfg.actions
        self.is_training = True 
        
        #init model state
        self.currt_state = self.empty_state

        #init replay memory
        self.replayMemory = deque()

        #init Q network
        self.Net()

    def Net(self):
        #pytorch network
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.map_size = (64, 10, 10)
        self.fc1 = nn.Linear(self.map_size[0]*self.map_size[1]*self.map_size[2], 256)
        self.fc2 = nn.Linear(256, self.actions)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return x

    def get_action_randomly(self):
        #get random action
        action = np.zeros(self.actions, dtype=np.float32)
        action_index = 0 if random.random() < 0.8 else 1
        action[action_index] = 1
        return action

    def get_action_optim(self):
        #get action by Q net
        state = self.currt_state
        state_var = Variable(torch.from_numpy(state), volatile=True).unsqueeze(0)
        if self.use_cuda:
            state_var = state_var.cuda()

        q_value = self.forward(state_var)
        _, action_index = torch.max(q_value, dim=1)
        action_index = action_index.data[0]
        action = np.zeros(self.actions, dtype=np.float32)
        action[action_index] = 1
        return action 

    def store_transition(self, o_next, action, reward, terminal):
        next_state = np.append(self.currt_state[1:,:,:], o_next, axis=0)
        self.replayMemory.append((self.currt_state, action, reward, next_state, terminal))
        if len(self.replayMemory) > self.mem_size:
            self.replayMemory.popleft()
        if not terminal:
            self.currt_state = next_state
        else:
            self.currt_state = self.empty_state 

    def close_train(self):
        self.is_training = False

    def reset_state(self):
        self.currt_state = self.empty_state

    def increase_step(self, time_step=1):
        self.time_step += time_step


        


