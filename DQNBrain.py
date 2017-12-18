#!/bin/env python3
#coding:utf-8
#Author:thewang93@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 

import collections
import random
from collections import deque
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.map_size = (64, 10, 10)
        self.fc1 = nn.Linear(self.map_size[0]*self.map_size[1]*self.map_size[2], 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        #forward procedure
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = view(x.size()[0], -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2
        return x

class DQNBrain(object):
    empty_frame = np.zeros((80, 80), dtype=np.float32)
    empty_state = np.stack((empty_frame, empty_frame, empty_frame, empty_frame), axis=0)
    def __init__(self, cfg):
        #init all hyperparameters
        self.is_training = cfg.is_training 
        self.is_cuda = cfg.is_cuda
        self.lr = cfg.lr
        self.mem_size = cfg.mem_size
        self.actions = cfg.actions
        self.batch_size = cfg.batch_size
        self.time_step = 0
	self.gamma = cfg.gamma
        self.init_e = cfg.init_e
        self.final_e = cfg.final_e 

        #init model state
        self.currt_state = self.empty_state

        #init Q network
        self.model = Net()
        if self.is_cuda:
            self.model = self.model.cuda()

        #init replay memory
        self.replay_memory = deque()

        #init training
        self.optimizer = torch.optim.RMSprop((self.model).parameters(), lr=self.lr)
        self.ceriterion = nn.MSELoss()

    def store_transition(self, o_next, action, reward, terminal):
        """
        #o_next 
        param: next observation
        type: numpy array, [1,80,80]
        #action
        param: bird action
        type: numpy array, [2]
        #reward
        param: agent reward
        #terminal
        param: agent failed or not
        """
        next_state = np.append(self.currt_state[1:,:,:], o_next, axis=0)
        self.replay_memory.append((self.currt_state, action, reward, next_state, terminal))
        if len(self.replay_memory) > self.mem_size:
            self.replay_memory.popleft()

        if not terminal:
            self.currt_state = next_state
        else:
            self.currt_state = empty_state

        return
    
    def train_batch(self):
        #train model one batch
        minibatch = random.sample(self.replay_memory, self.batch_size)

        state_batch = np.array([data[0] for data in minibatch])
        action_batch = np.array([data[1] for data in minibatch])
        reward_batch = np.array([data[2] for data in minibatch])
        next_state_batch = np.array([data[3] for data in minibatch])

        state_batch_var = Variable(torch.from_numpy(state_batch))
        next_state_batch_var = Variable(torch.from_numpy(next_state_batch))

        if self.is_cuda():
            state_batch_var = state_batch_var.cuda()
            next_state_batch_var = next_state_batch_var.cuda()

        q_value_next = self.model(next_state_batch_var)
        q_value = self.model(state_batch_var) 

        y_batch = reward_batch.astype(np._float32)
        max_q, _ = torch.max(q_value_next, dim=1)

        for i in range(self.batch_size):
            terminal = minibatch[i][4]
            if not terminal:
                y_batch[i] += self.gamma*max_q.data[i][0]
        
        y_batch = Variable(torch.from_numpy(y_batch))
        action_batch_var = Variable(torch.from_numpy(action_batch))
        if self.is_cuda:
            y_batch = y_batch.cuda()
        q_value = torch.sum(torch.mul(action_batch_var, q_value), dim=1)

        loss = self.ceriterion(q_value, y_batch)
        loss.backward()

        self.optimizer.step()
        


    #convert image to a 80*80 gray image
    def preprocess(observation):
        img = cv2.resize(observation, (80, 80))
        observation = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
        return np.reshape(observation,(1,80,80))

    def get_random_action(self):
        #return random action
        action = np.zeros(self.actions, dtype=np.float32)
        action_index = 0 if random.random()<0.8 else 1
        action[action_index] = 1
        return action

    def get_optim_action(self):
        state_var = Variable(torch.from_numpy(self.currt_state))
        if self.is_cuda:
            state_var.cuda()
        q_value = self.model(state_var)
        _, action_index = torch.max(q_value, dim=1)
        action_index = action_index.data[0]
        action = np.zeros(self.actions, dtype=np.float32)
        action[action_index] = 1
        return action 
    
    def init_optimizer(self):
        self.optimizer.zero_grad()

    def reset_state(self):
        self.currt_state = self.empty_state

    def increase_step(self):
        self.time_step += 1

























