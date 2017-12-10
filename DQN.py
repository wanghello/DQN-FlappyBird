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

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.map_size = (64, 16, 9)
        self.fc1 = nn.Linear(self.map_size[0]*self.map_size[1]*self.map_size[2], 256)
        self.fc2 = nn.Linear(256, cfg.actions)

    def foward(self, x):
        #foward procedure to get MSE loss
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return x


class DQN(object):
    empty_frame = np.zeros((80, 80), dtype=np.float32)
    empty_state = np.stack((empty_frame, empty_frame, empty_frame, empty_frame), axis=0)

    def __init__(self, cfg):
        #init all hyper parameters
        self.cfg = cfg
        self.time_step = 0
        self.epsilon = cfg.epsilon
        
        self.use_cuda = True
        
        #init model state
        self.currt_state = self.empty_state

        #init replay memory
        self.replayMemory = deque()

        #init Q network
        self.model = self.createQNetwork()

        #loss function
        self.ceriterion = nn.MSELoss()
        

        #init optimizer
        self.optimizer = torch.optim.RMSprop((self.model).parameters(), lr=cfg.lr)

    def createQNetwork(self):
        #pytorch network
        model = Net(self.cfg)
        return model

    def get_action_randomly(self):
        #random action
        action = np.zeros(self.cfg.actions, dtype=np.float32)
        action_index = 0 if random.random() < 0.8 else 1
        action[action_index] = 1
        return action

    def get_action_optim(self):
        state = self.current_state
        state_var = Variable(torch.from_numpy(state), volatile=True).unsqueeze(0)
        q_value = self.forward(state_var)
        _, action_index = torch.max(q_value, dim=1)
        action_index = action_index.data[0][0]
        action = np.zeros(self.cfg.actions, dtype=np.float32)
        action[action_index] = 1
        return action

    def get_action(self):
        cfg = self.cfg
        if cfg.is_training and random.random() <= cfg.epsilon:
            return self.get_action_randomly()
        return self.get_optim_action()


    def storeTransition(self, o_next, action, reward, terminal):
        cfg = self.cfg
        next_state = np.append(self.currt_state[1:,:,:], np.reshape(o_next,(1,o_next.shape[0],o_next.shape[1])), axis=0)
        self.replayMemory.append((self.currt_state, action, reward, next_state, terminal))
        if len(self.replayMemory) > cfg.mem_size:
            self.replayMemory.popleft()
        if not terminal:
            self.currt_state = next_state
        else:
            self.currt_state = self.empty_state


    def trainByBatch(self):
        #train model by one batch
        cfg = self.cfg
        #model = self.model
        minibatch = random.sample(self.replayMemory, cfg.batch_size)
        state_batch = np.array([data[0] for data in minibatch])
        action_batch = np.array([data[1] for data in minibatch])
        reward_batch = np.array([data[2] for data in minibatch])
        next_state_batch = np.array([data[3] for data in minibatch])

        state_batch_var = Variable(torch.from_numpy(state_batch))
        next_state_batch_var = Variable(torch.from_numpy(next_state_batch),volatile=True)
        q_value_next = self.model(next_state_batch_var)
        q_value = self.model(state_batch_var)
        y = reward_batch.astype(np.float32)
        max_q, _ = torch.max(q_value_next, dim=1)

        for i in xrange(cfg.batch_size):
            if not minibatch[i][4]:
                y[i] += cfg.gamma*max_q.data[i][0]

        y = Variable(torch.from_numpy(y))
        action_batch_var = Variable(torch.from_numpy(action_batch))
        q_value = torch.sum(torch.mul(action_batch_var, q_value), dim=1)
        loss = self.ceriterion(q_value, y)
        loss.backward()
        optimizer.step()
        self.model = model
    
    def resetState(self):
        self.currt_state = empty_state

    def increaseTimeStep(self, time_step=1):
        self.time_step += time_step

        


