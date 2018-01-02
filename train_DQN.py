#!/bin/env python3
#coding:utf-8
#Author:thewang93@gmail.com

from CFG import CFG 
from DQNBrain import DQNBrain 
from utils import *

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game

import numpy as np
import random
import torch 
import torch.nn as nn
from torch.autograd import Variable

def train_DQN():
    #init confirgations
    cfg = CFG()
    best_time_step = 0.
    
    #DQN brain
    dqn = DQNBrain(cfg)
    if cfg.use_cuda:
        dqn = dqn.cuda()

    #game start
    flappyBird = game.GameState()
    
    #set optimizer
    optimizer = torch.optim.RMSprop(dqn.parameters(), lr=cfg.lr)
    ceriterion = nn.MSELoss()

    #init replay memory by random action
    for i in range(cfg.observations):
        action = dqn.get_action_randomly()
        o, r, terminal = flappyBird.frame_step(action)
        o = preprocess(o)
        dqn.store_transition(o, action, r, terminal)

    for episode in range(cfg.max_episode):
        total_value = 0
        while True:
            optimizer.zero_grad()
            if random.random() <= cfg.epsilon:
                action = dqn.get_action_randomly()
            else:
                action = dqn.get_action_optim()

            o_next, r, terminal = flappyBird.frame_step(action)
            total_value += cfg.gamma*total_value + r
            o_next = preprocess(o_next)
            #update replay memory
            dqn.store_transition(o_next, action, r, terminal)
            dqn.increase_step()
            #train dqn brain model by one batch
            #step 1: sample training data from replay memory
            minibatch = random.sample(dqn.replayMemory, cfg.batch_size)
            state_batch = np.array([data[0] for data in minibatch])
            action_batch = np.array([data[1] for data in minibatch])
            reward_batch = np.array([data[2] for data in minibatch])
            next_state_batch = np.array([data[3] for data in minibatch])

            state_batch_var = Variable(torch.from_numpy(state_batch))
            next_state_batch_var = Variable(torch.from_numpy(next_state_batch))

            if cfg.use_cuda:
                state_batch_var = state_batch_var.cuda()
                next_state_batch_var = next_state_batch_var.cuda()

            #step 2: get label y
            q_value = dqn.forward(state_batch_var)
            q_value_next = dqn.forward(next_state_batch_var)

            y_batch = reward_batch.astype(np.float32)
            max_q, _ = torch.max(q_value_next, dim=1)
            for i in range(cfg.batch_size):
                if not minibatch[i][4]: #terminal
                    y_batch[i] = y_batch[i]*cfg.gamma + max_q.data[i]

            y_batch = Variable(torch.from_numpy(y_batch))
            action_batch_var = Variable(torch.from_numpy(action_batch))#predict action

            if cfg.use_cuda:
                y_batch = y_batch.cuda()
                action_batch_var = action_batch_var.cuda()

            q_value = torch.sum(torch.mul(action_batch_var, q_value), dim=1)#predict value

            #step 3: bp to update model
            loss = ceriterion(q_value, y_batch)
            loss.backward()

            optimizer.step()
            #end episode when bird's dead
            if terminal:
                dqn.time_step = 0
                break

        #update epsilon
        if dqn.epsilon > cfg.final_e:
            delta = (cfg.init_e - cfg.final_e)/cfg.exploration
            dqn.epsilon -= delta

        #test dqn per 100 episode
        if episode % 100 == 0:
            ave_step = test_DQN(dqn, episode)

        
        #if ave_step > best_time_step:
        #    best_time_step = ave_step
        #`    save_checkpoint({


def test_DQN(dqn, episode):
    """
    test DQN model
    param dqn: dqn model
    param episode: current episode
    """
    #test on 5 games
    case_num = 5

    dqn.close_train()
    ave_step = 0
    for i in range(case_num):
        dqn.time_step = 0
        flappyBird = game.GameState()
        o, r, terminal = flappyBird.frame_step([1,0])
        o = preprocess(o)
        dqn.reset_state()
        #play game until game end
        while True:
            action = dqn.get_action_optim()
            o, r, terminal = flappyBird.frame_step(action)
            if terminal:
                break #game over
            o = preprocess(o)
            dqn.currt_state = np.append(dqn.currt_state[1:,:,:], o, axis=0)
            dqn.increase_step()
        ave_step += dqn.time_step
    ave_step = ave_step / case_num
    print("episode:{}, average game steps:{}".format(episode, ave_step))
    return ave_step


if __name__ == '__main__':
    train_DQN()
