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

def train_DQN(cfg):
    best_time_step = 0.

    #game start
    flappyBird = game.GameState()
    
    #set optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=options.lr)
    ceriterion = nn.MSELoss()
    
    #DQN brain
    brain = DQNBrain(cfg)
    if cfg.use_cuda:
        brain = brain.cuda()

    #init replay memory by random action
    for i in range(cfg.observations):
        action = dqn.get_random_action()
        o, r, terminal = flappyBird.frame_step(action)
        o = preprocess(o)
        brain.store_transition(o, action, r, terminal)

    for episode in range(cfg.max_episode):
        total_value = 0
        while True:
            

        



