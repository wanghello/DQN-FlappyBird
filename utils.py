#!/bin/env python3
#coding:utf-8
#Author:thewang93@gmail.com

import cv2
import numpy as np

import torch

from DQNBrain import DQNBrain 

#convert image to a 80*80 gray image
def preprocess(observation):
    img = cv2.resize(observation, (80, 80))
    observation = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(1,80,80))


#save model
def save_checkpoint(state, path):
    torch.save(state, path)
    

#load model
def load_checkpoint(model, path, use_cuda):
    checkpoint = torch.load(path)
    pass

