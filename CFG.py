#!/bin/env python3
#coding:utf-8
#Author:thewang93@gmail.com

class CFG(object):
    is_training = True
    max_episode = max_episode=100000
    observations = 100
    use_cuda = True
    lr = 0.01
    mem_size = 5000
    actions = 2
    batch_size = 32
    time_step = 0
    init_e = 1
    final_e = 0.1
    epsilon = 1
    gamma = 0.99
    exploration = 50000

