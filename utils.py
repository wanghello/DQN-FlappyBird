#!/bin/env python3
#coding:utf-8
#Author:thewang93@gmail.com

#convert image to a 80*80 gray image
def preprocess(observation):
    img = cv2.resize(observation, (80, 80))
    observation = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(1,80,80))

def save_checkpoint():
    pass

def load_checkpoint():
    pass

