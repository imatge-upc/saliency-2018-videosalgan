import cv2
import os
import datetime
import numpy as np
from modules.clstm import ConvLSTMCell
import pickle
import torch
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from data_loader import DHF1K_frames

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor


learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
start_epoch = 1
epochs = 5
plot_every = 5
load_model = False
pretrained_model = './SalConvLSTM.pt'
frame_batch_size = 5 #out of memory at 10! with 2 gpus. Works with 7 but occasionally produces error as well.

number_of_videos = 10

# Parameters
params = {'batch_size': 1,
          'num_workers': 4,
          'pin_memory': True}

def main(params = params):

    # =================================================
    # ================ Data Loading ===================

    #Expect Error if either validation size or train size is 1
    train_set = DHF1K_frames( split = "train",
        transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])) #add a parameter node = training or validation
    print("Size of train set is {}".format(len(train_set)))
    val_set = DHF1K_frames( split = "validation",
        transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ]))
    print("Size of validation set is {}".format(len(val_set)))

    #print(len(train_set[0]))
    #print(len(train_set[1]))

    # List of data loaders
    train_loader = data.DataLoader(train_set, **params)
    val_loade = data.DataLoader(val_set, **params)



    # =================================================
    # ================== Training =====================

    for epoch in range(start_epoch, epochs):
        # train for one epoch
        load(train_loader)

def load(train_loader):

    for i, video in enumerate(train_loader):
        for j, (frame, gt) in enumerate(video):
            print(frame.size())
            print(gt.size())
            break


main()
