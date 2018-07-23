import cv2
import os
import datetime
import numpy as np
from modules.clstm import ConvLSTMCell
import torch
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from data_loader import Ego_frames

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor

clip_length = 5
pretrained_model = './SalConvLSTM.pt'

""" EgoMon
src = "/imatge/lpanagiotis/projects/saliency/public_html/2016-egomon/egomon_saliency_maps"
dst = "/imatge/lpanagiotis/work/Egomon/clstm_predictions"
"""
# GTEA
frame_size = 360

src = "/imatge/lpanagiotis/work/GTEA_Gaze/predictions"
dst = "/imatge/lpanagiotis/work/GTEA_Gaze/clstm_predictions"
# Parameters
params = {'batch_size': 1, # number of videos / batch, I need to implement padding if I want to do more than 1, but with DataParallel it's quite messy
          'num_workers': 4,
          'pin_memory': True}

def main():

    # =================================================
    # ================ Data Loading ===================

    #Expect Error if either validation size or train size is 1
    dataset = Ego_frames(
        frames_path = src,
        clip_length = clip_length,
        transforms = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor()
            ])
        )
         #add a parameter node = training or validation
    print("Size of test set is {}".format(len(dataset)))
    activity = dataset.match_i_to_act
    print(activity)

    #print(len(dataset[0]))
    #print(len(dataset[1]))

    loader = data.DataLoader(dataset, **params)

    # =================================================
    # ================= Load Model ====================

    # Using same kernel size as they do in the DHF1K paper
    # Amaia uses default hidden size 128
    # input size is 1 since we have grayscale images
    model = ConvLSTMCell(use_gpu=True, input_size=1, hidden_size=128, kernel_size=3)

    temp = torch.load(pretrained_model)['state_dict']
    # Because of dataparallel there is contradiction in the name of the keys so we need to remove part of the string in the keys:.
    from collections import OrderedDict
    checkpoint = OrderedDict()
    for key in temp.keys():
        new_key = key.replace("module.","")
        checkpoint[new_key]=temp[key]

    model.load_state_dict(checkpoint, strict=True)
    print("Pre-trained model loaded succesfully")

    #model = nn.DataParallel(model).cuda()
    #cudnn.benchmark = True #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    model = model.cuda()
    # ==================================================
    # ================== Inference =====================
    if not os.path.exists(dst):
        os.mkdir(dst)

    # switch to evaluate mode
    model.eval()

    for i, video in enumerate(loader):
        video_dst = os.path.join(dst, activity[i])
        if not os.path.exists(video_dst):
            os.mkdir(video_dst)

        count = 0
        state = None # Initially no hidden state
        print("Initiating inference for video {}".format(i))
        for j, (frame_names, clip) in enumerate(video):
            clip = Variable(clip.type(dtype).t(), requires_grad=False)
            for idx in range(clip.size()[0]):
                # Compute output
                (hidden, cell), saliency_map = model.forward(clip[idx], state)
                hidden = Variable(hidden.data)
                cell = Variable(cell.data)
                state = (hidden, cell)
                count+=1
                utils.save_image(saliency_map.data.cpu(), os.path.join(video_dst, frame_names[idx][0]))
                # j*clip_length+idx because we are iterating over batches of images and +1 because we don't want to start naming at 0

        print("Done with video '{}'".format(activity[i]))

    print("Inference Done")

main()
