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
from data_loader import DHF1K_frames

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor

clip_length = 20 #with 20 clips the loss seems to reach zero very fast
number_of_videos = 1 # DHF1K offers 700 labeled videos, the other 300 are held back by the authors
frame_size = (256, 192)
pretrained_model = './SalConvLSTM.pt'

# Parameters
params = {'batch_size': 1, # number of videos / batch, I need to implement padding if I want to do more than 1, but with DataParallel it's quite messy
          'num_workers': 4,
          'pin_memory': True}

def main():

    # =================================================
    # ================ Data Loading ===================

    #Expect Error if either validation size or train size is 1
    dataset = DHF1K_frames(
        number_of_videos = number_of_videos,
        clip_length = clip_length,
        split = "train",
        transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
            ])
        )
         #add a parameter node = training or validation
    print("Size of test set is {}".format(len(dataset)))

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

    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936

    # ==================================================
    # ================== Inference =====================

    if not os.path.exists("./test"):
        os.mkdir("./test")

    # switch to evaluate mode
    model.eval()

    for i, video in enumerate(loader):
        state = None # Initially no hidden state
        for j, (clip, gtruths) in enumerate(video):

            clip = Variable(clip.type(dtype).t(), requires_grad=False)
            gtruths = Variable(gtruths.type(dtype).t(), requires_grad=False)

            for idx in range(clip.size()[0]):
                #print(clip[idx].size()) needs unsqueeze
                # Compute output
                (hidden, cell), saliency_map = model.forward(clip[idx], state)
                hidden = Variable(hidden.data)
                cell = Variable(cell.data)
                state = (hidden, cell)

            utils.save_image(clip[idx].data.cpu(), "./test/input{}.png".format(j))
            utils.save_image(saliency_map.data.cpu(), "./test/output{}.png".format(j))

main()
