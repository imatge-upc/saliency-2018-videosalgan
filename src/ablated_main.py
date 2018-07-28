import cv2
import os
import datetime
import numpy as np
from modules.clstm import Conv
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

"""
Based on the superior BCE-based loss compared with MSE,
we also explored the impact of computing the content loss over
downsampled versions of the saliency map. This technique re-
duces the required computational resources at both training and
test times and, as shown in Table 3, not only does it not decrease
performance, but it can actually improve it. Given this results,
we chose to train SalGAN on saliency maps downsampled by
a factor 1/4, which in our architecture corresponds to saliency
maps of 64 × 48. - Salgan paper
"""

frame_size = 64 # 5 times lower than the original
learning_rate = 0.00001 #
decay_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
start_epoch = 1
epochs = 10
plot_every = 1
load_model = False
pretrained_model = './Ablated_Model_No_LSTM.pt'
clip_length = 1
number_of_videos = 700 # DHF1K offers 700 labeled videos, the other 300 are held back by the authors


writer = SummaryWriter('./log') #Tensorboard

# Parameters
params = {'batch_size': 1, # number of videos / batch, I need to implement padding if I want to do more than 1, but with DataParallel it's quite messy
          'num_workers': 4,
          'pin_memory': True}

def main(params = params):

    # =================================================
    # ================ Data Loading ===================

    #Expect Error if either validation size or train size is 1
    train_set = DHF1K_frames(
        number_of_videos = number_of_videos,
        clip_length = clip_length,
        split = "train",
        transforms = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor()
            ]))
    print("Size of train set is {}".format(len(train_set)))

    val_set = DHF1K_frames(
        number_of_videos = number_of_videos,
        clip_length = clip_length,
        split = "validation",
        transforms = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor()
            ]))
    print("Size of validation set is {}".format(len(val_set)))

    #print(len(train_set[0]))
    #print(len(train_set[1]))

    train_loader = data.DataLoader(train_set, **params)
    val_loader = data.DataLoader(val_set, **params)


    # =================================================
    # ================ Define Model ===================

    # Using same kernel size as they do in the DHF1K paper
    # Amaia uses default hidden size 128
    # input size is 1 since we have grayscale images
    model = Conv(use_gpu=True, input_size=1, filter_size=128, kernel_size=3)
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.KLDivLoss() # this produces 0 training loss only

    if load_model:
        # Load stored model:
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

    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=4, min_lr=0.0000001)


    # =================================================
    # ================== Training =====================

    if dtype == torch.cuda.ByteTensor:
        model.cuda()
        criterion = criterion.cuda()

    train_losses = []
    val_losses = []
    starting_time = datetime.datetime.now().replace(microsecond=0)
    print("Training started at : {}".format(starting_time))
    for epoch in range(start_epoch, epochs):
        #adjust_learning_rate(optimizer, epoch, decay_rate)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # cuda error occurs here
        val_loss = validate(val_loader, model, criterion, epoch)
        # if validation has not improved for a certain amount of epochs, reduce the learning rate:
        scheduler.step(val_loss)

        if epoch % plot_every == 0:
            train_losses.append(train_loss.cpu())
            val_losses.append(val_loss.cpu())

        print("Epoch {}/{} done with train loss {} and validation loss {}\n".format(epoch, epochs, train_loss, val_loss))

    print("Training started at {} and finished at : {} \n Now saving..".format(starting_time, datetime.datetime.now().replace(microsecond=0)))

    # ===================== #
    # ======  Saving ====== #

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.cpu().state_dict(),
        'optimizer' : optimizer.state_dict()
        }, 'SalConvLSTM.pt')
    """
    hyperparameters = {
        'momentum' : momentum,
        'weight_decay' : weight_decay,
        'learning_rate' : learning_rate,
        'decay_rate' : decay_rate,
        'epochs' : epochs,
        'batch_size' : batch_size
    }
    """

    to_plot = {
        'epoch_ticks': list(range(start_epoch, epochs, plot_every)),
        'train_losses': train_losses,
        'val_losses': val_losses
        }
    with open('to_plot.pkl', 'wb') as handle:
        pickle.dump(to_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)


mean = lambda x : sum(x)/len(x)

def adjust_learning_rate(optimizer, epoch, decay_rate=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (decay_rate ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer, epoch):

    # Switch to train mode
    model.train()

    video_losses = []
    print("Now commencing epoch {}".format(epoch))
    for i, video in enumerate(train_loader):
        #print(type(video))
        frame_losses = []
        start = datetime.datetime.now().replace(microsecond=0)
        print("Number of clips for video {} : {}".format(i,len(video)))
        state = None # Initially no hidden state
        for j, (frame, gt) in enumerate(video):

            # Reset Gradients
            optimizer.zero_grad()

            # Squeeze out the video dimension
            # [video_batch, clip_length, channels, height, width]
            # After transpose:
            # [clip_length, video_batch, channels, height, width]
            frame = Variable(frame.type(dtype).t())
            gt = Variable(gt.type(dtype).t())

            frame.squeeze(0)
            gt.squeeze(0)
            # Compute output
            saliency_map = model.forward(frame)
            # Compute loss
            loss = criterion(saliency_map, gt)
            # Keep score
            frame_losses.append(loss.data)

            loss.backward()
            optimizer.step()

            if i == 2 and j == 0:
                utils.save_image(frame[idx], "./test/frame.png")
                utils.save_image(saliency_map, "./test/sm.png")

            # Compute gradient and do optimizing step

            #state = (hidden, cell)
            #hidden = Variable(hidden.data, requires_grad=True)
            #cell = Variable(cell.data, requires_grad=True)

            """
            if (j+1)%20==0:
                print('Training Loss: {} Batch/Clip: {}/{} '.format(loss.data, i, j+1))
            """

        end = datetime.datetime.now().replace(microsecond=0)
        print('Epoch: {}\tVideo: {}\t Training Loss: {}\t Time elapsed: {}\t'.format(epoch, i, mean(frame_losses), end-start))
        video_losses.append(mean(frame_losses))

    return (mean(video_losses))


def validate(val_loader, model, criterion, epoch):

    # switch to evaluate mode
    model.eval()

    video_losses = []
    print("Now running validation..")
    for i, video in enumerate(val_loader):
        frame_losses = []
        state = None # Initially no hidden state
        for j, (frame, gt) in enumerate(video):

            frame = Variable(frame.type(dtype).t(), requires_grad=False)
            gt = Variable(gt.type(dtype).t(), requires_grad=False)


            frame.squeeze(0)
            gt.squeeze(0)
            # Compute output
            saliency_map = model.forward(frame)
            # Compute loss
            loss = criterion(saliency_map, gt)
            # Keep score
            frame_losses.append(loss.data)

        video_losses.append(mean(frame_losses))

    return(mean(video_losses))

if __name__ == '__main__':
    main()

    #utils.save_image(saliency_map.data.cpu(), "test.png")


