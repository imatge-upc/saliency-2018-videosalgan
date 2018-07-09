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
#from tensorboardX import SummaryWriter

from data_loader import DHF1K_frames

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor


learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
start_epoch = 1
epochs = 10
plot_every = 5
load_model = False
pretrained_model = './SalConvLSTM.pt'
clip_length = 5 #out of memory at 10! with 2 gpus. Works with 7 but occasionally produces error as well.
number_of_videos = 5


#writer = SummaryWriter('./log') #Tensorboard

# Parameters
params = {'batch_size': 1, #this is actually the number of videos
          'num_workers': 4,
          'pin_memory': True}

def main(params = params):

    # =================================================
    # ================ Data Loading ===================

    #Expect Error if either validation size or train size is 1
    train_set = DHF1K_frames(
        number_of_videos = number_of_videos,
        clip_length = clip_length,
        split = "train") #add a parameter node = training or validation
    print("Size of train set is {}".format(len(train_set)))
    val_set = DHF1K_frames(
        number_of_videos = number_of_videos,
        clip_length = clip_length,
        split = "validation")
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
    model = ConvLSTMCell(use_gpu=True, input_size=1, hidden_size=128, kernel_size=3)
    criterion = nn.BCEWithLogitsLoss()

    if load_model:
        # Need to check out how to work with dataparallel
        checkpoint = torch.load(pretrained_model)['state_dict']
        model.load_state_dict(checkpoint, strict=True)
        print("Pre-trained model loaded succesfully")

    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936

    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)


    # =================================================
    # ================== Training =====================

    if dtype == torch.cuda.ByteTensor:
        model.cuda()
        criterion = criterion.cuda()

    train_losses = []
    val_losses = []
    print("Training started at : {}".format(datetime.datetime.now().replace(microsecond=0)))
    for epoch in range(start_epoch, epochs):
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # cuda error occurs here
        val_loss = validate(val_loader, model, criterion, epoch)


        if epoch % plot_every == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        print("Epoch {}/{} done with train loss {} and validation loss {}\n".format(epoch, epochs, train_loss, val_loss))

    print("Training finished at : {} \n Now saving..".format(datetime.datetime.now().replace(microsecond=0)))

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
        for j, (clip, gtruths) in enumerate(video):

            # Reset Gradients
            optimizer.zero_grad()

            # Squeeze out the video dimension
            clip = Variable(clip.type(dtype)).squeeze(0)
            gtruths = Variable(gtruths.type(dtype)).squeeze(0)

            #print(frame.size()) #works! torch.Size([5, 1, 360, 640])
            #print(gt.size()) #works! torch.Size([5, 1, 360, 640])

            # Keep the hidden state after each batch of frames and initialize next batch with it
            for idx in range(clip.size()[0]):
                #print(clip[idx].size()) needs unsqueeze
                # Compute output
                state, saliency_map = model.forward(clip[idx].unsqueeze(0), state)
                # Compute loss
                loss = criterion(saliency_map, gtruths[idx].unsqueeze(0))
                # Keep score
                frame_losses.append(loss.data)
                # Accumulate gradients
                if idx == (clip.size()[0]-1):
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)

            # Repackage to avoid backpropagating further through time
            (hidden, cell) = state
            hidden = Variable(hidden.data)
            cell = Variable(cell.data)
            state = (hidden, cell)

            #hidden = Variable(hidden.data)
            #cell = Variable(cell.data)


            # Compute gradient and do optimizing step
            optimizer.step()

            #state = (hidden, cell)
            #hidden = Variable(hidden.data, requires_grad=True)
            #cell = Variable(cell.data, requires_grad=True)

            if (j+1)%20==0:
                print('Training Loss is {} at batch {} in video {}'.format(loss.data, j+1, i))

        #writer.add_scalar('Train/Loss', mean(frame_losses), i)
        end = datetime.datetime.now().replace(microsecond=0)
        print('Epoch: {}\tVideo: {}\t Training Loss: {}\t Time elapsed: {}\t'.format(epoch, i, mean(frame_losses), end-start))
        video_losses.append(mean(frame_losses))

    return (mean(video_losses))


def validate(val_loader, model, criterion, epoch):

    # switch to evaluate mode
    model.eval()

    video_losses = []
    for i, video in enumerate(val_loader):
        frame_losses = []
        state = None # Initially no hidden state
        for j, (clip, gtruths) in enumerate(video):

            # squeeze out the video dimension
            clip = Variable(clip.type(dtype), requires_grad=False).squeeze(0)
            gtruths = Variable(gtruths.type(dtype), requires_grad=False).squeeze(0)


            for idx in range(clip.size()[0]):
                #print(clip[idx].size()) needs unsqueeze
                # Compute output
                (hidden, cell), saliency_map = model.forward(clip[idx].unsqueeze(0), state)

                hidden = Variable(hidden.data)
                cell = Variable(cell.data)
                state = (hidden, cell)

                # Compute loss
                loss = criterion(saliency_map, gtruths[idx].unsqueeze(0))
                # Keep score
                frame_losses.append(loss.data)

        print('Epoch: {}\tVideo {}\t Validation Loss {}\t'.format(
            epoch, i+1, mean(frame_losses)))
        video_losses.append(mean(frame_losses))
        #writer.add_scalar('Val/Loss', mean(frame_losses), i)

    return(mean(video_losses))

if __name__ == '__main__':
    main()

    #utils.save_image(saliency_map.data.cpu(), "test.png")


