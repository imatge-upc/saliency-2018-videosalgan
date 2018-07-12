import torch
import os
import cv2
import datetime
import numpy as np
from torch.utils import data
from PIL import Image
from torchvision import utils


# The DataLoader for our specific video datataset with extracted frames
class DHF1K_frames(data.Dataset):

  def __init__(self, split, clip_length, number_of_videos, val_perc = 0.15, frame_size = (256, 192)):

        self.frame_size = frame_size
        self.cl = clip_length
        self.frames_path = "/imatge/lpanagiotis/work/DHF1K/predictions" # in our case it's salgan saliency maps
        self.gt_path = "/imatge/lpanagiotis/work/DHF1K/maps" #ground truth


        # A list to keep all video lists of salgan predictions, which will be our dataset.
        self.video_list = []

        # A list to keep all the dictionaries of ground truth - saliency map pairings for each video
        self.gts_list = []

        start = datetime.datetime.now().replace(microsecond=0) # Gives accurate human readable time, rounded down not to include too many decimals
        for i in range(1, number_of_videos+1): #700 videos in DHF1K

            # The way the folder structure is organized allows to simply iterate over the range of the number of total videos.
            gt_files = os.listdir(os.path.join(self.gt_path,str(i)))
            frame_files = os.listdir(os.path.join(self.frames_path,str(i)))
            #print("for video {} the frames are {}".format(i, len(frame_files))) # This is correct

            # a list of lists
            self.video_list.append(frame_files)
            # Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the).
            gt_files_sorted = sorted(gt_files, key = lambda x: int(x.split(".")[0]) )
            frame_files_sorted = sorted(frame_files, key = lambda x: int(x.split(".")[0]) )
            pack = zip(gt_files_sorted, frame_files_sorted)

            # Make dictionary where keys are the saliency maps and values are the ground truths
            gt_frame_pairings = {}
            for gt, frame in pack:
                gt_frame_pairings[frame] = gt

            self.gts_list.append(gt_frame_pairings)
            if i%50==0:
                print("Pairings related to video {} organized.".format(i))
                print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))



        # pack a list of data with the corresponding list of ground truths
        # Split the dataset to validation and training
        limit = int(round(val_perc*len(self.video_list)))
        if split == "validation":
          self.video_list = self.video_list[:limit]
          self.gts_list = self.gts_list[:limit]
          self.first_video_no = 1 #This needs to be specified to find the correct directory in our case. It will be different for each split since these directories signify videos.
        elif split == "train":
          self.video_list = self.video_list[limit:]
          self.gts_list = self.gts_list[limit:]
          self.first_video_no = limit+1




  def __len__(self):
        'Denotes the total number of samples'
        return len(self.video_list)

  def __getitem__(self, video_index):

        'Generates one sample of data'
        # Select sample video (frame list), in our case saliency map list
        frames = self.video_list[video_index]
        gts = self.gts_list[video_index]

        # Due to the split in train and validation we need to add this number to the video_index to find the correct video (to match the files in the path with the video list the training part uses)
        true_index = self.first_video_no + video_index #this matches the correct video number

        data = []
        gt = []
        packed = []
        for i, frame in enumerate(frames):

          # Load data and get ground truth
          path_to_frame = os.path.join(self.frames_path, str(true_index), frame)
          print(path_to_frame) #path is good
          X = cv2.imread(path_to_frame, cv2.IMREAD_GRAYSCALE)

          # Normalize
          X = X.astype(np.float32)
          X = (X - X.min())/(X.max()-X.min())

          """
          norm_X = np.zeros((size_ima[0], size_ima[1]))
          norm_X = cv2.normalize(X, dst=norm_X, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) #normalize is destroying the image
          #cv2.normalize(X, X, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) #normalize is destroying the image
          """
          cv2.imwrite("X1.png",X*255)
          X = cv2.resize(X, self.frame_size, interpolation=cv2.INTER_AREA)
          cv2.imwrite("X2.png",X*255)
          X = np.expand_dims(X, 0)
          cv2.imwrite("X3.png",X[0]*255)


          # There is only one channel and python would automatically omit it, we need to avoid that.
          #X = Image.fromarray(X)
          #if self.transforms is not None:
          #    X = self.transforms(X)
          #X = torch.from_numpy(X).unsqueeze(0) # Use unsqueeze because grayscale has 1 channel and it would be omitted but pytorch expects to see it. (something weird happens here probably, get back to it)

          path_to_gt = os.path.join(self.gt_path, str(true_index), gts[frame])
          y = cv2.imread(path_to_gt, cv2.IMREAD_GRAYSCALE)

          # Normalize
          y = y.astype(np.float32)
          y = (y - y.min())/(y.max()-y.min())

          y = cv2.resize(y, self.frame_size, interpolation=cv2.INTER_AREA)
          y = np.expand_dims(y, 0) # There is only one channel and python would automatically omit it, we need to avoid that.
          #y = Image.fromarray(y)
          #if self.transforms is not None:
          #    y = self.transforms(y)
          #y = torch.from_numpy(y).unsqueeze(0)

          data.append(X)
          gt.append(y)

          """
          except RuntimeError:

            print("Unexpected error at frame {} video {}".format(i, video_index))
            print("Index that parses video list: {}\n Index that specifies video in path: {}\nPath {}\n Length frames {}".format(index, true_index, path_to_frame, len(frames)))
            print("Index that parses the list needs to be 1 less than index that specifies video in the path. Check that this is correct and that length of frames matches the true length.")
          """

          if (i+1)%self.cl == 0 or i == (len(frames)-1):
            #print(np.array(data).shape) #looks okay
            data_tensor = torch.FloatTensor(data) #bug was actually here
            gt_tensor = torch.FloatTensor(gt)
            packed.append((data_tensor,gt_tensor)) # pack a list of data with the corresponding list of ground truths
            data = []
            gt = []
            print(data_tensor[0])
            utils.save_image((data_tensor[0]*255).type(torch.ByteTensor), "./test/dt{}.png".format(i))
            utils.save_image((gt_tensor[0]*255).type(torch.ByteTensor), "./test/gt{}.png".format(i))
            print(data_tensor[0]*255)
            exit()


        return packed




