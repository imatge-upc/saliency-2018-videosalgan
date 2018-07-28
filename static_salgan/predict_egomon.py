import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
from constants import *
from models.model_bce import ModelBCE


def test(path_to_images, path_output_maps, model_to_test=None):
    list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*'))]
    # Load Data
    list_img_files.sort()
    for curr_file in tqdm(list_img_files, ncols=20):
        print os.path.join(path_to_images, curr_file + '.png')
        img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) #Changed to .png
        predict(model=model_to_test, image_stimuli=img, name=curr_file, path_output_maps=path_output_maps)
        exit

def main():
    # Create network
    model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    # Here need to specify the epoch of model sanpshot
    load_weights(model.net['output'], path='gen_', epochtoload=90)

    src = "/imatge/lpanagiotis/projects/saliency/public_html/2016-egomon/video_clean"
    dst = "/imatge/lpanagiotis/work/Egomon/temp"

    folders = os.listdir(src)
    for folder in folders:
        frames_folder = "frames_" + folder
        src_path = os.path.join(src, folder, frames_folder)

        print("Now predicting for {}".format(folder))
        new_directory = os.path.join(dst, folder)
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        # Here need to specify the path to images and output path
        test(path_to_images = src_path, path_output_maps = new_directory, model_to_test = model)

if __name__ == "__main__":
    main()
