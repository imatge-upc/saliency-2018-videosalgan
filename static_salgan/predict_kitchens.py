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
        print os.path.join(path_to_images, curr_file + '.jpg')
        img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.jpg'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) #image is loaded correctly
        """
        cv2.imshow("img",img)
        cv2.waitKey(1)
        exit()
        """
        predict(model=model_to_test, image_stimuli=img, name=curr_file, path_output_maps=path_output_maps)


def main():
    # Create network
    model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    # Here need to specify the epoch of model sanpshot
    load_weights(model.net['output'], path='gen_', epochtoload=90)
    #In my case I need to do it for every video folder, so

    src = "/imatge/lpanagiotis/work/Epic-Kitchens/object_detection_images"
    dst = "/imatge/lpanagiotis/work/Epic-Kitchens/saliency_maps"
    for x in ["train", "test"]:
        print("Now predicting frames for {}".format(x))
        root_path = os.path.join(src, x)
        people = os.listdir(root_path)

        for person in people:
            person_path = os.path.join(src, x, person)
            videos = os.listdir(person_path)

            for video in videos: #Every video is a directory
                # Define our source file
                source_video = os.path.join(person_path, video)

                # Define our destination directory
                frames = os.listdir(source_video)
                destination_dir = os.path.join(dst, x, person, video)
                if not os.path.exists(destination_dir):
                    os.mkdir(destination_dir)
                print("destination is {}".format(destination_dir))
                test(path_to_images=source_video, path_output_maps=destination_dir, model_to_test=model)

if __name__ == "__main__":
    main()
