import os
import cv2

from frame_extraction import frame_iterator


# I want to extract the frames from the originally downloaded videos and put them in my directory so that they will not be backed up. I will also copy the annotations to make it easier to use later.
original_directory           = "/projects/saliency/GTEA_Gaze/Videos/"
video_files      = os.listdir(original_directory)

extracted_frames_directory   = "/imatge/lpanagiotis/work/GTEA_Gaze/frames/"

for video_file in video_files:
    # The video is named something like "001.mpg"
    number_of_file = video_file.split(".")[0]
    # My working directory path. Each video will have a folder of its own.
    path_to_extracted_frames = os.path.join(extracted_frames_directory, str(number_of_file))
    if not os.path.exists(path_to_extracted_frames):
        os.mkdir(path_to_extracted_frames)

    count = 0
    for frame in frame_iterator(os.path.join(original_directory, video_file), max_num_frames=10000 ):
        count+=1

        path_to_new_frame = os.path.join(path_to_extracted_frames, str(count)+".png")
        cv2.imwrite(path_to_new_frame, frame)
        """
        import matplotlib.pyplot as plt
        plt.imshow(frame)
        plt.show()
        break
        """
        #It works!

    print("Frames successfully extracted from video {}".format(number_of_file))





