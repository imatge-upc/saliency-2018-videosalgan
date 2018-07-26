import os
import cv2
import numpy as np
import datetime
from gaze_io_sample import parse_gtea_gaze

gaze_path = "/imatge/lpanagiotis/projects/saliency/GTEA_Gaze/gaze_data"
frames_path = "/imatge/lpanagiotis/work/GTEA_Gaze/frames"

dst = "/imatge/lpanagiotis/work/GTEA_Gaze/ground_truths"
if not os.path.exists(dst):
    os.mkdir(dst)

# gaze type
gaze_type = ['untracked', 'fixation', 'saccade', 'unknown', 'truncated']

file_names = os.listdir(frames_path)

"""
files = {}
for file_name in file_names:
    name = file_name.split("_")[0]
    if name in files:
        files[name]+=1
    else:
        files[name]=0
"""
#example_name = "P10-R01-PastaSalad"
#example_name = "OP02-R07-Pizza"
#example_name = "P26-R05-Cheeseburger"
start = datetime.datetime.now().replace(microsecond=0)

print("Commencing production of maps at time {}".format(start))
for i, name in enumerate(file_names):
    #name = example_name
    test_file_01 = os.path.join(gaze_path, name+".txt")
    #print(test_file_01)
    #print(name)
    folder_of_frames = sorted(os.listdir(os.path.join(frames_path, name)), key = lambda x: int(x.split(".")[0]))
    number_of_frames = len(folder_of_frames)

    if os.path.exists(test_file_01):
        test_data_01 = parse_gtea_gaze(test_file_01)
        # print the loaded gaze
        print('Loaded gaze data from {:s}'.format(test_file_01))
        """
        print('Frame {:d}, Gaze Point ({:02f}, {:0.2f}), Gaze Type: {:s}'.format(
            1000,
            test_data_01[1000, 0],
            test_data_01[1000, 1],
            gaze_type[int(test_data_01[1000, 2])]
        ))
        """

        print("Number of frames in my data : {} \n Number of frames in their gaze data : {}".format(number_of_frames, test_data_01.shape[0]))

    else:
        import pickle
        error_message = "No gaze data provided for {}".format(name)
        print(error_message)
        with open('errors.txt', 'wb') as handle:
            pickle.dump(error_message, handle, protocol=pickle.HIGHEST_PROTOCOL)
        continue

    path_to_folder = os.path.join(dst, name)
    if not os.path.exists(path_to_folder):
        os.mkdir(path_to_folder)

    if number_of_frames < test_data_01.shape[0]:
        iterator = number_of_frames
    else:
        iterator = test_data_01.shape[0]

    for j in range(iterator):
        frame_name = folder_of_frames[j]
        #print(frame_name)
        frame = cv2.imread(os.path.join(frames_path, name, frame_name))
        #print(frame.shape)
        gt = np.zeros(frame.shape)
        x = test_data_01[j, 1]
        y = test_data_01[j, 0]
        #print((x,y))

        """
        This produces index error
        x = test_data_01[j, 0]
        y = test_data_01[j, 1]
        px = x*gt.shape[1]
        py = y*gt.shape[0]
        """
        px = x*gt.shape[0]
        py = y*gt.shape[1]
        #print(px,py)
        frame[int(px), int(py)]=255
        gt[int(px), int(py)]=255
        path_to_output = os.path.join(path_to_folder, "{}.png".format(j))
        cv2.imwrite( path_to_output, gt )

    print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))
