import os
import cv2
import numpy as np
from gaze_io_sample import parse_gtea_gaze

gaze_path = "/imatge/lpanagiotis/projects/saliency/GTEA_Gaze/gaze_data"
frames_path = "/imatge/lpanagiotis/work/GTEA_Gaze/frames"

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
example_name = "P10-R01-PastaSalad"
example_name = "OP02-R07-Pizza"

for name in file_names:
    name = example_name
    test_file_01 = os.path.join(gaze_path, name+".txt")
    print(test_file_01)
    print(name)
    folder_of_frames = os.listdir(os.path.join(frames_path, name))
    print(len(folder_of_frames))

    if os.path.exists(test_file_01):
        test_data_01 = parse_gtea_gaze(test_file_01)
        # print the loaded gaze
        print('Loaded gaze data from {:s}'.format(test_file_01))
        print('Frame {:d}, Gaze Point ({:02f}, {:0.2f}), Gaze Type: {:s}'.format(
            1000,
            test_data_01[1000, 0],
            test_data_01[1000, 1],
            gaze_type[int(test_data_01[1000, 2])]
        ))

        print(test_data_01.shape)
        print(test_data_01[0])
        #print(test_data_01[test_data_01[:,2]=="fixation"].shape)

    else:
        print("No gaze data provided for {}".format(name))

    for i in range(10):
        frame_name = folder_of_frames[i]
        frame = cv2.imread(os.path.join(frames_path, name, frame_name))
        print(frame.shape)
        gt = np.zeros(frame.shape)
        x = test_data_01[i, 0]
        y = test_data_01[i, 1]
        print((x,y))
        px = x*gt.shape[1]
        py = y*gt.shape[0]
        print(px,py)
        frame[int(px), int(py)]=255
        gt[int(px), int(py)]=255
        cv2.imwrite("test_frame{}.png".format(i),frame)
        cv2.imwrite("test_gt{}.png".format(i),gt)


    break
