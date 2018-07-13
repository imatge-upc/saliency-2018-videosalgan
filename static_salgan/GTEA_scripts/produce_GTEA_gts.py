import os
import numpy as np
from gaze_io_sample import parse_gtea_gaze

root_path = "/imatge/lpanagiotis/projects/saliency/GTEA_Gaze"

# gaze type
gaze_type = ['untracked', 'fixation', 'saccade', 'unknown', 'truncated']

img_path = os.path.join(root_path, "Images")
file_names = os.listdir(img_path)

files = {}
for file_name in file_names:
    name = file_name.split("_")[0]
    if name in files:
        files[name]+=1
    else:
        files[name]=0

for name in files.keys():
    test_file_01 = os.path.join(root_path, "gaze_data", name+".txt")
    print(test_file_01)
    print(files[name])

    if os.path.exists(test_file_01):
        test_data_01 = parse_gtea_gaze(test_file_01)
        # print the loaded gaze
        print 'Loaded gaze data from {:s}'.format(test_file_01)
        print 'Frame {:d}, Gaze Point ({:02f}, {:0.2f}), Gaze Type: {:s}'.format(
            1000,
            test_data_01[1000, 0],
            test_data_01[1000, 1],
            gaze_type[int(test_data_01[1000, 2])]
        )

        print(test_data_01.shape)
        #print(test_data_01[test_data_01[:,2]=="fixation"].shape)

    else:
        print("No gaze data provided for {}".format(name))
