import cv2
import os
import numpy as np
import pickle
import datetime

from gaze_io_sample import parse_gtea_gaze
from joblib import Parallel, delayed

gaze_path = "/imatge/lpanagiotis/projects/saliency/GTEA_Gaze/gaze_data"
pred_path = "/imatge/lpanagiotis/work/GTEA_Gaze/predictions"

# gaze type
gaze_type = ['untracked', 'fixation', 'saccade', 'unknown', 'truncated']

file_names = os.listdir(pred_path)

continue_calculations = False

if continue_calculations:
    with open('metrics.txt', 'rb') as handle:
        final_metric_list = pickle.load(handle)
else:
    final_metric_list = []

start = datetime.datetime.now().replace(microsecond=0)

def main():

    for i, name in enumerate(file_names):
        #name = example_name
        gaze_file = os.path.join(gaze_path, name+".txt")
        #print(gaze_file)
        #print(name)
        folder_of_predictions = sorted(os.listdir(os.path.join(pred_path, name)), key = lambda x: int(x.split(".")[0]))
        number_of_frames = len(folder_of_predictions)

        if os.path.exists(gaze_file):
            gaze_data = parse_gtea_gaze(gaze_file)
            # print the loaded gaze
            print('Loaded gaze data from {:s}'.format(gaze_file))
            """
            print('Frame {:d}, Gaze Point ({:02f}, {:0.2f}), Gaze Type: {:s}'.format(
                1000,
                gaze_data[1000, 0],
                gaze_data[1000, 1],
                gaze_type[int(gaze_data[1000, 2])]
            ))
            """

            print("Number of frames in my data : {} \n Number of frames in their gaze data : {}".format(number_of_frames, gaze_data.shape[0]))

        else:
            import pickle
            error_message = "No gaze data provided for {}".format(name)
            print(error_message)
            with open('errors.txt', 'wb') as handle:
                pickle.dump(error_message, handle, protocol=pickle.HIGHEST_PROTOCOL)
            continue

        if number_of_frames < gaze_data.shape[0]:
            iterator = number_of_frames
        else:
            iterator = gaze_data.shape[0]

        NSS_list = Parallel(n_jobs=8)(delayed(inner_worker)(j, folder_of_predictions, gaze_data, name) for j in range(iterator)) #run 8 frames simultaneously

        nss_mean = np.mean( NSS_list )

        print("For video {} the metrics are:".format(i))
        #print("AUC-JUDD is {}".format(aucj_mean))
        #print("AUC-SHUFFLED is {}".format(aucs_mean))
        print("NSS is {}".format(nss_mean))
        print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))
        print("==============================")

        final_metric_list.append(nss_mean)

        with open('metrics.txt', 'wb') as handle:
            pickle.dump(final_metric_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Aucj = np.mean([y[0] for y in final_metric_list])
    #Aucs = np.mean([y[1] for y in final_metric_list])
    Nss = np.mean(final_metric_list)

    print("Final average of metrics is:")
    #print("AUC-JUDD is {}".format(Aucj))
    #print("AUC-SHUFFLED is {}".format(Aucs))
    print("NSS is {}".format(Nss))

# The directories are named 1-1000 so it should be easy to iterate over them
def inner_worker(j, folder_of_predictions, gaze_data, name, pred_path=pred_path): #packed should be a list of tuples (annotation, prediction)


    pred_name = folder_of_predictions[j]
    #print(pred_name)
    saliency_map = cv2.imread(os.path.join(pred_path, name, pred_name))
    #print(pred.shape)
    x = gaze_data[j, 1]
    y = gaze_data[j, 0]
    px = x*saliency_map.shape[0]
    py = y*saliency_map.shape[1]

    ground_truth = (int(px),int(py))

    #saliency_map_norm = normalize_map(saliency_map)

    # Calculate metrics
    #AUC_JUDD = auc_judd(saliency_map_norm, ground_truth)
    #AUC_SHUF = auc_shuff(saliency_map_norm, ground_truth, ground_truth)
    NSS = nss(saliency_map, ground_truth)

    return (NSS)

def nss(s_map,gt_coords):

    s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)
    NSS = np.mean(s_map_norm[gt_coords[0],gt_coords[1]])

    return (NSS)

main()
