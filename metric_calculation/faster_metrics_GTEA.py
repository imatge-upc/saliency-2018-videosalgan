from salience_metrics import auc_judd, auc_shuff, cc, nss, similarity, normalize_map, discretize_gt
"""
DHF1K paper: "we  employ  five  classic  met-rics,  namely  Normalized  Scanpath  Saliency  (NSS),  Sim-ilarity Metric (SIM), Linear Correlation Coefficient (CC),AUC-Judd (AUC-J), and shuffled AUC (s-AUC).""
"""
import cv2
import os
import numpy as np
import pickle
import datetime

gt_directory = "/imatge/lpanagiotis/work/GTEA_Gaze/ground_truths"
sm_directory = "/imatge/lpanagiotis/work/GTEA_Gaze/predictions"

continue_calculations = False

if continue_calculations:
    with open('metrics.txt', 'rb') as handle:
        final_metric_list = pickle.load(handle)
else:
    final_metric_list = []

# The directories are named 1-1000 so it should be easy to iterate over them
def inner_worker(i, packed, gt_path, sm_path): #packed should be a list of tuples (annotation, prediction)

    gt, sm = packed
    ground_truth = cv2.imread(os.path.join(gt_path, gt),cv2.IMREAD_GRAYSCALE)
    saliency_map = cv2.imread(os.path.join(sm_path, sm),cv2.IMREAD_GRAYSCALE)

    #print(ground_truth.shape)
    #print(saliency_map.shape)
    #ground_truth = discretize_gt(ground_truth)

    ground_truth = cv2.dilate(ground_truth,(5,5), iterations = 1)
    ground_truth = cv2.GaussianBlur(ground_truth, (15, 15), 1)

    saliency_map_norm = normalize_map(saliency_map) # The functions are a bit haphazard. Some have normalization within and some do not.
    #print(ground_truth[ground_truth!=0])
    # Calculate metrics
    #AUC_JUDD = auc_judd(saliency_map_norm, ground_truth)
    #AUC_SHUF = auc_shuff(saliency_map_norm, ground_truth, ground_truth)
    NSS = nss(saliency_map_norm, ground_truth)

    return (NSS)

GTEA_vids = os.listdir(sm_directory)
start = datetime.datetime.now().replace(microsecond=0)
for vid in GTEA_vids:

    gt_path = os.path.join(gt_directory, vid)
    sm_path = os.path.join(sm_directory, vid)

    gt_files = os.listdir(gt_path)
    sm_files = os.listdir(sm_path)
    #Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the).
    gt_files_sorted = sorted(gt_files, key = lambda x: int(x.split(".")[0]) )
    sm_files_sorted = sorted(sm_files, key = lambda x: int(x.split(".")[0]) )
    pack = zip(gt_files_sorted, sm_files_sorted)
    print("Files related to video {} sorted.".format(vid))
##
    ##https://stackoverflow.com/questions/35663498/how-do-i-return-a-matrix-with-joblib-python
    from joblib import Parallel, delayed
    metric_list = Parallel(n_jobs=8)(delayed(inner_worker)(n, packed, gt_path, sm_path) for n, packed in enumerate(pack)) #run 8 frames simultaneously

    #aucj_mean = np.mean([x[0] for x in metric_list])
    #aucs_mean = np.mean([x[1] for x in metric_list])
    nss_mean = np.mean( metric_list )

    print("For video {} the metrics are:".format(vid))
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

