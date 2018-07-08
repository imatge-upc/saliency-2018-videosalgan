import cv2
import os
import numpy as np
import time
import pickle

sample = "/imatge/lpanagiotis/video-salgan-2018/src/test.png"
#sm_directory = "/imatge/lpanagiotis/work/DHF1K_extracted/predictions"

img = cv2.imread(sample)

print(type(img))
cv2.imshow("img", img)
cv2.waitKey()
