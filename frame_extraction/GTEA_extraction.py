import os
import cv2

from frame_extraction import find_fps

CAP_PROP_POS_MSEC=0 #Current position of the video file in milliseconds.
def frame_iterator(filename, max_num_frames):
  """Uses OpenCV to iterate over all frames of filename at a given frequency.

  Args:
    filename: Path to video file (e.g. mp4)
    every_ms: The duration (in milliseconds) to skip between frames.
    max_num_frames: Maximum number of frames to process, taken from the
      beginning of the video.

  Yields:
    RGB frame with shape (image height, image width, channels)
  """
  fps = find_fps(filename)
  print(fps)
  import numpy as np
  every_ms = np.floor(1000/fps) #rounding seems to be important. In the first video, without rounding I get 356, but with rounding I get the correct number of 450.
  print(every_ms)

  video_capture = cv2.VideoCapture()
  if not video_capture.open(filename):
    print >> sys.stderr, 'Error: Cannot open video file ' + filename
    return
  last_ts = -99999  # The timestamp of last retrieved frame.
  num_retrieved = 0

  while num_retrieved < max_num_frames:
    # Skip frames
    while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
      if not video_capture.read()[0]:
        return

    last_ts = video_capture.get(CAP_PROP_POS_MSEC)
    has_frames, frame = video_capture.read()
    if not has_frames:
        break
    yield frame
    num_retrieved += 1


# I want to extract the frames from the originally downloaded videos and put them in my directory so that they will not be backed up. I will also copy the annotations to make it easier to use later.
original_directory = "/projects/saliency/GTEA_Gaze/raw_videos/"
video_files = os.listdir(original_directory)

extracted_frames_directory   = "/imatge/lpanagiotis/work/GTEA_Gaze/temp/"
if not os.path.exists(extracted_frames_directory):
    os.mkdir(extracted_frames_directory)

for video_file in video_files:
    # The video is named something like "OP02-R07-Pizza.mp4"
    name_of_file = video_file.split(".")[0]
    # My working directory path. Each video will have a folder of its own.
    path_to_extracted_frames = os.path.join(extracted_frames_directory, str(name_of_file))
    if not os.path.exists(path_to_extracted_frames):
        os.mkdir(path_to_extracted_frames)

    count = 0
    for frame in frame_iterator(os.path.join(original_directory, video_file), max_num_frames=100000 ):
        count+=1
        if count == 1:
            continue
        path_to_new_frame = os.path.join(path_to_extracted_frames, str(count)+".png")
        cv2.imwrite(path_to_new_frame, frame)
        """
        import matplotlib.pyplot as plt
        plt.imshow(frame)
        plt.show()
        break
        """
        #It works!

    print("Frames successfully extracted from video {}".format(name_of_file))

    if count > 2:
        break



