#https://docs.python.org/3.5/library/tarfile.html#tarfile-objects

import tarfile
import os

src = "/imatge/lpanagiotis/projects/saliency/epic-kitchens/object_detection_images"

dst = "/imatge/lpanagiotis/work/Epic-Kitchens/object_detection_images"

# Folder Structure corresponds to : train/test -> person -> videos

for x in ["train", "test"]:
    print("Now extracting frames for {}".format(x))
    root_path = os.path.join(src, x)
    people = os.listdir(root_path)

    for person in people:
        person_path = os.path.join(src, x, person)
        videos = os.listdir(person_path)

        for video in videos:

            # Define our source tar file
            source_tar_file = os.path.join(person_path, video)

            # Define our destination directory
            video_dir = video.split(".")[0] #remove the tar extension
            destination_dir = os.path.join(dst, x, person, video_dir)
            if not os.path.exists(destination_dir):
                os.mkdir(destination_dir)

            # Extract the frames from the tar file to the destination folder
            video_file = tarfile.open(name = source_tar_file)

            video_file.extractall(path = destination_dir)
            print("Video {} extracted to destination folder".format(video))
