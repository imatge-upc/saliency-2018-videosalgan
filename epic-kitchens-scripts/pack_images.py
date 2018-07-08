#https://docs.python.org/3.5/library/tarfile.html#tarfile-objects

import tarfile
import os
import datetime

src = "/imatge/lpanagiotis/work/Epic-Kitchens/saliency_maps"

dst = "/imatge/lpanagiotis/projects/saliency/public_html/epic-kitchens/saliency_maps"

# Folder Structure corresponds to : train/test -> person -> videos

def main():

    for x in ["train", "test"]:
        print("Commencing packing of videos for {}..".format(x))
        root_path = os.path.join(src, x)
        people = os.listdir(root_path)

        for person in people:
            person_path = os.path.join(src, x, person)
            videos = os.listdir(person_path)

            print("Now packing videos of person {}..".format(person))
            start = datetime.datetime.now().replace(microsecond=0)

            for video in videos:

                # Define our source folder, containing images
                source_folder = os.path.join(person_path, video)

                # Define our destination directory
                destination_dir = os.path.join(dst, x, person)

                tar_file = os.path.join(destination_dir,video+".tar")

                make_tarfile(tar_file = tar_file, source_file = source_folder)

            end = datetime.datetime.now().replace(microsecond=0)
            print("Person {} done, time elapsed:".format(person, end-start))

def make_tarfile(tar_file, source_file):
    with tarfile.open(tar_file, "w:gz") as tar:
        tar.add(source_file)

if __name__=="__main__":
    main()
