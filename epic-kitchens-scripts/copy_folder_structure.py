#https://www.pythoncentral.io/how-to-recursively-copy-a-directory-folder-in-python/

from shutil import copytree, ignore_patterns

src = "/imatge/lpanagiotis/projects/saliency/epic-kitchens/object_detection_images"

dst = "/imatge/lpanagiotis/work/Epic-Kitchens"
dst = "/imatge/lpanagiotis/projects/saliency/public_html/epic-kitchens"
#dst directory must not exist
#dst = dst + "/object_detection_images" # Here I will extract the tar files
dst = dst + "/saliency_maps" # Here I will produce the output
#dst = dst + "/dynamic_saliency_maps" # Here I will produce the output


copytree(src, dst, ignore = ignore_patterns('*.tar')) #copy folder structure to our public-to-be directory ignoring the files, (all files are tar)

