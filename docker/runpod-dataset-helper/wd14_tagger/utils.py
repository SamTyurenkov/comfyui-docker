# autotag/utils.py
import os

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

def iter_images(path):
    if os.path.isfile(path):
        yield path
        return

    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                yield os.path.join(root, f)