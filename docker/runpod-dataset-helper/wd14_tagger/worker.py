# autotag/worker.py
import os
from PIL import Image
from .jobs import autotag_jobs, autotag_lock
from .tagger import WD14Tagger
from .utils import iter_images  # yields image paths from a directory

# Fixed models
FIXED_MODELS = {
    "wd-v1-4-moat-tagger-v2": {"threshold": 0.35, "character_threshold": 0.85},
    "wd-convnext-tagger-v3": {"threshold": 0.35, "character_threshold": 0.85}
}

def autotag_worker(job_id, path, mode="all"):
    """
    Worker for autotagging using tag_multi for all images.
    - Directory: iterates images, calls tag_multi per image
    - Single image: also calls tag_multi
    """
    try:
        if os.path.isdir(path):
            images = list(iter_images(path))
        elif os.path.isfile(path):
            images = [path]
        else:
            raise ValueError(f"Invalid path: {path}")

        total = len(images)
        with autotag_lock:
            autotag_jobs[job_id]["total"] = total

        for img_path in images:
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if mode == "missing" and os.path.exists(txt_path):
                continue

            image = Image.open(img_path).convert("RGB")
            tags = WD14Tagger.tag_multi(image, FIXED_MODELS, replace_underscore=True)

            caption = ", ".join(tags)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)

            with autotag_lock:
                autotag_jobs[job_id]["done"] += 1
                autotag_jobs[job_id]["results"][img_path] = caption

        with autotag_lock:
            autotag_jobs[job_id]["status"] = "completed"

    except Exception as e:
        with autotag_lock:
            autotag_jobs[job_id]["status"] = "error"
            autotag_jobs[job_id]["error"] = str(e)
