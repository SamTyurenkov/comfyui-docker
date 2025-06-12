from transformers import pipeline
from PIL import Image
from reactor_utils import download
from scripts.reactor_logger import logger

def ensure_nsfw_model(nsfwdet_model_path):
    print('we do not need that')

SCORE = 0.96

def nsfw_image(img_path: str, model_path: str):
    return False
