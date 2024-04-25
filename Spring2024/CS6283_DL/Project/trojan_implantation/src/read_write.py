import cv2
import numpy as np
from PIL import Image


def load_image(img_path) -> np.ndarray:
    return Image.open(img_path)


def save_img(img_path: str, img):
    img.save(img_path)

