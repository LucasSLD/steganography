from PIL import Image
import numpy as np
from model import ImageCompressorSteganography
import torch
import os

def read_pgm(file_name):
    with open(file_name, 'rb') as f:
        # Skip header
        f.readline()
        dimensions = f.readline().split()
        width, height = int(dimensions[0]), int(dimensions[1])
        max_val = int(f.readline())

        # Read the image data
        img = np.fromfile(f, dtype=np.uint8, count=width * height).reshape((height, width))
    return img


def precover_to_cover(precover_batch_path : str, model : ImageCompressorSteganograĥy, file_id : int):
    # files = os.
    with torch.no_grad():
        net = model.cuda()
        net.eval()
        