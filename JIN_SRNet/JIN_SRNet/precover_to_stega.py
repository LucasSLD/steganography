from model import ImageCompressor
from torchvision import transforms
from PIL import Image
import numpy as np
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


def precover_to_cover(
    precover_folder_path : str, 
    cover_folder_path : str, 
    model : ImageCompressor, 
    file_id_start : int, 
    file_id_stop : int):
    
    files = os.listdir(precover_folder_path)
    assert file_id_start-1 < len(files)
    assert file_id_stop-1 < len(files)
    assert file_id_stop >= file_id_start
    
    files_to_convert = files[file_id_start-1:file_id_stop]
    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()
    with torch.no_grad():
        net = model.cuda()
        net.eval()

        for file_name in files_to_convert:
            precover = read_pgm(precover_folder_path + file_name)
            precover_t = to_tensor(precover)
            cover_t, mse_loss, bpp = net(precover_t)
            cover = to_pil(cover_t[0])
            cover.save(cover_folder_path + file_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",required=True,help="Path to the folder of precover images")
    parser.add_argument("-o","--output",required=True,help="Path to the folder of cover images")
    parser.add_argument("-b","--begin",default=1, type=int,help="Number of first image to convert")