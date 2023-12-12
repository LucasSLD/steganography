import sys
sys.path.append("../../iclr_17_compression/")
from model import ImageCompressor, ImageCompressorSteganography
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
    model_path : str, 
    p : float = .0, 
    file_id_start : int = 1, 
    file_id_stop : int = -1):
    """
    Generates cover images using ImageCompressor model on a range of precover images
    Args:
        precover_folder_path (str): path to the precovers' folder
        cover_folder_path (str): path where to store cover images
        p (float): insertion rate = 2*p (if steganography is active)
        model_path (str): path to the parameters of the ImageCompressor model
        file_id_start (int, optional): Lower bound of the range of images.
        file_id_stop (int, optional): Upper bound of the range of images.
    """
    model = ImageCompressorSteganography(p)
    load_model(model,model_path)
    files = os.listdir(precover_folder_path)
    assert p <= .5 # insertion rate cannot be higher than 1
    assert file_id_start-1 < len(files)
    assert file_id_stop-1 < len(files)
    if file_id_stop != -1: assert file_id_stop >= file_id_start
    
    files_to_convert = files[file_id_start-1:file_id_stop if file_id_stop!=-1 else len(files)]
    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()
    with torch.no_grad():
        net = model.cuda()
        net.eval()

        stega  = p > .0
        for file in files_to_convert:
            precover = read_pgm(precover_folder_path + "/" + file)
            precover_t = to_tensor(precover).repeat(1,3,1,1).cuda()
            cover_t, mse_loss, bpp = net(precover_t,stega=stega)
            cover = to_pil(cover_t[0])
            file_name, _ = os.path.splitext(file)
            cover.save(cover_folder_path + "/" + file_name + ".jpg")
            print("hey!")


if __name__ == "__main__":
    import argparse
    from model import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",required=True,help="Path to the folder of precover images")
    parser.add_argument("-o","--output",required=True,help="Path to the folder of cover images")
    parser.add_argument("-b","--begin",default=1, type=int,help="Number of the first image to convert")
    parser.add_argument("-e","--end",default=-1,type=int,help="Number of the last image to convert")
    parser.add_argument("-m","--model",required=True,help="Path to the ImageCompressor model to use")
    parser.add_argument("-p","--probability",type=float,default=.0,help="half insertion rate")

    args = parser.parse_args()

    precover_to_cover(args.name, args.output, args.model, args.probability, args.begin, args.end)