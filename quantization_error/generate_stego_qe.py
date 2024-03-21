import sys
sys.path.append("../iclr_17_compression/")
sys.path.append("../Tools")
from model import ImageCompressorSteganography_QE, ImageCompressorSteganography_QE_modified_cost
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import os
from math import log
from tqdm import tqdm
from my_utils import ternary_entropy, pgm_to_tensor

def precover_to_stego(
    precover_folder_path : str, 
    output_path : str,
    model_path : str, 
    p : float = .0,
    use_modified_cost: bool = False,
    file_id_start : int = 1, 
    file_id_stop : int = -1):
    """
    Generates cover or stego images using ImageCompressor model on a range of precover images
    Args:
        precover_folder_path (str): path to the precovers' folder
        output_path (str): path where to store cover/stego images
        p (float): insertion rate = 2*p (if steganography is active)
        model_path (str): path to the parameters of the ImageCompressor model
        file_id_start (int, optional): Lower bound of the range of images.
        file_id_stop (int, optional): Upper bound of the range of images.
    """
    if use_modified_cost:
        model = ImageCompressorSteganography_QE_modified_cost(p)
    else:
        model = ImageCompressorSteganography_QE(p)
    
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
        for file in tqdm(files_to_convert):
            precover_t = pgm_to_tensor(precover_folder_path + "/" + file)
            file_name, _ = os.path.splitext(file)
            compressed_img_t, mse_loss, bpp = net(precover_t,stega=stega)
            
            torch.save(compressed_img_t, output_path + "/" + file_name + ".pt")

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
    parser.add_argument("-c","--cost",type=bool,default=False,help="If True, use ImageCompressorSteganography_QE_modified_cost")

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output,exist_ok=True)
    if args.probability > 0.:
        info = f"p = {args.probability}\nH3(p) = {ternary_entropy(args.probability)}"
        with open(args.output + "/" + "_info.txt","w") as f:
            f.write(info)
    precover_to_stego(args.name, args.output, args.model, args.probability,args.cost, args.begin, args.end)