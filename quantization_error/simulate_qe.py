import sys
sys.path.append("../iclr_17_compression")
from model import *
from test_model import plot_tensor, plot_np_array
from datasets import TestKodakDataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../Tools")
from my_utils import pgm_to_tensor, plot_tensor, plot_np_array

def apply_qe_to_kodak(path_model : str, img_indexes : list[int], plot=False, stega=False, p=.0, plot_precover=False,plot_hist=False):
    with torch.no_grad():
        model = ImageCompressorSteganography_QE(p)
        load_model(model,path_model)
        net = model.cuda()
        net.eval()
        if stega:
            sumBpp_stega = 0
            sumPsnr_stega_cover = 0

        test_dataset = TestKodakDataset(data_dir='../data1/liujiaheng/data/kodak')
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        
        for batch_idx, input in enumerate(test_loader):
            if batch_idx in img_indexes:
                input = input.cuda()
                cover_image, mse_loss, bpp = net(input,stega=False)
                
                mse_loss = torch.mean((cover_image - input).pow(2))
                mse_loss, bpp = torch.mean(mse_loss), torch.mean(bpp)
                psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
                sumBpp += bpp
                sumPsnr += psnr
                msssim = ms_ssim(cover_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
                msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
                sumMsssimDB += msssimDB
                sumMsssim += msssim
                cnt += 1

                if plot_precover:
                    precover_img_np = input[0].permute(1,2,0).cpu().numpy()
                    plot_np_array(precover_img_np,"precover image")
                
                if plot:
                    cover_img_np = cover_image[0].permute(1,2,0).cpu().numpy()
                    plot_np_array(cover_img_np,"cover image")

                if stega:
                    stega_image, _, bpp_stega = net(input,stega=True,plot_hist=plot_hist)

                    mse_loss_stega_cover = torch.mean((stega_image - cover_image).pow(2))
                    psnr_stega_cover = 10 * (torch.log(1. / mse_loss_stega_cover) / np.log(10))
                    bpp_stega = torch.mean(bpp_stega)

                    sumBpp_stega += bpp_stega
                    sumPsnr_stega_cover += psnr_stega_cover

                    if plot:
                        stega_img_np = stega_image[0].permute(1,2,0).cpu().numpy()
                        plot_np_array(stega_img_np,"stego image")
        
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt

        if stega:
            sumBpp_stega /= cnt
            sumPsnr_stega_cover /= cnt
            return sumBpp_stega.cpu().item(), sumPsnr_stega_cover.cpu().item()

        return sumBpp.cpu().item(), sumPsnr.cpu().item()

def network_output_to_numpy(net_out) -> np.ndarray:
    """Converts output of ImageCompressor networks to numpy arrays

    Args:
        net_out (Tensor): tensor of an image produced by the ImageCompressor network (or one of its variations). It has the shape (1,channels,height,width)

    Returns:
        np.ndarray: numpy array of an image with shape (height,width,channels) and values between 0 and 1
    """
    assert len(net_out.shape) == 3 or len(net_out.shape) == 4
    if len(net_out.shape) == 4: net_out = net_out[0] # the first axis is useless when there are 4 dimensions
    to_pil = transforms.ToPILImage()
    img = np.array(to_pil(net_out.cpu()))
    return img

def diff_cover_stego(cover_t, stego_t):
    """
    Args:
        cover_t (Tensor): cover tensor of an image (output of ImageCompressor model)
        stego_t (Tensor): stego tensor of an image (output of ImageCompressor model)

    Returns:
        np.ndarray: difference between cover image and stego image
    """
    cover_np = network_output_to_numpy(cover_t)
    stego_np  = network_output_to_numpy(stego_t)
    return cover_np.astype(int) - stego_np.astype(int)

def apply_qe_to_bossbase_img(
    precover_folder_path : str,
    model_path : str,
    idx: int,
    p : float = 0.,
    plot_cover_stego=False,
    plot_diff=False,
    figsize=None):
    """Apply ImageCompressorSteganography_QE to an image from bossbase

    Args:
        precover_folder_path (str): path to bossbase pgm images folder
        model_path (str): path to the weights of the model
        idx (int): number in the name of the pgm file to which we apply the model
        p (float, optional): half insertion rate (used to get the size of the message). Defaults to 0..
        plot_cover_stego (bool, optional): _description_. Defaults to False.
        plot_diff (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        model = ImageCompressorSteganography_QE(p)
        load_model(model,model_path)
        net = model.cuda()
        net.eval()
        
        img_t = pgm_to_tensor(precover_folder_path + "/" + str(idx) + ".pgm")
        cover_t, _, _ = net(img_t,stega=False)
        stego_t, _, _, modification_rate = net(img_t,stega=True,return_modification_rate=True,print_positive_proba=True)

        if plot_cover_stego:
            plot_tensor(cover_t,"cover")
            plot_tensor(stego_t,"stego")
        mse_loss = torch.mean((img_t - stego_t).pow(2))
        psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
        diff = diff_cover_stego(cover_t,stego_t)[:,:,0] # we work with grayscale img so only 1 channel is necessary
        if plot_diff:
            plot_np_array(diff,title="cover - stego",colorbar=True,figsize=figsize)
        return psnr.cpu().numpy(), diff, modification_rate

def apply_naive_to_bossbase_img(
    precover_folder_path : str,
    model_path : str,
    idx: int,
    p : float = 0.,
    plot_cover_stego=False,
    plot_diff=False,
    figsize=None):
    """Apply ImageCompressorSteganography to an image from bossbase

    Args:
        precover_folder_path (str): path to bossbase pgm images folder
        model_path (str): path to the weights of the model
        idx (int): number in the name of the pgm file to which we apply the model
        p (float, optional): half insertion rate (used to get the size of the message). Defaults to 0..
        plot_cover_stego (bool, optional): _description_. Defaults to False.
        plot_diff (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        model = ImageCompressorSteganography(p)
        load_model(model,model_path)
        net = model.cuda()
        net.eval()
        
        img_t = pgm_to_tensor(precover_folder_path + "/" + str(idx) + ".pgm")
        cover_t, _, _ = net(img_t,stega=False)
        stego_t, _, _ = net(img_t,stega=True)

        if plot_cover_stego:
            plot_tensor(cover_t,"cover")
            plot_tensor(stego_t,"stego")
        mse_loss = torch.mean((img_t - stego_t).pow(2))
        psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
        diff = diff_cover_stego(cover_t,stego_t)[:,:,0] # we work with grayscale img so only 1 channel is necessary
        if plot_diff:
            plot_np_array(diff,title="cover - stego",colorbar=True,figsize=figsize)
        return psnr.cpu().numpy(), diff