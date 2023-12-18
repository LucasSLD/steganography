from model import *
from datasets import TestKodakDataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.ToPILImage()

def plot_image(img_numpy_array, title : str = None):
    if title is not None: plt.title(title)
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
    plt.imshow(img_numpy_array)
    plt.show()

def plot_tensor(tensor, title : str = None):
    assert tensor.shape[0] == 1 # we plot 4D tensors which in reality represents 3D images (I don't know what is the purpose of the first axis)
    array = tensor[0].permute(1,2,0).cpu().numpy()
    plot_image(array, title)

def test(path_model : str, img_indexes : list[int], plot=False, stega=False, p=.0):
    with torch.no_grad():
        model = ImageCompressorSteganography(p)
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

                if plot:
                    precover_img_np = input[0].permute(1,2,0).cpu().numpy()
                    cover_img_np = cover_image[0].permute(1,2,0).cpu().numpy()
                    plot_image(precover_img_np,"precover image")
                    plot_image(cover_img_np,"cover image")

                if stega:
                    stega_image, _, bpp_stega = net(input,stega=True)

                    mse_loss_stega_cover = torch.mean((stega_image - cover_image).pow(2))
                    psnr_stega_cover = 10 * (torch.log(1. / mse_loss_stega_cover) / np.log(10))
                    bpp_stega = torch.mean(bpp_stega)

                    sumBpp_stega += bpp_stega
                    sumPsnr_stega_cover += psnr_stega_cover

                    if plot:
                        stega_img_np = stega_image[0].permute(1,2,0).cpu().numpy()
                        plot_image(stega_img_np,"stega image")
        
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt

        if stega:
            sumBpp_stega /= cnt
            sumPsnr_stega_cover /= cnt
            return sumBpp_stega.cpu().item(), sumPsnr_stega_cover.cpu().item()

        return sumBpp.cpu().item(), sumPsnr.cpu().item()