from model import *
from datasets import TestKodakDataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.ToPILImage()

def plot_image(img_numpy_array, title : str):
    plt.title(title)
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
    plt.imshow(img_numpy_array)
    plt.show()

def test(path_model : str, img_indexes : list[int], stega=False, p=.0):
    with torch.no_grad():
        if stega:
            model_stega = ImageCompressorSteganography(p)
            load_model(model_stega,path_model)
            net_stega = model_stega.cuda()
            net_stega.eval()
        model_cover = ImageCompressor()
        load_model(model_cover,path_model)
        net_cover = model_cover.cuda()
        test_dataset = TestKodakDataset(data_dir='../data1/liujiaheng/data/kodak')
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=1)
        net_cover.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        
        for batch_idx, input in enumerate(test_loader):
            if batch_idx in img_indexes:
                input = input.cuda()
                cover_image, mse_loss, bpp = net_cover(input)
                
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

                precover_img_np = input[0].permute(1,2,0).cpu().numpy()
                cover_img_np = cover_image[0].permute(1,2,0).cpu().numpy()
                plot_image(precover_img_np,"precover image")
                plot_image(cover_img_np,"cover image")
                if stega:
                    stega_image, _, _ = net_stega(input)
                    stega_img_np = stega_image[0].permute(1,2,0).cpu().numpy()
                    plot_image(stega_img_np,"stega image")
        
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt