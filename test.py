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

def test(path_model : str, img_indexes : list[int]):
    with torch.no_grad():
        model = ImageCompressor()
        load_model(model,path_model)
        net = model.cuda()
        test_dataset = TestKodakDataset(data_dir='../data1/liujiaheng/data/kodak')
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=1)
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        
        for batch_idx, input in enumerate(test_loader):
            if batch_idx in img_indexes:
                input = input.cuda()
                clipped_recon_image, mse_loss, bpp = net(input)

                mse_loss = torch.mean((clipped_recon_image - input).pow(2))
                mse_loss, bpp = torch.mean(mse_loss), torch.mean(bpp)
                psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
                sumBpp += bpp
                sumPsnr += psnr
                msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
                msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
                sumMsssimDB += msssimDB
                sumMsssim += msssim
                cnt += 1

                img_ori_np = input[0].permute(1,2,0).cpu().numpy()
                img_reco_np = clipped_recon_image[0].permute(1,2,0).cpu().numpy()
                plot_image(img_ori_np,"original image")
                plot_image(img_reco_np,"reconstructed image")
        
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt