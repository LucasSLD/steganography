from model import *
from datasets import TestKodakDataset
from PIL import Image
from torchvision import transforms

logger = logging.getLogger("ImageCompression")
transform = transforms.ToPILImage()

def test(model : str):
    with torch.no_grad():
        model = ImageCompressor()
        net = model.cuda()
        test_dataset = TestKodakDataset(data_dir='../data1/liujiaheng/data/kodak')
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            input = input.cuda()
            clipped_recon_image, mse_loss, bpp = net(input)
            mse_loss = torch.mean((clipped_recon_image - input).pow(2))
            mse_loss, bpp = \
                torch.mean(mse_loss), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            # logger.info("Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(bpp, psnr, msssim, msssimDB))
            cnt += 1
            transform(input[0]).show()
            transform(clipped_recon_image[0]).show()
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        # logger.info("Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))