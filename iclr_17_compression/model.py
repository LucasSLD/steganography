import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from models import *
sys.path.append("../quantization_error/")
from embedding_simulator import Embedding_simulator as es
from math import log
import matplotlib.pyplot as plt
sys.path.append("../Tools/")
from my_utils import plot_tensor

def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class ImageCompressor(nn.Module):
    def __init__(self, out_channel_N=128):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net_17(out_channel_N=out_channel_N)
        self.Decoder = Synthesis_net_17(out_channel_N=out_channel_N)
        self.bitEstimator = BitEstimator(channel=out_channel_N)
        self.out_channel_N = out_channel_N

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        feature = self.Encoder(input_image)
        batch_size = feature.size()[0]
        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        recon_image = self.Decoder(compressed_feature_renorm)
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        # def feature_probs_based_sigma(feature, sigma):
        #     mu = torch.zeros_like(sigma)
        #     sigma = sigma.clamp(1e-10, 1e10)
        #     gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        #     probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        #     total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
        #     return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator(z + 0.5) - self.bitEstimator(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = iclr18_estimate_bits_z(compressed_feature_renorm)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        
        return clipped_recon_image, mse_loss, bpp_feature


class ImageCompressorSteganography(nn.Module):
    def __init__(self, p=0.0, out_channel_N=128):
        super(ImageCompressorSteganography, self).__init__()
        self.Encoder = Analysis_net_17(out_channel_N=out_channel_N)
        self.Decoder = Synthesis_net_17(out_channel_N=out_channel_N)
        self.bitEstimator = BitEstimator(channel=out_channel_N)
        self.out_channel_N = out_channel_N
        self.p = p

    def forward(self, input_image, stega=False):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        feature = self.Encoder(input_image)
        batch_size = feature.size()[0]
        compressed_feature_renorm = torch.round(feature)

        if stega:
            probabilities = torch.Tensor([self.p, 1 - 2*self.p, self.p]) # probability 2*p of modifying the value of a feature
            shape = compressed_feature_renorm.shape[1:]
            values = torch.multinomial(probabilities,shape[0]*shape[1]*shape[2],replacement=True) - 1
            values = values.reshape((shape[0],shape[1],shape[2])).cuda()
            compressed_feature_renorm[0].add_(values)

        recon_image = self.Decoder(compressed_feature_renorm)
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator(z + 0.5) - self.bitEstimator(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = iclr18_estimate_bits_z(compressed_feature_renorm)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])

        return clipped_recon_image, mse_loss, bpp_feature

def ternary_entropy(p: float) -> float:
    """
    Ternary entropy
    Args:
        p (float): half insertion rate, probability p of adding 1 and proba p of substracting 1

    Returns:
        float: ternary entropy for given p
    """
    return -2*p*log(p,2) - (1-2*p)*log(1-2*p,2)

class ImageCompressorSteganography_QE(nn.Module): # Image compressor model with steganography performed with quantization error method
    def __init__(self, p=0.0, out_channel_N=128):
        super(ImageCompressorSteganography_QE, self).__init__()
        self.Encoder = Analysis_net_17(out_channel_N=out_channel_N)
        self.Decoder = Synthesis_net_17(out_channel_N=out_channel_N)
        self.bitEstimator = BitEstimator(channel=out_channel_N)
        self.out_channel_N = out_channel_N
        self.p = p

    def forward(self, input_image, stega=False, plot_hist=False, return_modification_rate=False, print_positive_proba=False):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        feature = self.Encoder(input_image) # unquantized coefficients
        batch_size = feature.size()[0]
        compressed_feature_renorm = torch.round(feature) # quantized coefficients

        if stega:
            shape = compressed_feature_renorm.shape[1:]
            message_length = ternary_entropy(self.p) * shape[0]*shape[1]*shape[2]
            quantization_error = torch.sub(feature, compressed_feature_renorm).flatten()
            
            rho_P1 = (1 - 2*quantization_error).cpu().numpy()
            rho_M1 = (1 + 2*quantization_error).cpu().numpy()
            p_change_P1, p_change_M1 = es.compute_proba(rho_P1,rho_M1,message_length,shape[0]*shape[1]*shape[2])
            
            if plot_hist:
                plt.hist(quantization_error.cpu().numpy(),50)
                plt.grid()
                plt.yscale("log")
                plt.title("Quantization error histogram")
                plt.show()
                
                plt.hist(rho_P1,50)
                plt.yscale("log")
                plt.title("rhoP1")
                plt.show()

                plt.hist(rho_M1,50)
                plt.yscale("log")
                plt.title("rhoM1")
                plt.show()

                plt.hist(p_change_M1,50)
                plt.grid()
                plt.title("Histogram of -1 probabilities")
                plt.yscale("log")
                plt.show()
                
                plt.hist(p_change_P1,50)
                plt.grid()
                plt.yscale("log")
                plt.title("Histogram of +1 probabilities")
                plt.show()
                print("number of coef in latent space:",p_change_M1.shape[0])
                print("number of positive proba of -1:",np.count_nonzero(p_change_M1))
                print("number of positive proba of +1:",np.count_nonzero(p_change_P1))
            

            compressed_feature_np_flat = compressed_feature_renorm.cpu().numpy().flatten()
            feature_QE = es.process(compressed_feature_np_flat,p_change_P1,p_change_M1) # flattened coefficients in latent space with modification using quantization error
            compressed_feature_renorm[0] = torch.Tensor(feature_QE).reshape(shape[0],shape[1],shape[2])

            # psnr between precover and stego
            # pseudo_cover_feature = torch.round(feature).cpu().numpy().flatten() # where cover != stego we take unquantized coefficients
            # pseudo_cover_feature[pseudo_cover_feature != feature_QE] = feature.cpu().numpy().flatten()[pseudo_cover_feature != feature_QE]
            # pseudo_cover = torch.zeros_like(compressed_feature_renorm)
            # pseudo_cover[0] = torch.Tensor(pseudo_cover_feature).reshape(shape[0],shape[1],shape[2])
            # pseudo_cover_img = self.Decoder(pseudo_cover)
            # stego_img = self.Decoder(compressed_feature_renorm)
            # plot_tensor(pseudo_cover_img,"pseudo cover")
            # plot_tensor(stego_img,"stego")
            # mse = torch.mean((pseudo_cover_img - stego_img)**2)
            # psnr = 10 * torch.log(1./mse) / np.log(10)
            # print("psnr = ",psnr)
            
            if print_positive_proba: 
                # print all > 0 probabilities of change (+1 or -1)
                def get_index_value(a):
                    """
                    Returns a sorted 2D array where each row is a couple (index, value)
                    Args:
                        a (ndarray): 1D array 

                    Returns:
                        ndarray: 2D array sorted by rows based on the value
                    """
                    return np.array(sorted(
                            list(zip(np.nonzero(a)[0],a[a != 0])),
                            key= lambda x: x[1],
                            reverse=True))

                print("1st column = index in vector of probability ; 2nd colmun = value of the proba of change")
                print("+1 positive probabilities:\n",get_index_value(p_change_P1))
                print("=========================")
                print("-1 positive probabilities:\n",get_index_value(p_change_M1))

                # checking if the modifications in latent space happened where the probabilities of change where higher
                print("",get_index_value(feature_QE - compressed_feature_np_flat))

        recon_image = self.Decoder(compressed_feature_renorm)
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator(z + 0.5) - self.bitEstimator(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = iclr18_estimate_bits_z(compressed_feature_renorm)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        
        if return_modification_rate and stega:
            modification_rate = np.count_nonzero(feature_QE - compressed_feature_np_flat)/feature_QE.shape[0] # nb of modified coef/ total number of coef
            return clipped_recon_image, mse_loss, bpp_feature, modification_rate

        return clipped_recon_image, mse_loss, bpp_feature

class ImageCompressorSteganography_QE_modified_cost(nn.Module): # Image compressor model with steganography performed with quantization error method and non-zero cost for adding 0 to quantized coefficients
    def __init__(self, p=0.0, out_channel_N=128):
        super(ImageCompressorSteganography_QE_modified_cost, self).__init__()
        self.Encoder = Analysis_net_17(out_channel_N=out_channel_N)
        self.Decoder = Synthesis_net_17(out_channel_N=out_channel_N)
        self.bitEstimator = BitEstimator(channel=out_channel_N)
        self.out_channel_N = out_channel_N
        self.p = p

    def forward(self, input_image, stega=False, plot_hist=False, return_modification_rate=False, print_positive_proba=False):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        feature = self.Encoder(input_image) # unquantized coefficients
        batch_size = feature.size()[0]
        compressed_feature_renorm = torch.round(feature) # quantized coefficients

        if stega:
            shape = compressed_feature_renorm.shape[1:]
            message_length = ternary_entropy(self.p) * shape[0]*shape[1]*shape[2]
            quantization_error = torch.sub(feature, compressed_feature_renorm).flatten()
            
            qe_np  = quantization_error.cpu().numpy()
            rho_P1 = 1 - qe_np
            rho_M1 = 1 + qe_np
            rho_0  = np.abs(qe_np)
            p_change_P1, p_change_M1 = es.compute_proba_0(rho_P1,rho_M1,rho_0,message_length,shape[0]*shape[1]*shape[2])
            
            if plot_hist:
                # plt.hist(quantization_error.cpu().numpy(),50)
                # plt.grid()
                # plt.yscale("log")
                # plt.title("Quantization error histogram")
                # plt.show()
                
                plt.hist(rho_P1,50)
                plt.yscale("log")
                plt.title("rhoP1")
                plt.show()

                plt.hist(rho_M1,50)
                plt.yscale("log")
                plt.title("rhoM1")
                plt.show()

                plt.hist(rho_0,50)
                plt.yscale("log")
                plt.title("rho0")
                plt.show()

                plt.hist(p_change_M1,50)
                plt.grid()
                plt.title("Histogram of -1 probabilities")
                plt.yscale("log")
                plt.show()
                
                plt.hist(p_change_P1,50)
                plt.grid()
                plt.yscale("log")
                plt.title("Histogram of +1 probabilities")
                plt.show()
                print("number of coef in latent space:",p_change_M1.shape[0])
                print("number of positive proba of -1:",np.count_nonzero(p_change_M1))
                print("number of positive proba of +1:",np.count_nonzero(p_change_P1))
            

            compressed_feature_np_flat = compressed_feature_renorm.cpu().numpy().flatten()
            feature_QE = es.process(compressed_feature_np_flat,p_change_P1,p_change_M1) # flattened coefficients in latent space with modification using quantization error
            compressed_feature_renorm[0] = torch.Tensor(feature_QE).reshape(shape[0],shape[1],shape[2])
            
            if print_positive_proba: 
                # print all > 0 probabilities of change (+1 or -1)
                def get_index_value(a):
                    """
                    Returns a sorted 2D array where each row is a couple (index, value)
                    Args:
                        a (ndarray): 1D array 

                    Returns:
                        ndarray: 2D array sorted by rows based on the value
                    """
                    return np.array(sorted(
                            list(zip(np.nonzero(a)[0],a[a != 0])),
                            key= lambda x: x[1],
                            reverse=True))

                print("1st column = index in vector of probability ; 2nd colmun = value of the proba of change")
                print("+1 positive probabilities:\n",get_index_value(p_change_P1))
                print("=========================")
                print("-1 positive probabilities:\n",get_index_value(p_change_M1))

                # checking if the modifications in latent space happened where the probabilities of change where higher
                print("",get_index_value(feature_QE - compressed_feature_np_flat))

        recon_image = self.Decoder(compressed_feature_renorm)
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator(z + 0.5) - self.bitEstimator(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = iclr18_estimate_bits_z(compressed_feature_renorm)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        
        if return_modification_rate and stega:
            modification_rate = np.count_nonzero(feature_QE - compressed_feature_np_flat)/feature_QE.shape[0] # nb of modified coef/ total number of coef
            return clipped_recon_image, mse_loss, bpp_feature, modification_rate

        return clipped_recon_image, mse_loss, bpp_feature