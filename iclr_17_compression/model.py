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

    def forward(self, input_image, stega=False):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        feature = self.Encoder(input_image)
        batch_size = feature.size()[0]
        compressed_feature_renorm = torch.round(feature)

        if stega:
            shape = compressed_feature_renorm.shape[1:]
            message_length = ternary_entropy(self.p)
            quantization_error = torch.sub(feature,compressed_feature_renorm).flatten()
            rho_P1 = (1 - 2*quantization_error).cpu().numpy()
            rho_M1 = (1 + 2*quantization_error).cpu().numpy()
            p_change_P1, p_change_M1 = es.compute_proba(rho_P1,rho_M1,message_length,shape[0]*shape[1]*shape[2])
            feature_QE = es.process(compressed_feature_renorm.cpu().numpy().flatten(),p_change_P1,p_change_M1)
            compressed_feature_renorm[0] = torch.Tensor(feature_QE).reshape(shape[0],shape[1],shape[2])
            print(torch.mean((compressed_feature_renorm == torch.round(feature)).float()))

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