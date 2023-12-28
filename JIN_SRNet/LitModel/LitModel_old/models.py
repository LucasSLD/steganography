import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from functools import partial
from efficientnet_pytorch import EfficientNet
import numpy as np
from torch import nn
import timm
import numpy as np
import types
import torch
from SRNet import SRNet, OneHotSRNet
from timm.models.layers import Swish as SwishMe
zoo_params = {
    
    'srnet': {
        'fc_name': 'fc',
        'fc': nn.Linear(in_features=512, out_features=2, bias=True),
        'init_op': partial(SRNet, in_channels=3, nclasses=2)
    },
    
    'srnet_OH': {
        'fc_name': 'fc',
        'fc': nn.Linear(in_features=512+32, out_features=2, bias=True),
        'init_op': partial(OneHotSRNet, in_channels=3, nclasses=2, T=5)
    },
    
    'seresnext26_32x4d': {
        'fc_name': 'last_linear',
        'fc': nn.Linear(in_features=2048, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'seresnext26_32x4d', pretrained=True)
    },
    
    'efficientnet-b0': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=1280, out_features=2, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b0')
    },
    
    'efficientnet-b2': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=1408, out_features=4, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b2')
    },
    
    'efficientnet-b4': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=1792, out_features=2, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b4')
    },
    
    'efficientnet-b5': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=2048, out_features=2, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b5')
    },
    
    'efficientnet-b6': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=2304, out_features=4, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b6')
    },
    
    'efficientnet-b7': {
        'fc_name': '_fc',
        'fc': nn.Linear(in_features=2560, out_features=2, bias=True),
        'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b7')
    },
    
    'mixnet_xl': {
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1536, out_features=4, bias=True),
        'init_op': partial(timm.create_model, 'mixnet_xl', pretrained=True)
    },
    
    'mixnet_s': {
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1536, out_features=4, bias=True),
        'init_op':  partial(timm.create_model, 'mixnet_s', pretrained=True)
    }, 
    
    'mixnet_s_fromscratch': {
        'fc_name': 'classifier',
        'fc': nn.Linear(in_features=1536, out_features=4, bias=True),
        'init_op':  partial(timm.create_model, 'mixnet_s', pretrained=False)
    }, 
    
    'seresnet18': {
        'fc_name': 'last_linear',
        'fc': nn.Linear(in_features=512, out_features=4, bias=True),
        'init_op':  partial(timm.create_model, 'seresnet18', pretrained=True) #partial(seresnet18, pretrained=True, num_classes=4)
    }, 
}

def get_net(model_name):
    net = zoo_params[model_name]['init_op']()
    setattr(net, zoo_params[model_name]['fc_name'], zoo_params[model_name]['fc'])
    return net

def surgery_seresnet(net): 
    net.layer0.pool = nn.Identity()
    #net.prelayer = nn.Sequential(nn.Conv2d(3, 6, 3, stride=1, padding=3, bias=False),
    #                             nn.BatchNorm2d(6),
    #                             nn.ReLU(inplace=True),
    #                             nn.Conv2d(6, 12, 3, stride=1, padding=3, bias=False),
    #                             nn.BatchNorm2d(12),
    #                             nn.ReLU(inplace=True),
    #                             nn.Conv2d(12, 36, 3, stride=1, padding=3, bias=False),
    #                             nn.BatchNorm2d(36),
    #                             nn.ReLU(inplace=True),
    #                            )
    #
    #def new_forward_features(self, x):
    #    x = self.prelayer(x)
    #    x = self.layer0(x)
    #    x = self.layer1(x)
    #    x = self.layer2(x)
    #    x = self.layer3(x)
    #    x = self.layer4(x)
    #    return x
    #
    #net.forward_features = types.MethodType(new_forward_features, net)
    #net.layer0.conv1.weight = nn.Parameter(net.layer0.conv1.weight.repeat(1, 12, 1, 1))
    
    return net



def surgery_seresnext(net):
    net.drop_rate = 0.0
    net.layer0.pool = nn.Identity()

    #net.prelayer = nn.Sequential(nn.Conv2d(3, 6, 3, stride=1, padding=3, bias=False),
    #                             nn.BatchNorm2d(6),
    #                             nn.ReLU(inplace=True),
    #                             nn.Conv2d(6, 12, 3, stride=1, padding=3, bias=False),
    #                             nn.BatchNorm2d(12),
    #                             nn.ReLU(inplace=True),
    #                             nn.Conv2d(12, 36, 3, stride=1, padding=3, bias=False),
    #                             nn.BatchNorm2d(36),
    #                             nn.ReLU(inplace=True),
    #                            )
    #
    #def new_forward_features(self, x):
    #    x = self.prelayer(x)
    #    x = self.layer0(x)
    #    x = self.layer1(x)
    #    x = self.layer2(x)
    #    x = self.layer3(x)
    #    x = self.layer4(x)
    #    return x
    #
    #net.forward_features = types.MethodType(new_forward_features, net)
    #
    #net.layer0.conv1.weight = nn.Parameter(net.layer0.conv1.weight.repeat(1, 12, 1, 1))
    
    return net


def surgery_seresnet0(net): 
    
    net.prelayer = nn.Sequential(nn.Conv2d(3, 6, 3, stride=1, padding=3, bias=False),
                                 nn.BatchNorm2d(6),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(6, 12, 3, stride=1, padding=3, bias=False),
                                 nn.BatchNorm2d(12),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(12, 36, 3, stride=1, padding=3, bias=False),
                                 nn.BatchNorm2d(36),
                                 nn.ReLU(inplace=True),
                                )
    
    def new_forward_features(self, x):
        x = self.prelayer(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    net.forward_features = types.MethodType(new_forward_features, net)
    
    net.layer0.conv1.weight = nn.Parameter(net.layer0.conv1.weight.repeat(1, 12, 1, 1))
    
    return net


def surgery_b0midstem(net): 
    
    num_channels = net._conv_stem.out_channels # 48
    
    net._conv_stem.stride = (1,1)
    
    net._midstems = nn.ModuleList([timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels, noskip=True),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels, stride=2)])
    
    def new_extract_features(self, inputs):
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        
        for idx, block in enumerate(self._midstems):
            x = block(x)
    
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
    
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        return x
    
    net.extract_features = types.MethodType(new_extract_features, net)
        
    return net


def surgery_b0midstem1(net): 
    
    num_channels = net._conv_stem.out_channels
    
    net._conv_stem.stride = (1,1)
    
    net._midstems = nn.ModuleList([timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels, 
                                                                                    exp_ratio=1, se_ratio=0.5, noskip=True),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels, exp_ratio=1, se_ratio=0.5),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels, exp_ratio=1.5, se_ratio=0.25),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels, exp_ratio=2, se_ratio=0.25),
                    timm.models.efficientnet_blocks.InvertedResidual(in_chs=num_channels, out_chs=num_channels, stride=2, exp_ratio=2, se_ratio=0.25)])
    
    def new_extract_features(self, inputs):
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        
        for idx, block in enumerate(self._midstems):
            x = block(x)
    
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
    
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        return x
    
    net.extract_features = types.MethodType(new_extract_features, net)
        
    return net



def surgery_b0(net): 
    
    net.prelayer = nn.Sequential(nn.Conv2d(3, 6, 3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(6),
                                 nn.ReLU6(inplace=True),
                                 nn.Conv2d(6, 12, 3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(12),
                                 nn.ReLU6(inplace=True),
                                 nn.Conv2d(12, 36, 3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(36),
                                 nn.ReLU6(inplace=True))
    
    def new_extract_features(self, inputs):
        # Stem
        x = self.prelayer(inputs)
        x = self._swish(self._bn0(self._conv_stem(x)))
    
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
    
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        return x
    
    net.extract_features = types.MethodType(new_extract_features, net)
    
    net._conv_stem.weight = nn.Parameter(net._conv_stem.weight.repeat(1, 12, 1, 1))
    
    return net


def to_color(net):
    net.prelayer = nn.Conv2d(net.in_channels, 3, 1, stride=1, padding=1, bias=True)
    #net._conv_stem.stride = (1,1)
    def new_extract_features(self, inputs):
        # Stem
        x = self.prelayer(inputs)
        x = self._swish(self._bn0(self._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
    
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        return x
    
    net.extract_features = types.MethodType(new_extract_features, net)
    
    return net

def no_stride(net):
    net._conv_stem.stride = (1,1)
    
    return net
#
