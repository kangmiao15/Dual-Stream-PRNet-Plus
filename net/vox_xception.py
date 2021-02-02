import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def get_norm_module(num_channels):
    return nn.BatchNorm3d(num_channels)

class SeparableConv3d(nn.Module):

    def __init__(self, in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels,
                kernel_size, stride, padding, dilation, groups=int(in_channels/8), bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels,
                kernel_size=1, stride=1, padding=0, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class VoxBlock(nn.Module):

    def __init__(self, in_channels, out_channels, repeat, stride=1, dilation=1,
            start_with_relu=True, grow_first=True):
        super(VoxBlock, self).__init__()

        if out_channels != in_channels or stride != 1:
            self.skip = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                    get_norm_module(out_channels)
                    )
        else:
            self.skip = None

        if grow_first:
            channels = [out_channels] * repeat
        else:
            channels = [in_channels] * (repeat-1) + [out_channels]

        layers = []
        last_chns = in_channels
        for chns in channels:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv3d(last_chns, chns, 3, stride=1,
                padding=dilation, dilation=dilation, bias=False))
            layers.append(get_norm_module(chns))
            last_chns = chns
        
        if stride != 1:
            #layers.append(nn.MaxPool3d(3, stride, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv3d(last_chns, chns, 3, stride=2,
                padding=dilation, dilation=dilation, bias=False))
            layers.append(get_norm_module(chns))

        if not start_with_relu:
            layers = layers[1:]
        else:
            layers[0] = nn.ReLU(inplace=False)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        h = self.block(x)
        if self.skip is not None:
            x = self.skip(x)
        return h+x


class VoxXception(nn.Module):

    def __init__(self, in_channels, chns=[32, 64, 128, 256]):
        super(VoxXception, self).__init__()
        chns1, chns2, chns3, chns4 = chns

        # entry flow
        self.entry_conv = nn.Sequential(
                nn.Conv3d(in_channels, chns1, 3, stride=2, padding=1, bias=False),
                get_norm_module(chns1),
                nn.ReLU(inplace=True),
                nn.Conv3d(chns1, chns1, 3, stride=1, padding=1, bias=False),
                get_norm_module(chns1),
                nn.ReLU(inplace=True)
                )
        self.entry_block = nn.Sequential(
                VoxBlock(chns1, chns2, 3, 2, start_with_relu=False, grow_first=True),
                VoxBlock(chns2, chns2, 3, 1, start_with_relu=True, grow_first=True),
                )

        # middle flow
        self.middle_block = nn.Sequential(
                VoxBlock(chns2, chns3, 3, 2, start_with_relu=True, grow_first=True),
                VoxBlock(chns3, chns3, 3, 1, start_with_relu=True, grow_first=True),
                VoxBlock(chns3, chns3, 3, 1, start_with_relu=True, grow_first=True),
                VoxBlock(chns3, chns3, 3, 1, start_with_relu=True, grow_first=True),
                )

        # exit flow
        self.exit_block = nn.Sequential(
                VoxBlock(chns3, chns4, 3, 2, start_with_relu=True, grow_first=True),
                VoxBlock(chns4, chns4, 3, 1, start_with_relu=True, grow_first=True),
                )

    def forward(self, x):
        h1 = self.entry_conv(x)     # 1/2
        h2 = self.entry_block(h1)   # 1/4
        h3 = self.middle_block(h2)  # 1/8
        h4 = self.exit_block(h3)    # 1/16

        return h1, h2, h3, h4





