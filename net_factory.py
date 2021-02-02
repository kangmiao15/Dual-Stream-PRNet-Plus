import torch
import torch.nn as nn
import torch.nn.functional as F

from net.vox_resnet import VoxResNet
from net.vox_xception import VoxXception
from net.voxnet import VoxNet
from net.resnet import resnet50
from net.cascade_morph_net import CascadeMorphNet3d, CascadeMorphNet2d
from net.cascade_fusion_net import CascadeFusionNet3d
from net.baseline import Baseline3d, Baseline2d


def build_backend(backend, in_channels, conv_size):
    if backend == 'resnet':
        backend = VoxResNet(in_channels=in_channels, chns=conv_size)
    elif backend == 'xception':
        backend = VoxXception(in_channels=in_channels, chns=conv_size)
    elif backend == 'voxnet':
        backend = VoxNet(in_channels=in_channels, chns=conv_size)
    else:
        print("unknow backend: %s" % backend)
        raise NotImplementedError
    return backend
    

def build_network3d(netname, symmetry=True):
    print("symmetry", symmetry)
    backend, header = netname.split('_')

    # config 
    in_channels = 1
    # miccai
    # conv_size = [8, 16, 32, 32]
    # fpn_dim = 32
    # MedIA
    conv_size = [8, 16, 16, 32, 32]
    fpn_dim = [32, 32, 16, 16, 8]

    # build backend
    if symmetry:
        backend = build_backend(backend, in_channels, conv_size)
    else:
        backend = [ 
            build_backend(backend, in_channels, conv_size),
            build_backend(backend, in_channels, conv_size)
        ]
    print('# backend parameters:', sum(param.numel() for param in backend.parameters()))

    # build structure
    if header == "cascade":
        net = CascadeMorphNet3d(backend, conv_size, fpn_dim=fpn_dim, symmetry=symmetry)
    elif header == "fusion":
        net = CascadeFusionNet3d(backend, conv_size, fpn_dim=fpn_dim)
    elif header == "baseline":
        net = Baseline3d(backend, conv_size, fpn_dim=fpn_dim, symmetry=symmetry)
    else:
        raise NotImplementedError

    output_scale = 0.5

    return net, output_scale


def build_network2d(netname, symmetry=True):
    print("symmetry", symmetry)
    # config 
    in_channels = 1
    conv_size = [64, 256, 512, 1024]
    fpn_dim = 256

    if netname != "resnet":
        raise NotImplementedError

    # build backend
    if symmetry:
        backend = resnet50(pretrained=True)
    else:
        backend = [ 
            resnet50(pretrained=True),
            resnet50(pretrained=True)
        ]
    
    # build structure
    net = CascadeMorphNet2d(backend, conv_size, fpn_dim=fpn_dim, symmetry=symmetry)
    output_scale = 0.5

    return net, output_scale


def build_discriminator():
    # config 
    in_channels = 1
    conv_size = [8, 16, 32, 64]

    # build backend
    backend = VoxResNet(in_channels=in_channels, chns=conv_size, num_output=1)

    return backend



if __name__ == '__main__':

    net, _ = build_network3d("resnet")
    x = torch.rand(2, 1, 16, 16, 16)
    net(x)

    net, _ = build_network2d("resnet")
    x = torch.rand(2, 3, 64, 64)
    net(x)
