import torch
import torch.nn as nn
import torch.nn.functional as F

from .refine_pyramid import RefinePyramid2d, RefinePyramid3d

from op.warp_flow import apply_offset


def permute_channel_last(x):
    dims = [0] + list(range(2, x.dim())) + [1]
    return x.permute(*dims)


def permute_channel_first(x):
    dims = [0, x.dim()-1] + list(range(1, x.dim()-1))
    return x.permute(*dims)


class _Baseline(nn.Module):
    
    def __init__(self, backend, conv_size, fpn_dim, symmetry):
        super(_Baseline, self).__init__()
        self.num_pyramid = len(conv_size)
        self.symmetry = symmetry


    def forward(self, x):
        batch_size = int(x.size(0)/2)
        if self.symmetry:
            pyramid = self.pyramid(x)
            feature_warp = [ feature[:batch_size, ...] for feature in pyramid ]
            feature_fix = [ feature[batch_size:, ...] for feature in pyramid ]
        else:
            feature_warp = self.pyramid_warp(x[:batch_size, ...])
            feature_fix = self.pyramid_fix(x[batch_size:, ...])

        flow = self.offset(torch.cat([feature_warp[0], feature_fix[0]], dim=1))
        delta_list = [ flow ]
        flow = apply_offset(flow)
        flow = permute_channel_first(flow)
        last_flow = F.interpolate(flow, scale_factor=2, mode=self.interp_mode)

        batch_size = int(x.size(0)/2)
        x_warp = x[:batch_size, ...]
        x_warp = F.grid_sample(x_warp, permute_channel_last(last_flow),
                     mode='bilinear', padding_mode="border")

        return x_warp, last_flow, delta_list
    
        
class Baseline3d(_Baseline):
    
    def __init__(self, backend, conv_size, fpn_dim, symmetry=True):
        super(Baseline3d, self).__init__(backend, conv_size, fpn_dim, symmetry)
        if symmetry:
            self.pyramid = RefinePyramid3d(backend, conv_size, fpn_dim)
        else:
            self.pyramid_warp = RefinePyramid3d(backend[0], conv_size, fpn_dim)
            self.pyramid_fix = RefinePyramid3d(backend[1], conv_size, fpn_dim)
        self.interp_mode = "trilinear"
        self.offset = nn.Conv3d(fpn_dim*2, 3, kernel_size=3, padding=1)


class Baseline2d(_Baseline):
    
    def __init__(self, backend, conv_size, fpn_dim, symmetry=True):
        super(Baseline2d, self).__init__(backend, conv_size, fpn_dim, symmetry)
        if symmetry:
            self.pyramid = RefinePyramid2d(backend, conv_size, fpn_dim)
        else:
            self.pyramid_warp = RefinePyramid2d(backend[0], conv_size, fpn_dim)
            self.pyramid_fix = RefinePyramid2d(backend[1], conv_size, fpn_dim)
        self.interp_mode = "bilinear"
        self.offset = nn.Conv2d(fpn_dim*2, 3, kernel_size=3, padding=1)
