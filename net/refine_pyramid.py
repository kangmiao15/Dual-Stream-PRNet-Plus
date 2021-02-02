import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class _RefinePyramid(nn.Module):

    def __init__(self, backend, conv_size, fpn_dim):

        super(_RefinePyramid, self).__init__()
        self.conv_size = conv_size
        self.backend = backend

        # adaptive 
        self.adaptive = []
        for chns_in in list(reversed(conv_size)):
            adaptive_layer = self.build_adaptive(chns_in, fpn_dim)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)

        # output conv
        self.smooth = []
        for i in range(len(conv_size)):
            smooth_layer = self.build_smooth(fpn_dim)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)
    

    def build_adaptive(self, chns_in, fpn_dim):
        raise NotImplementedError
    

    def build_smooth(self, fpn_dim):
        raise NotImplementedError
        

    def forward(self, x):
        if self.backend is None:
            conv_ftr_list = x
        else:
            conv_ftr_list = self.backend(x)
        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive[i](conv_ftr)

            # fuse
            if last_feature is not None:
                feature = feature + F.interpolate(last_feature, scale_factor=2, mode=self.interp_mode)
            
            # smooth
            feature = self.smooth[i](feature)

            last_feature = feature
            feature_list.append(feature)

        return tuple(reversed(feature_list))


class RefinePyramid2d(_RefinePyramid):

    def __init__(self, backend, conv_size, fpn_dim):
        super(RefinePyramid2d, self).__init__(backend, conv_size, fpn_dim)
        self.interp_mode = "bilinear"


    def build_adaptive(self, chns_in, fpn_dim):
        adaptive_layer = nn.Conv2d(chns_in, fpn_dim, kernel_size=1)
        return adaptive_layer
    

    def build_smooth(self, fpn_dim):
        smooth_layer = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
        return smooth_layer


class RefinePyramid3d(_RefinePyramid):

    def __init__(self, backend, conv_size, fpn_dim):
        super(RefinePyramid3d, self).__init__(backend, conv_size, fpn_dim)
        self.interp_mode = "trilinear"


    def build_adaptive(self, chns_in, fpn_dim):
        adaptive_layer = nn.Conv3d(chns_in, fpn_dim, kernel_size=1)
        return adaptive_layer
    

    def build_smooth(self, fpn_dim):
        smooth_layer = nn.Conv3d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
        return smooth_layer