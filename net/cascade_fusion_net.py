import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .refine_pyramid import RefinePyramid3d
from .voxnet import VoxNet
from .voxel_morph import VoxelMorphCVPR2018
from op.warp_flow import apply_offset
from correlation_package.correlation import Correlation

from utils.visualization import vis_flow, label2color

def permute_channel_last(x):
    dims = [0] + list(range(2, x.dim())) + [1]
    return x.permute(*dims)

def permute_channel_first(x):
    dims = [0, x.dim()-1] + list(range(1, x.dim()-1))
    return x.permute(*dims)

def draw_label(data, label):
    color_map = label2color(label)
    mask = label > 0
    mask = np.stack([mask, mask, mask], axis=-1)
    data = np.stack([data, data, data], axis=-1)
    data[mask] = color_map[mask]
    return data

class DownSample(nn.Module):

    def __init__(self, ftr1, ftr2):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ftr1, ftr1, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm3d(ftr1),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool3d(2),
            nn.Conv3d(ftr1, ftr1, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(ftr1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(ftr1, ftr2, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(ftr2),
            nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)

class Bottle(nn.Module):

    def __init__(self, ftr1, ftr2):
        super(Bottle, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ftr1, ftr2, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(ftr2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(ftr2, ftr2, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(ftr2),
            nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class _CascadeFusionNet(nn.Module):
    
    def __init__(self, backend, conv_size, fpn_dim, keep_delta=True):
        super(_CascadeFusionNet, self).__init__()
        self.num_pyramid = len(conv_size)
        self.keep_delta = keep_delta

    def forward(self, x):
        '''
        # nodual cascade 
        pyramid = self.pyramid(x)
        len_f = len(pyramid)
        # coarse to fine
        pyramid = reversed(pyramid)
        last_flow = None
        delta_list = []
        for lvl, feature in enumerate(pyramid):
            # apply flow
            # N, 3, D, H, W
            flow = self.offset[lvl](feature)
            if self.keep_delta:
                delta_list.append(flow)
            flow = apply_offset(flow)
            if last_flow is not None:
                flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode="border")
            else:
                flow = permute_channel_first(flow)
            if lvl < len_f-2:
                last_flow = F.interpolate(flow, scale_factor=2, mode=self.interp_mode)
            else:
                last_flow = flow
        x_warp = x[:, 0, ...]
        x_warp = x_warp.unsqueeze(1)
        x_warp = F.grid_sample(x_warp, permute_channel_last(last_flow),
                     mode='bilinear', padding_mode="border")
        return x_warp, last_flow, delta_list              
        '''
        # dual cascade 
        batch_size = int(x.size(0)/2)
        pyramid = self.pyramid(x)
        feature_warp = [ feature[:batch_size, ...] for feature in pyramid ]
        feature_fix = [ feature[batch_size:, ...] for feature in pyramid ]
        # coarse to fine
        feature_warp = reversed(feature_warp)
        feature_fix = reversed(feature_fix)
        last_flow = None
        delta_list = []
        # delta_last_list = []
        for lvl, (x_warp, x_fix) in enumerate(zip(feature_warp, feature_fix)):
            # apply flow
            if last_flow is not None:
                x_warp = F.grid_sample(x_warp, permute_channel_last(last_flow.detach()),
                     mode='bilinear', padding_mode="border")
            # fusion
            flow = self.offset[lvl](x_warp, x_fix)
            # cascade
            # flow = self.offset[lvl](torch.cat([x_warp, x_fix], dim=1))
            if self.keep_delta:
                delta_list.append(flow)
            flow = apply_offset(flow)
            if last_flow is not None:
                flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode="border")
            else:
                flow = permute_channel_first(flow)
            # last_flow = F.interpolate(flow, scale_factor=2, mode=self.interp_mode)
            if lvl < len(pyramid)-2:
                last_flow = F.interpolate(flow, scale_factor=2, mode=self.interp_mode)
            else:
                last_flow = flow
            # if self.keep_delta:
            #     delta_last_list.append(last_flow)
        x_warp = x[:batch_size, ...]
        x_warp = F.grid_sample(x_warp, permute_channel_last(last_flow),
                     mode='bilinear', padding_mode="border")
        return x_warp, last_flow, delta_list


class CascadeFusionNet3d(_CascadeFusionNet):
    
    def __init__(self, backend, conv_size, fpn_dim, keep_delta=True):
        super(CascadeFusionNet3d, self).__init__(backend, conv_size, fpn_dim, keep_delta)
        self.pyramid = backend
        # self.pyramid = RefinePyramid3d(backend, conv_size, fpn_dim)
        self.interp_mode = "trilinear"

        # adaptive 
        self.fuse = []
        self.resblock = []
        self.offset = []
        for i in range(len(conv_size)):
            # fusion
            offset_layer = FlowNet(conv_size[-i-1]*2)
            # cascade
            # offset_layer = nn.Conv3d(conv_size[-i-1]*2, 3, kernel_size=3, padding=1)
            self.offset.append(offset_layer)
        self.offset = nn.ModuleList(self.offset)
        


class FlowNet_bk(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(FlowNet, self).__init__()
        med_channels = int(in_channels/2)
        self.fuse_layer = nn.Sequential(
                nn.Conv3d(in_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.resblock_layer_1 = nn.Sequential(
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        # self.resblock_layer_2 = nn.Sequential(
        #         nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
        #     )
        # self.corr_layer = Correlation(pad_size=3, kernel_size=1, max_displacement=3, stride1=1, stride2=2, corr_multiply=1)
        # self.conv_layer = nn.Sequential(
        #         nn.Conv3d(med_channels+27, med_channels, kernel_size=3, padding=1),
        #         nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
                # nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
            #     nn.ReLU(inplace=True),
            # )
        self.offset_layer = nn.Conv3d(med_channels, out_channels, kernel_size=3, padding=1)
        # self.offset_layer = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        # import pdb; pdb.set_trace()
        # corr = self.corr_layer(x1, x2)
        # corr = F.relu(corr)
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse_layer(x)
        x = x + self.resblock_layer_1(x)
        # x = x + self.resblock_layer_2(x)
        # x = self.conv_layer(torch.cat([x, corr], dim=1))
        # x = self.conv_layer(x)
        x = self.offset_layer(x)
        return x
    
class FlowNet(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(FlowNet, self).__init__()
        med_channels = int(in_channels/2)
        # med_channels = 27
        self.corr_layer = Correlation(pad_size=3, kernel_size=1, max_displacement=3, stride1=1, stride2=2, corr_multiply=1)
        # self.conv = nn.Sequential(
        #         nn.Conv3d(27, 8, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
        #     )
        self.conv_layer = nn.Sequential(
                nn.Conv3d(27+in_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.resblock_layer_1 = nn.Sequential(
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        # self.resblock_layer_2 = nn.Sequential(
        #         nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
        #     )
        self.offset_layer = nn.Conv3d(med_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x3 = self.corr_layer(x1, x2)
        # x3 = self.conv(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv_layer(x)
        x = x + self.resblock_layer_1(x)
        # x = x + self.resblock_layer_2(x)
        x = self.offset_layer(x)
        return x

class SegNet(nn.Module):
    def __init__(self, in_channels, num_classes, chns=[8,16,32,64]):
        super(SegNet, self).__init__()

        ftr1, ftr2, ftr3, ftr4 = chns
        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels, ftr1, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(ftr1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(ftr1, ftr2, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(ftr2),
            nn.LeakyReLU(inplace=True)
            )
        
        self.downsample_1 = DownSample(ftr2, ftr3)

        self.downsample_2 = DownSample(ftr3, ftr4)

        self.downsample_3 = DownSample(ftr4, ftr4)

        self.bottle_1 = Bottle(ftr4+ftr4, ftr4)

        self.bottle_2 = Bottle(ftr3+ftr4, ftr3)

        self.bottle_3 = Bottle(ftr2+ftr3, ftr2)

        self.predict = nn.Conv3d(ftr2, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv_1(x)
        h1 = x
        x = self.downsample_1(x)
        h2 = x
        x = self.downsample_2(x)
        h3 = x
        x = self.downsample_3(x)

        x = F.interpolate(x, scale_factor=2, mode='trilinear')
        x = torch.cat([x, h3], dim=1)
        del h3
        x = self.bottle_1(x)

        x = F.interpolate(x, scale_factor=2, mode='trilinear')
        x = torch.cat([x, h2], dim=1)
        del h2
        x = self.bottle_2(x)

        x = F.interpolate(x, scale_factor=2, mode='trilinear')
        x = torch.cat([x, h1], dim=1)
        del h1
        x = self.bottle_3(x)

        x = self.predict(x)

        return x


class CascadeSeg(nn.Module):
    def __init__(self, backend, conv_size, fpn_dim, num_classes=6):
        super(CascadeSeg, self).__init__()
        self.net_reg = CascadeMorphNet3d(backend, conv_size, fpn_dim, keep_delta=False)
        self.net_seg = SegNet(in_channels=1, num_classes=num_classes)
    
    def forward(self, x):
        # train seg only
        with torch.no_grad():
            moved, last_flow, _ = self.net_reg(x)
        seg = self.net_seg(x)

        # single_stream
        # with torch.no_grad():
        #     moved, last_flow, _ = self.net_reg(x)
        # batch_size = int(x.size(1)/2)
        # x = torch.cat((x[:batch_size, ...], x[batch_size:, ...]), dim=0)
        # seg = self.net_seg(x)
        return moved, last_flow, seg
    

class VoxmorphSeg(nn.Module):
    def __init__(self, conv_size, fpn_dim, num_classes=6):
        super(VoxmorphSeg, self).__init__()
        self.net_reg = VoxelMorphCVPR2018()
        self.net_seg = SegNet(in_channels=1, num_classes=num_classes)
        # self.net_seg = UNet(in_channels=1, num_classes=num_classes)

    def forward(self, x):
        # train seg only
        with torch.no_grad():
            moved, last_flow, dealt_list = self.net_reg(x)
            del dealt_list
        batch_size = int(x.size(1)/2)
        x = torch.cat((x[:, :batch_size, ...], x[:, batch_size:, ...]), dim=0)
        seg = self.net_seg(x)
        return moved, last_flow, seg



        

