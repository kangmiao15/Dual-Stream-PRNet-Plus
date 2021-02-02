import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VoxRex(nn.Module):
    
    def __init__(self, in_channels):
        super(VoxRex, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            # nn.GroupNorm(8, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(8, in_channels),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )


    def forward(self, x):
        return self.block(x)+x


class VoxResNet(nn.Module):

    ''' base backend '''
    def __init__(self, in_channels, chns=[32,64,64,64], num_output=-1):

        super(VoxResNet, self).__init__()
        ftr1,ftr2,ftr3,ftr4 = chns
        # stage 1
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(in_channels, ftr1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ftr1),
            # nn.GroupNorm(8, ftr1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ftr1, ftr1, kernel_size=3, padding=1, bias=False)
            )
        # stage 2
        self.conv1_2 = nn.Sequential(
            nn.BatchNorm3d(ftr1),
            # nn.GroupNorm(8, ftr1),
            nn.ReLU(inplace=True),
            nn.Conv3d(ftr1, ftr2, kernel_size=3, stride=2, padding=1, bias=True)
            )
        self.voxres2 = VoxRex(ftr2)
        self.voxres3 = VoxRex(ftr2)
        # stage 3
        self.conv4 = nn.Sequential(
            nn.BatchNorm3d(ftr2),
            # nn.GroupNorm(8, ftr2),
            nn.ReLU(inplace=True),
            nn.Conv3d(ftr2, ftr3, kernel_size=3, stride=2, padding=1, bias=True)
            )
        self.voxres5 = VoxRex(ftr3)
        self.voxres6 = VoxRex(ftr3)
        # stage 4
        self.conv7 = nn.Sequential(
            nn.BatchNorm3d(ftr3),
            # nn.GroupNorm(8, ftr3),
            nn.ReLU(inplace=True),
            nn.Conv3d(ftr3, ftr4, kernel_size=3, stride=2, padding=1, bias=True)
            )

        self.voxres8 = VoxRex(ftr4)
        self.voxres9 = VoxRex(ftr4)

        self.num_output = num_output
        if self.num_output > 0:
            self.predict = nn.Sequential( # nn.Dropout3d(inplace=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(ftr4, num_output, 1)
            )


    def forward(self, x):
        x = self.conv1_1(x)
        h1 = x
        x = self.conv1_2(x)
        x = self.voxres2(x)
        x = self.voxres3(x)
        h2 = x
        x = self.conv4(x)
        x = self.voxres5(x)
        x = self.voxres6(x)
        h3 = x
        x = self.conv7(x)
        x = self.voxres8(x)
        x = self.voxres9(x)
        h4 = x

        if self.num_output > 0:
            predict = self.predict(h4)
            return F.adaptive_avg_pool3d(predict, 1).view(predict.size(0), -1)
        else:
            return h1, h2, h3, h4



