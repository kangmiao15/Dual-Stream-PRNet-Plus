import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VoxRex(nn.Module):
   
    def __init__(self, in_channels):
        
        super(VoxRex, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)+x


class VoxNet(nn.Module):

    ''' base backend '''
    def __init__(self, in_channels, chns=[32,64,64,64], num_output=-1):

        super(VoxNet, self).__init__()
        ftr1,ftr2,ftr3,ftr4,ftr5 = chns
        # stage 1
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(in_channels, ftr1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            )
        # stage 2
        self.conv1_2 = nn.Sequential(
            nn.Conv3d(ftr1, ftr2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
            )
        # self.voxres2 = VoxRex(ftr2)
        # self.voxres3 = VoxRex(ftr2)
        # stage 3
        self.conv4 = nn.Sequential(
            nn.Conv3d(ftr2, ftr3, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
            )
        # self.voxres5 = VoxRex(ftr3)
        # self.voxres6 = VoxRex(ftr3)
        # stage 4
        self.conv7 = nn.Sequential(
            nn.Conv3d(ftr3, ftr4, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
            )

        # self.voxres8 = VoxRex(ftr4)
        # self.voxres9 = VoxRex(ftr4)
        # stage 5
        self.conv8 = nn.Sequential(
            nn.Conv3d(ftr4, ftr5, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
            )

        # self.voxres10 = VoxRex(ftr5)
        # self.voxres11 = VoxRex(ftr5)
        
        self.decv1 = nn.Sequential(
                nn.Conv3d(ftr5, ftr5, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.decv2 = nn.Sequential(
                nn.Conv3d(ftr5+ftr4, ftr4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )    
        self.decv3 = nn.Sequential(
                nn.Conv3d(ftr4+ftr3, ftr3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.decv4 = nn.Sequential(
                nn.Conv3d(ftr3+ftr2, ftr2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.decv5 = nn.Sequential(
                nn.Conv3d(ftr2+ftr1, ftr1, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
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
        # x = self.voxres2(x)
        # x = self.voxres3(x)
        h2 = x
        x = self.conv4(x)
        # x = self.voxres5(x)
        # x = self.voxres6(x)
        h3 = x
        x = self.conv7(x)
        # x = self.voxres8(x)
        # x = self.voxres9(x)
        h4 = x
        x = self.conv8(x)
        # x = self.voxres10(x)
        # x = self.voxres11(x)
        
        x = self.decv1(F.interpolate(x, scale_factor=2, mode='trilinear'))
        d5 = x
        x = torch.cat([x, h4], dim=1)
        del h4
        x = self.decv2(F.interpolate(x, scale_factor=2, mode='trilinear'))
        d4 = x
       
        x = torch.cat([x, h3], dim=1)
        del h3
        x = self.decv3(F.interpolate(x, scale_factor=2, mode='trilinear'))
        d3 = x

        x = torch.cat([x, h2], dim=1)
        del h2
        x = self.decv4(F.interpolate(x, scale_factor=2, mode='trilinear'))
        d2 = x

        x = torch.cat([x, h1], dim=1)
        del h1
        x = self.decv5(x)
        d1 = x

        if self.num_output > 0:
            predict = self.predict(h4)
            return F.adaptive_avg_pool3d(predict, 1).view(predict.size(0), -1)
        else:
            return d1, d2, d3, d4, d5