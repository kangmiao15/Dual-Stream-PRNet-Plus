import os
import sys
import torch
import numpy as np
sys.path.append('../')
# sys.path.append('/opt/FTE/users/miakang/research/data_augumentation/correlation_package_cu9')
from correlation import Correlation
from correlation_package_cu9.correlation import Correlation as Correlation_old
import pdb

if __name__ == "__main__":
    input1 = torch.ones([1, 1, 160, 160, 128])*4
    input2 = torch.ones([1, 1, 160, 160, 128])*5
    input1 = input1.cuda()
    input1 = input2.cuda()
    # net = Correlation(pad_size=3, kernel_size=3, max_displacement=3, stride1=1, stride2=2, corr_multiply=1)
    net = Correlation(pad_size=3, kernel_size=1, max_displacement=3, stride1=1, stride2=2, corr_multiply=1)

    net = net.cuda()
    net = torch.nn.DataParallel(net)
    output = net(input1, input2)

    # net_old = Correlation_old(pad_size=3, kernel_size=1, max_displacement=3, stride1=1, stride2=2, corr_multiply=1)
    # net_old = net_old.cuda()
    # net_old = torch.nn.DataParallel(net_old)
    # output_old = net_old(input1, input2)
    # output_old = output_old.cpu().numpy()
    # output = output.cpu().numpy()
    # pdb.set_trace()
    # diff = output-output_old
    print(output.shape)
    print(output.max())
    output = output.cpu().numpy()
    # np.save('./co1.npy', output)
    # print('diff', (output-output_old).max())
