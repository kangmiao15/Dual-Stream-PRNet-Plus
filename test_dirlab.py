import os
import sys
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import sys
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import cv2
from skimage.morphology import closing, square, opening

from dataset.dirlab_dataset import DIRLabDataset
from dataset.pair_dataset import PairDataset
from op.warp_flow import add_offset, revert_offset
# from op.interpn_landmarks import warp_landmarks
from op.warp_flow import warp_landmarks
from utils.visualization import vis_flow, label2color
from evaluator import EvalTarRegErr
from preprocess import *
from net_factory import build_network3d as build_network

import time
import warnings

warnings.filterwarnings("ignore")

def dirlab_4dct_header():
    """
    size and voxel spacing of the images are available at https://www.dir-lab.com/ReferenceData.html
    """
    dirlab_info = dict()
    for cn in range(1, 11):
        dirlab_info['case' + str(cn)] = {}
    dirlab_info['case1']['Size'] = [256, 256, 94]
    dirlab_info['case2']['Size'] = [256, 256, 112]
    dirlab_info['case3']['Size'] = [256, 256, 104]
    dirlab_info['case4']['Size'] = [256, 256, 99]
    dirlab_info['case5']['Size'] = [256, 256, 106]
    dirlab_info['case6']['Size'] = [512, 512, 128]
    dirlab_info['case7']['Size'] = [512, 512, 136]
    dirlab_info['case8']['Size'] = [512, 512, 128]
    dirlab_info['case9']['Size'] = [512, 512, 128]
    dirlab_info['case10']['Size'] = [512, 512, 120]

    dirlab_info['case1']['Spacing'] = [0.97, 0.97, 2.5]
    dirlab_info['case2']['Spacing'] = [1.16, 1.16, 2.5]
    dirlab_info['case3']['Spacing'] = [1.15, 1.15, 2.5]
    dirlab_info['case4']['Spacing'] = [1.13, 1.13, 2.5]
    dirlab_info['case5']['Spacing'] = [1.10, 1.10, 2.5]
    dirlab_info['case6']['Spacing'] = [0.97, 0.97, 2.5]
    dirlab_info['case7']['Spacing'] = [0.97, 0.97, 2.5]
    dirlab_info['case8']['Spacing'] = [0.97, 0.97, 2.5]
    dirlab_info['case9']['Spacing'] = [0.97, 0.97, 2.5]
    dirlab_info['case10']['Spacing'] = [0.97, 0.97, 2.5]

    return dirlab_info

def compute_tre(mov_lmk, ref_lmk, spacing):
    #TRE, unit: mm
    diff = (ref_lmk - mov_lmk) * spacing
    diff = torch.Tensor(diff)
    tre = diff.pow(2).sum(1).sqrt()
    mean, std = tre.mean(), tre.std()
    return mean, std, diff

def net_forward(net, data_warp, data_fix, label_warp, gpu):
    if gpu:
        data_warp = data_warp.cuda()
        data_fix = data_fix.cuda()
    data = torch.stack([data_warp, data_fix])
    data = data.unsqueeze(1)

    with torch.no_grad():
        # forward
        predict, flow, delta_list = net(data)
        np.save('./dir_flow.npy', flow)
        # warp label
        # flow = add_offset(flow)
        # flow = revert_offset(flow)

        # flow = flow.squeeze(2)
        # flow = flow.cpu().numpy()
        # flow = np.stack([flow[:, 2, ...], flow[:, 1, ...], flow[:, 0, ...]], axis=1)
        # label_warp = label_warp.cpu().numpy()
        # label_warp = warp_landmarks(label_warp, flow)

        label_warp = warp_landmarks(label_warp, flow)

        # scale offset
        offset_list = []
        for lvl, delta in enumerate(delta_list):
            scale = 16/(2**lvl)
            # delta = scale*F.interpolate(delta, scale_factor=scale, mode='trilinear')
            delta = delta.squeeze()
            delta = delta.cpu().numpy()
            offset_list.append(delta)
        
        # revert flow

    predict = predict.squeeze()
    predict = predict.cpu().numpy()
    flow = flow.squeeze()

    return predict, flow, offset_list, label_warp

def draw_label(data, label):
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8
    data = np.stack((data, data, data), axis=-1)
    # 要画的点的坐标
    for point in label:
        x = round(point[0])
        y = round(point[1])
        z = round(point[2])
        cv2.circle(data[x, :, :, :], (y,z), point_size, point_color, thickness)
    return data

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train bone segmentation model')
    parser.add_argument("--net", help="netname for network factory", type=str)
    parser.add_argument("--model_file", help="saved model file", type=str)
    parser.add_argument("--data_root", help="validation data folder", type=str)
    parser.add_argument("--gpu", help="use GPU", default=False, type=bool)
    parser.add_argument("--fold", help="fold for cross validation", default=1, type=int)
    parser.add_argument("--vis", help="visualization", default=False, type=bool)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('gpu', args.gpu)
    
    # build dataset
    # trans_data = [ CenterCroplandmarks(dst_size=(96,256,256)) ]
    trans_data = [ Padlandmarks(dst_size=(96,256,256)) ]
    trans_pair = []
    data_list = [ ("case%d" % (args.fold) , "T%02d" % j) for j in [0, 50]]
    print(data_list)
    dataset = DIRLabDataset(args.data_root, data_list, trans_data, with_label=True)
    dataset = PairDataset(dataset, None, trans_pair, with_label=True, data_name='dirlab')

    num_pair = len(dataset)
    print("number of pair: %d" % num_pair)

    # build network
    net, _ = build_network(args.net)
    print('# net parameters:', sum(param.numel() for param in net.parameters()))
    net.load_state_dict(torch.load(args.model_file, map_location="cpu"))
    if args.gpu:
        print('GPU')
        net = net.cuda()
    # testing 
    random_idx = list(range(num_pair))
    dir_info = dirlab_4dct_header()
    for i in random_idx[0:1]:
        print("processing %d" % i)
        # fetch data
        data_left, data_right = dataset[i]
        data_left, label_left = data_left
        data_right, label_right = data_right
        #label_right to get the warped position 
        pred_left, flow, offset_list, warp_label_left = net_forward(net, data_left, data_right, label_right, args.gpu)
        # compute TRE
        case_num = 'case' + str(args.fold)
        img_spacing = [2.5, 1.0, 1.0]
        raw_shape = np.flip(dir_info[case_num]['Size'])
        resize_factor = data_left.shape / raw_shape #new_spacing
        # resize the landmark
        label_left = label_left * resize_factor
        label_right = label_right * resize_factor

        #flow sampling
        ref_lmk_index = np.round(label_right).astype('int32')

        label_warp = label_right.copy()
        for i in range(300):
            wi, hi, di = ref_lmk_index[i]
            w0, h0, d0 = flow[0, :, di, hi, wi]
            label_warp[i] += [w0, h0, d0]
        #    break
            
        # compute TRE
        #no reg
        tre_mean_bf, tre_std_bf, diff_br = compute_tre(label_left, label_right, img_spacing)
        print('TRE-before reg, mean: {:.2f},std: {:.2f}'.format(
                tre_mean_bf, tre_std_bf))
        #with reg
        tre_mean_af, tre_std_af, diff_ar = compute_tre(label_left, label_warp, img_spacing)
        print('TRE-after reg, mean: {:.2f},std: {:.2f}'.format(
                tre_mean_af, tre_std_af))

        if args.vis is False:
            continue
        # to numpy
        data_left = data_left.numpy()
        data_left = (data_left*255).astype(np.uint8)
        data_right = data_right.numpy()
        data_right = (data_right*255).astype(np.uint8)
        label_right = label_right.numpy().astype(np.uint8)
        label_left = label_left.numpy().astype(np.uint8)
        pred_left = (pred_left*255).astype(np.uint8)

        # offset_list = [ vis_flow(offset) for offset in offset_list ]
        # flow = vis_flow(flow)
        # data_left_org = draw_label(data_left, label_left)
        # data_right = draw_label(data_right, label_right)
        # data_left_antwarp = draw_label(pred_left, warp_label_left)
        # pred_closing = draw_label(pred_left, warp_label_left_closing)
        for j in range(0, data_left.shape[0]):
            result = np.concatenate([data_left[j, ...], data_right[j, ...], pred_left[j, ...]], axis=1)
            # result = np.concatenate([data_left[..., j, :], data_right[..., j, :]], axis=1)
            print(data_left.shape)
            result = cv2.resize(result, None, fx=2, fy=2)
            # delta = np.concatenate([o[j, ...] for o in offset_list], axis=1)
            # delta = cv2.resize(delta, None, fx=2, fy=2)
            #cv2.imshow("delta", delta)
            cv2.imwrite('./temp/dir_result_%s.jpg' % j, result)
            # cv2.imshow("result", result)
            #cv2.imshow("flow", cv2.resize(flow[j, ...], None, fx=2, fy=2))
            # cv2.waitKey()
    
