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
import xlwt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import cv2
from skimage.morphology import closing, square, opening

from dataset.mind101_dataset import Mind101Dataset
from dataset.pair_dataset import PairDataset
from op.warp_flow import revert_offset
from utils.visualization import vis_flow, label2color
from evaluator import EvalDiceScore
from preprocess import *
from net_factory import build_network3d as build_network

import time
import warnings

warnings.filterwarnings("ignore")
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train bone segmentation model')
    parser.add_argument("--net", help="netname for network factory", type=str)
    parser.add_argument("--model_file", help="saved model file", type=str)
    parser.add_argument("--data_root", help="validation data folder", type=str)
    parser.add_argument("--is_cuda", help="use GPU", default=False, type=bool)
    parser.add_argument("--vis", help="visualization", default=False, type=bool)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def net_forward(net, data_warp, data_fix, label_warp, is_cuda):
    if is_cuda:
        data_warp = data_warp.cuda()
        data_fix = data_fix.cuda()
        label_warp = label_warp.cuda()
    data = torch.stack([data_warp, data_fix])
    data = data.unsqueeze(1)
    label_warp = label_warp[None, None, ...]

    with torch.no_grad():
        # forward
        predict, flow, delta_list = net(data)

        # warp label
        label_warp = F.grid_sample(label_warp, flow.permute(0, 2, 3, 4, 1),
            mode='nearest', padding_mode="border")

        # scale offset
        offset_list = []
        for lvl, delta in enumerate(delta_list):
            scale = 16/(2**lvl)
            # delta = scale*F.interpolate(delta, scale_factor=scale, mode='trilinear')
            delta = delta.squeeze()
            delta = delta.cpu().numpy()
            offset_list.append(delta)
        
        # revert flow
        flow = revert_offset(flow)

    predict = predict.squeeze()
    predict = predict.cpu().numpy()
    flow = flow.squeeze()
    flow = flow.cpu().numpy()
    label_warp = label_warp.squeeze()
    label_warp = label_warp.cpu().numpy().astype(np.uint8)

    return predict, flow, offset_list, label_warp

def draw_label(data, label):
    color_map = label2color(label)
    mask = label > 0
    mask = np.stack([mask, mask, mask], axis=-1)
    data = np.stack([data, data, data], axis=-1)
    data[mask] = color_map[mask]
    return data


if __name__ == '__main__':
    args = parse_args()
    # build dataset
    trans_data = [ CenterCrop(dst_size=(160,192,160)) ]
    # trans_data = [ CenterCrop(dst_size=(160,176,128)) ]
    trans_pair = []
    data_list = os.listdir(args.data_root)
    dataset = Mind101Dataset(args.data_root, data_list, trans=trans_data, with_label=True)
    dataset = PairDataset(dataset, None, trans_pair, with_label=True)
    num_pair = len(dataset)
    print("number of pair: %d" % num_pair)

    # build network
    net, _ = build_network(args.net)
    print('# net parameters:', sum(param.numel() for param in net.parameters()))
    net.load_state_dict(torch.load(args.model_file, map_location="cpu"))
    if args.is_cuda:
        print('GPU')
        net = net.cuda()
    
    #
    regions = [ "frontal_lobe", "parietal_lobe", "occipital_lobe", "temporal_lobe", "cingulate_lobe" ]
    evaluator_list = []
    for i in range(len(regions)):
        evaluator_list.append( EvalDiceScore() )

    # testing 
    random_idx = list(range(num_pair))
    # random.shuffle(random_idx)
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('sheet1')
    # for i in random_idx:
    for i in [68,69,70]:
        print("processing %d" % i)

        # fetch data
        data_left, data_right = dataset[i]
        data_left, label_left = data_left
        data_right, label_right = data_right
        '''
        # for cross validate
        data_left = data_left.permute(2, 1, 0)
        label_left = label_left.permute(2, 1, 0)
        data_right = data_right.permute(2, 1, 0)
        label_right = label_right.permute(2, 1, 0)
        '''
        pred_left, flow, offset_list, warp_label_left = net_forward(net, data_left, data_right, label_left, args.is_cuda)

        # to numpy
        data_left = data_left.numpy()
        data_left = (data_left*255).astype(np.uint8)
        data_right = data_right.numpy()
        data_right = (data_right*255).astype(np.uint8)
        label_right = label_right.numpy().astype(np.uint8)
        label_left = label_left.numpy().astype(np.uint8)
        pred_left = (pred_left*255).astype(np.uint8)
        # kernel = square(2)
        kernel = square(2)
        warp_label_left_closing = np.zeros(warp_label_left.shape, dtype=warp_label_left.dtype)
        for j in range(warp_label_left.shape[0]):
            data = warp_label_left[j, ...]
            data = closing(data, kernel)
            # data = opening(data, kernel)            
            warp_label_left_closing[j, ...] = data
        # warp_label_left = warp_label_left_closing

        # evaluate
        sum = 0
        for num, evaluator in enumerate(evaluator_list):
            region_left = warp_label_left_closing == num+1
            # region_left = warp_label_left == num+1
            region_right = label_right == num+1
            evaluator.AddResult(region_left, region_right)
            sum += evaluator.Eval()
        avg = sum/(len(evaluator_list)-2)
        # worksheet.write(i, 2, avg)
        # workbook.save('./cascade_result.xls')  
        if args.vis is False:
            continue
        offset_list = [ vis_flow(offset) for offset in offset_list ]
        flow = vis_flow(flow)
        data_left = draw_label(data_left, label_left)
        data_right = draw_label(data_right, label_right)
        pred_left_raw = draw_label(pred_left, warp_label_left)
        pred_closing = draw_label(pred_left, warp_label_left_closing)
        # voxel = np.load('/home/miakang/code/voxelmorph/data/mind/moved_6_11.npy')
        # np.save('./result/lpba_mind_' + str(i), pred_left)
        # lpba_mind = np.load('./result/lpba_mind_' + str(i) + '.npy')
        # np.save('./result/result_compare_mind/dual_prnet_plus.npy', pred_left_raw)
        # np.save('./result/result_compare_mind/dual_prnet_plus_closing.npy', pred_closing)
        for j in range(0, data_left.shape[2]):
            result = np.concatenate([data_left[..., j, :], data_right[..., j, :],  pred_closing[..., j, :], pred_left_raw[..., j, :]], axis=1)
            # result = np.concatenate([data_left[..., j, :], data_right[..., j, :]], axis=1)
            print(data_left.shape)
            result = cv2.resize(result, None, fx=2, fy=2)
            # delta = np.concatenate([o[j, ...] for o in offset_list], axis=1)
            # delta = cv2.resize(delta, None, fx=2, fy=2)
            #cv2.imshow("delta", delta)
            cv2.imwrite('./temp/result_%s.jpg' % j, result)
            # cv2.imshow("result", result)
            #cv2.imshow("flow", cv2.resize(flow[j, ...], None, fx=2, fy=2))
            # cv2.waitKey()
    sum = 0    
    for region, evaluator in zip(regions, evaluator_list):
        val = evaluator.Eval()
        print("%s: %.3f" % (region, val))
        sum += val
    avg = sum / len(regions)
    print('avg', avg)
    
