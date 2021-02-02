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
from skimage.morphology import closing, square
import time

from dataset.lpba40_dataset import LPBA40Dataset
from dataset.mind101_dataset import Mind101Dataset
from dataset.pair_dataset import PairDataset
from op.warp_flow import revert_offset, apply_offset
from net.cascade_fusion_net import permute_channel_last
from utils.visualization import vis_flow, label2color
from evaluator import EvalDiceScore, EvalMeanDice
from preprocess import *
from net_factory import build_network3d as build_network

import pdb
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
    parser.add_argument("--is_cuda", help="use GPU", default=True, type=bool)
    parser.add_argument("--vis", help="visualization", default=False, type=bool)
    parser.add_argument("--vis_label", help="visualization with label", default=True, type=bool)
    

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
        # predict, flow, delta_list, delta_last_list = net(data)
        predict, flow, delta_list = net(data)
        # pdb.set_trace()
        # warp label
        label_warp = F.grid_sample(label_warp, flow.permute(0, 2, 3, 4, 1),
            mode='nearest', padding_mode="border")

        # scale offset
        offset_list = []
        scale_last_list = [4, 2, 1, 1, 1]
        scale_list = [8, 4, 2, 1, 1]
        warp_list = []
        for lvl, delta in enumerate(delta_list):
            scale = scale_list[lvl] 
            # scale_last = scale_last_list[lvl]
            delta = scale*F.interpolate(delta, scale_factor=scale, mode='trilinear')

            # delta_last = delta_last_list[lvl]
            # delta_last = F.interpolate(delta_last, scale_factor=scale_last, mode='trilinear')
            # x_warp = F.grid_sample(data_warp[None, None, ...], permute_channel_last(delta_last),
                    #  mode='bilinear', padding_mode="border")

            # x_warp = x_warp.squeeze()
            # x_warp = x_warp.cpu().numpy()
            # warp_list.append(x_warp)
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

    return predict, flow, offset_list, label_warp, warp_list


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
    data_list = [ ("S%02d" % (i+31)) for i in range(10) ]
    dataset = LPBA40Dataset(args.data_root, data_list, trans=trans_data, with_label=True)
    # data_list = os.listdir(args.data_root)
    # dataset = Mind101Dataset(args.data_root, data_list, trans=trans_data, with_label=True)
    dataset = PairDataset(dataset, None, trans_pair, with_label=True)
    num_pair = len(dataset)
    print("number of pair: %d" % num_pair)

    # build network
    net, _ = build_network(args.net)
    print('# net parameters:', sum(param.numel() for param in net.parameters()))
    net.load_state_dict(torch.load(args.model_file, map_location="cpu"))
    net = net.cuda()
    
    regions = [ "frontal_lobe", "parietal_lobe", "occipital_lobe", "temporal_lobe",
        "cingulate_lobe", "putamen", "hippocampus" ]
    
    # regions = [str(i) for i in range(21, 35)]+[str(i) for i in range(41, 51)]+['61', '62']+[str(i) for i in range(65, 69)]+[str(i) for i in range(81, 93)]+['101', '102']+['121', '122']+[str(i) for i in range(161, 167)]+['181', '182']
    
    evaluator_list = []
    evaluator_save = []
    for i in range(len(regions)):
        evaluator_list.append( EvalDiceScore() )
        evaluator_save.append( EvalMeanDice() )

    # testing 
    random_idx = list(range(num_pair))
    # random.shuffle(random_idx)
    for i in random_idx:
        print("processing %d" % i)
        start_time = time.clock()
        # fetch data
        data_left, data_right = dataset[i]
        data_left, label_left = data_left
        data_right, label_right = data_right
        
        pred_left, flow, offset_list, warp_label_left, warp_list= net_forward(net, data_left, data_right, label_left, args.is_cuda)

        end_time = time.clock()
        print('time is', end_time-start_time)

        # to numpy
        data_left = data_left.numpy()
        data_left_raw = data_left.copy()
        data_left = (data_left*255).astype(np.uint8)
        data_right = data_right.numpy()
        data_right_raw = data_right.copy()
        data_right = (data_right*255).astype(np.uint8)
        print(pred_left.max())
        pred_left_raw = pred_left.copy()
        pred_left = (pred_left*255).astype(np.uint8)
        print('after', pred_left.max())
        label_right = label_right.numpy().astype(np.uint8)
        label_left = label_left.numpy().astype(np.uint8)

        kernel = square(3)
        warp_label_left_closing = np.zeros_like(warp_label_left)
        for j in range(warp_label_left.shape[0]):
            data = warp_label_left[j, ...]
            data = closing(data, kernel)
            warp_label_left_closing[j, ...] = data

        # evaluate
        for j, evaluator in enumerate(evaluator_list):
            region_left = warp_label_left_closing == j+1
            region_right = label_right == j+1
            # region_left = warp_label_left == int(regions[i])
            # region_right = label_right == int(regions[i])
            evaluator.AddResult(region_left, region_right)

        for j, evaluator in enumerate(evaluator_save):
            region_left = warp_label_left_closing == j+1
            region_right = label_right == j+1
            evaluator.AddResult(region_left, region_right)

        if args.vis is False:
            continue
        
        # vis
        offset_list = [ vis_flow(offset) for offset in offset_list ]
        flow = vis_flow(flow)
        data_left = draw_label(data_left, label_left)
        print('data_left_vis', data_left.shape)
        data_right = draw_label(data_right, label_right)
        pred_left_before = draw_label(pred_left, warp_label_left)
        pred_left_closing = draw_label(pred_left, warp_label_left_closing)
        # np.save('./result/result_compare_lpba/dual_prnet_plus.npy', pred_left_before)
        # np.save('./result/result_compare_lpba/dual_prnet_plus_closing.npy', pred_left_closing)
        for j in range(data_left.shape[0]):
            # vox_result = np.concatenate((voxel_data_left[j, ...], voxel_data_right[j, ...], voxel_data_warp[j, ...]), axis=1)
            result = np.concatenate([data_left[j, :, :], data_right[j, :, :], pred_left_before[j, :, :], pred_left_closing[j, :, :]], axis=1)
            print(data_left.shape)
            # result = cv2.resize(result, None, fx=2, fy=2)
            delta = np.concatenate([o[j, ...] for o in offset_list] + [flow[j, ...]], axis=1)
            warps = np.concatenate([data_left_raw[j, :, :]]+[w[j, ...] for w in warp_list] + [pred_left_raw[j, ...]]+[data_right_raw[j, ...]], axis=1)
            # delta = cv2.resize(delta, None, fx=2, fy=2)
            cv2.imshow("delta", delta)
            cv2.imshow("result", result)
            cv2.imshow("warps", warps)
            # cv2.imshow("vox_result", vox_result)
            # cv2.imshow("flow", cv2.resize(flow[j, ...], None, fx=2, fy=2))
            k = cv2.waitKey()
            if k == 27:
                cv2.destroyAllWindows()
                break
              
    sum = 0    
    for region, evaluator in zip(regions, evaluator_list):
        val = evaluator.Eval()
        print("%s: %.3f" % (region, val))
        sum += val
    avg = sum / len(regions)
    print('avg', avg)

    for region, evaluator in zip(regions, evaluator_save):
        eval_list = evaluator.Eval()
        eval_array = np.array(eval_list, dtype=np.float32)
        # np.save('./result/diff_methods_on_lpba/LPBA40/dual_prnet_plus_cross_val/Mind_cross_dual_PRNet_plus_%s.npy' % region, eval_array)
