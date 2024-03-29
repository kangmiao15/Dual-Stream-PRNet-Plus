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
from op.warp_flow import add_offset, revert_offset, apply_offset
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
    print(spacing)
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
    last_flow = apply_offset(delta_list[-1])
    from net.cascade_fusion_net import permute_channel_last
    last_warp = F.grid_sample(data_warp, permute_channel_last(last_flow),
                     mode='bilinear', padding_mode="border")
    predict = predict.squeeze()
    predict = predict.cpu().numpy()
    last_warp = last_warp.squeeze()
    last_warp = last_warp.cpu().numpy()
    flow = flow.squeeze()

    return predict, flow, delta_list, last_warp

def draw_label(data, label):
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8
    data = np.stack((data, data, data), axis=-1)
    # 要画的点的坐标
    for i, point in enumerate(label):
        x = round(point[0])
        y = round(point[1])
        z = round(point[2])
        cv2.circle(data[x, :, :, :], (y,z), point_size, point_color, thickness)
        text = str(i) + '_' + 'x%s_y%s_z%s'% (x, y, z)
        cv2.putText(data[x, :, :, :], str(i), (y-20,z-10),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
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
    #rp_label_left build dataset
    trans_data = [ CropPadlandmarks(dst_size=(96,224,224)) ]
    # trans_data = [ Padlandmarks(dst_size=(96,256,256)) ]
    trans_pair = []
    data_list = [ ("case%d" % (args.fold) , "T%02d" % j) for j in [0, 50]]
    print(data_list)
    dataset_dir = DIRLabDataset(args.data_root, data_list, trans_data, with_label=False)
    dataset = PairDataset(dataset_dir, None, trans_pair, with_label=False, data_name='dirlab')
    data_left, data_right = dataset[0]
    # fetch data
    if args.fold in range(1, 6):
        lmk_path = args.data_root.replace('mha','') + 'Case%gPack/ExtremePhases/' % args.fold
        mov_lmk_fname = 'Case%g_300_T00_xyz.txt' % args.fold
        ref_lmk_fname = 'Case%g_300_T50_xyz.txt' % args.fold
    else:
        lmk_path = args.data_root.replace('mha', '') + 'Case%gPack/extremePhases/' % args.fold
        mov_lmk_fname = 'case%g_dirLab300_T00_xyz.txt' % args.fold
        ref_lmk_fname = 'case%g_dirLab300_T50_xyz.txt' % args.fold

    label_left_raw = np.loadtxt(os.path.join(lmk_path,mov_lmk_fname), dtype=int)
    label_right_raw = np.loadtxt(os.path.join(lmk_path,ref_lmk_fname), dtype=int)
    dir_info = dirlab_4dct_header()
    case_num = 'case' + str(args.fold)
    raw_tre = compute_tre(label_left_raw, label_right_raw, dir_info[case_num]['Spacing'])
    print('raw TRE-before reg, mean: {:.2f},std: {:.2f}'.format(
            raw_tre[0], raw_tre[1]))
    label_left_raw = np.stack([label_left_raw[:, 2], label_left_raw[:, 0], label_left_raw[:, 1]], axis=1)
    label_right_raw = np.stack([label_right_raw[:, 2], label_right_raw[:, 0], label_right_raw[:, 1]], axis=1)
    img_spacing = np.array([2.5, 1.0, 1.0])
    raw_shape = np.flip(dir_info[case_num]['Size'])
    resize_factor = dataset_dir.data_shape / raw_shape #new_spacing
    # resize the landmark
    label_left = (label_left_raw-1) * resize_factor - dataset_dir.delta
    label_right = (label_right_raw-1) * resize_factor- dataset_dir.delta
    # build network
    net, _ = build_network(args.net)
    print('# net parameters:', sum(param.numel() for param in net.parameters()))
    net.load_state_dict(torch.load(args.model_file, map_location="cpu"))
    if args.gpu:
        print('GPU')
        net = net.cuda()
    pred_left, flow, delta_list, last_warp = net_forward(net, data_left, data_right, label_right, args.gpu)
    flow = revert_offset(flow.unsqueeze(0))
    flow = flow.squeeze()
    # compute TRE
    ref_lmk_index = np.round(label_right).astype('int32')
    label_warp = label_right.copy()
    for i in range(300):
        di, hi, wi = ref_lmk_index[i]
        h0, w0, d0= delta_list[-1][0, :, di, hi, wi]
        label_warp[i] += [d0, h0, w0]
        
    # compute TRE
    #no reg
    tre_mean_bf, tre_std_bf, diff_br = compute_tre(label_left, label_right, img_spacing)
    print('TRE-before reg, mean: {:.2f},std: {:.2f}'.format(
            tre_mean_bf, tre_std_bf))
    #with reg
    tre_mean_af, tre_std_af, diff_ar = compute_tre(label_left, label_warp, img_spacing)
    print('TRE-after reg, mean: {:.2f},std: {:.2f}'.format(tre_mean_af, tre_std_af))
    if args.vis:
        # to numpy
        data_left = data_left.numpy()
        data_left = (data_left*255).astype(np.uint8)
        data_right = data_right.numpy()
        data_right = (data_right*255).astype(np.uint8)
        label_right = label_right.astype(np.uint32)
        label_left = label_left.astype(np.uint32)
        label_warp = label_warp.astype(np.uint32)
        pred_left = (pred_left*255).astype(np.uint8)
        last_warp = (last_warp*255).astype(np.uint8)
        with open('./data/4DCT/Case1Pack/Images/case1_T00_s.img', 'rb') as fid:
            data_left_raw = np.fromfile(fid, np.int16)
            data_left_raw = data_left_raw.reshape(dir_info[case_num]['Size'][::-1])
            data_left_raw = data_left_raw.astype(np.float32)
            data_left_raw = np.clip(data_left_raw, -1000, 400)
            data_left_raw = (data_left_raw-data_left_raw.min())/(data_left_raw.max()-data_left_raw.min())
            data_left_raw = (data_left_raw*255.0).astype(np.uint8)
        with open('./data/4DCT/Case1Pack/Images/case1_T50_s.img', 'rb') as fid:
            data_right_raw = np.fromfile(fid, np.int16)
            data_right_raw = data_right_raw.reshape(dir_info[case_num]['Size'][::-1])
            data_right_raw = data_right_raw.astype(np.float32)
            data_right_raw = np.clip(data_right_raw, -1000, 400)
            data_right_raw = (data_right_raw-data_right_raw.min())/(data_right_raw.max()-data_right_raw.min())
            data_right_raw = (data_right_raw*255.0).astype(np.uint8)
        # flow = vis_flow(flow)
        # offset_list = [ vis_flow(offset) for offset in offset_list ]
        data_left_raw = draw_label(data_left_raw, label_left_raw-1)
        data_right_raw = draw_label(data_right_raw, label_right_raw-1)
        data_left = draw_label(data_left, label_left)
        data_right = draw_label(data_right, label_right)
        data_warp = draw_label(pred_left, label_warp)
        last_warp = draw_label(last_warp, label_warp)
        for j in range(0, data_left.shape[0]):
            # result_raw = np.concatenate([data_left_raw[j, ...], data_right_raw[j, ...]], axis=1)
            result = np.concatenate([data_left[j, ...], data_right[j, ...], data_warp[j, ...], last_warp[j, ...]], axis=1)
            # result = np.concatenate([data_left[..., j, :], data_right[..., j, :]], axis=1)
            # result = cv2.resize(result, None, fx=2, fy=2)
            # result_raw = cv2.resize(result_raw, None, fx=2, fy=2)
            # delta = np.concatenate([o[j, ...] for o in offset_list], axis=1)
            # delta = cv2.resize(delta, None, fx=2, fy=2)
            #cv2.imshow("delta", delta)
            cv2.imwrite('./temp/dir_result_%s.jpg' % j, result)
            # cv2.imwrite('./temp/dir_result_raw_%s.jpg' % j, result_raw)
            # cv2.imshow("result", result)
            #cv2.imshow("flow", cv2.resize(flow[j, ...], None, fx=2, fy=2))
            # cv2.waitKey()

