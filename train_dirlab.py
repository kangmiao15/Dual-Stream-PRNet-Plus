import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import cv2

from dataset.dirlab_dataset import DIRLabDataset
from dataset.pair_dataset import PairDataset
from op.losses import NCCLoss, grad_loss3d
from net_factory import build_network3d, build_discriminator
from solver import Solver
from adversarial_solver import AdversarialSolver
from preprocess import *
import warnings

warnings.filterwarnings("ignore")

def build_dataset(args):
    trans_data = [
            CenterCropDIR(dst_size=(96,224,224),rnd_offset=5)
            # CenterCrop(dst_size=(160,176,128), rnd_offset=5),
        ]
    trans_pair = [
        # RandomFlip(axes=[0,1,2])
    ]
    train_idx = [i for i in range(1, 11)]
    train_idx.pop(args.fold)
    data_list = [ ("case%d" % (i) , "T%02d" % j) for i in train_idx for j in range(0, 100, 10)]
    data_list.sort()
    val_data_list = [ ("case%d" % (args.fold) , "T%02d" % j) for j in range(0, 100, 10)]
    train_data = DIRLabDataset(args.train_data_root, data_list, trans=trans_data)
    train_data = PairDataset(train_data, None, trans_pair, data_name='dirlab')
    val_data = DIRLabDataset(args.train_data_root, val_data_list, with_label=True)
    val_data = PairDataset(val_data, None, trans_pair, data_name='dirlab')

    return train_data, val_data

def loss_fn(pred_warp, data_fix, flow, delta_list):
    loss_smi = NCCLoss(window=(9,9,9))(pred_warp, data_fix)
    loss_smooth = 0.05 * sum([ grad_loss3d(delta) for delta in delta_list ])
    return loss_smi, loss_smooth



def train(args, world_size, rank):    

    # make output folder
    output_dir = './output/dirlab_%s_%s' % (args.net, args.tag)
    if rank == 0:
        os.makedirs(os.path.join(output_dir), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'tensorboard'), exist_ok=True)

    # build network
    net, _ = build_network3d(args.net)
    #net_D = build_discriminator()
    print(net)
    print('# net parameters:', sum(param.numel() for param in net.parameters()))

    # build dataset
    train_data, val_data = build_dataset(args)

    # build solver
    criterion = loss_fn
    solver = Solver(net, train_data, args.batch_size, criterion, output_dir, world_size, rank)

    # start training
    for i_epoch in range(args.num_epoch):

        # train
        solver.train_one_epoch()
        i_epoch = solver.num_epoch

        if i_epoch % args.save_interval == 0 and rank == 0:
            save_path = solver.save_model()
            print('save model at %s' % save_path)

def test_dataset(args):
    # build dataset
    dataset, _ = build_dataset(args)

    num_slice = len(dataset)
    print("number of pair: %d" % num_slice)

    for i in range(num_slice):
        data_left, data_right = dataset[i]

        data_left = data_left.numpy()
        data_left = (data_left*255).astype(np.uint8)

        data_right = data_right.numpy()
        data_right = (data_right*255).astype(np.uint8)

        for j in range(data_left.shape[0]):
            cv2.imshow("frame_left", data_left[j, :, :])
            cv2.imshow("frame_right", data_right[j, :, :])
            cv2.waitKey(50)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train bone segmentation model')
    parser.add_argument( '--world_size', help='number of process', default=1, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--net", help="netname for network factory", type=str)
    parser.add_argument("--num_epoch", help="number of total epoch", default=300, type=int)
    parser.add_argument("--batch_size", help="batch size", default=1, type=int)
    parser.add_argument("--train_data_root", help="training data folder", type=str)
    parser.add_argument("--fold", help="fold for cross validation", default=0, type=int)
    parser.add_argument("--save_interval", help="save interval", default=5, type=str)
    parser.add_argument("--tag", help="output name", default='default', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # parse augmentations
    args = parse_args()

    world_size = args.world_size
    rank = args.local_rank
    torch.cuda.set_device(rank)

    # setup distribution
    if world_size > 1:
        dist.init_process_group(
                backend="nccl",
                init_method="env://", # TODO
                rank=rank
            )

    # start
    train(args, world_size, rank)
    #test_dataset(args)

