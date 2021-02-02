import sys
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data import sampler
import numpy as np

from op.warp_flow import apply_offset
from tricks import WarmUpLR
from tensorboardX import SummaryWriter

class CollateFn:
    # CollateFn for unsupervised training (without label)
    def __init__(self):
        pass

    def __call__(self, batch_data):
        left_img_list = []
        left_label_list = []
        right_img_list = []
        right_label_list = []
        for left, right in batch_data:
            left_img_list.append(left[0])
            left_label_list.append(left[1])
            right_img_list.append(right[0])
            right_label_list.append(right[1])

        img_left = torch.stack(left_img_list, dim=0)
        label_left = torch.stack(left_label_list, dim=0)
        img_right = torch.stack(right_img_list, dim=0)
        label_right = torch.stack(right_label_list, dim=0)

        img_left = img_left.unsqueeze(1)
        label_left = label_left.unsqueeze(1)
        img_right = img_right.unsqueeze(1)
        label_right = label_right.unsqueeze(1)

        return img_left, label_left, img_right, label_right

class Solver(object):

    def __init__(self, net, dataset, batch_size, criterion, output_dir, world_size=1, rank=0):
        self.net = net
        self.dataset = dataset
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.world_size = world_size
        self.rank = rank
        self.criterion = criterion

        self.net.cuda()
        if world_size > 1:
            self.net_DDP = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[rank],
                output_device=rank
                )
            self.batch_data = DataLoader(self.dataset, batch_size=self.batch_size, sampler=DistributedSampler(self.dataset),
                num_workers=1, collate_fn=CollateFn(), pin_memory=True, drop_last=True)
        else:
            self.net_DDP = net
            self.batch_data = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=1, collate_fn=CollateFn(), pin_memory=True, drop_last=True)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=1e-5)
        self.warm_epoch = 5
        self.max_iter = len(self.batch_data)
        self.warmup_scheduler = WarmUpLR(self.optimizer, self.max_iter * self.warm_epoch)


        # training stage
        self.writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
        self.num_iter = 0
        self.num_epoch = 0


    def train_one_epoch(self, iter_size=1):
        self.net.train()
        max_iter = len(self.batch_data)

        for i_batch, (data_warp, label_warp, data_fix, label_fix) in enumerate(self.batch_data):
            self.num_iter += 1
            data_warp = data_warp.cuda()
            data_warp.requires_grad_()
            data_fix = data_fix.cuda()
            data_fix.requires_grad_()
            label_warp = label_warp.cuda()
            label_warp.requires_grad_()
            label_fix = label_fix.cuda()
            label_fix.requires_grad_()
            data = torch.cat([data_warp, data_fix], dim=0)
            
            # forward
            # import pdb; pdb.set_trace()
            pred_warp, flow, delta_list = self.net(data)
            # warp label
            label_warp = F.grid_sample(label_warp, flow.permute(0, 2, 3, 4, 1),
            mode='nearest', padding_mode="border")
            losses = self.criterion(pred_warp, data_fix, label_warp, label_fix, flow, delta_list)
            if type(losses) is tuple or type(losses) is list:
                total_loss = None
                for i, loss in enumerate(losses):
                    loss = loss.mean()
                    if total_loss is None:
                        total_loss = loss
                    else:
                        total_loss += loss
                    self.writer.add_scalar('loss%d' % i, loss.item(), self.num_iter)
                    print("loss%d: %.6f" % (i, loss.item()))
                loss = total_loss
            else:
                loss = losses.mean()

            self.writer.add_scalar('loss', loss.item(), self.num_iter)
            print('epoch %d, iter %d/%d, loss: %f' % 
                    (self.num_epoch, i_batch, max_iter, loss.item()))

            # backward
            if not math.isnan(loss.item()):
                loss.backward()
            if i_batch % iter_size == 0:
                nn.utils.clip_grad_norm_(self.net.parameters(), 10)
                self.optimizer.step()
                self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            if self.num_epoch <= self.warm_epoch:
                self.warmup_scheduler.step()
        self.writer.file_writer.flush()
        self.num_epoch += 1

        return loss.item()


    def save_model(self):
        model_name = 'epoch_%04d.pt' % (self.num_epoch)
        save_path = os.path.join(self.output_dir, model_name)
        save_model = self.net
        torch.save(save_model.state_dict(), save_path)
        return save_path
