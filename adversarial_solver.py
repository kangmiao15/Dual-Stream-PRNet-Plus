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

from tensorboardX import SummaryWriter
from solver import CollateFn

 
class AdversarialSolver(object):

    def __init__(self, net_G, net_D, dataset, criterion, output_dir, world_size=1, rank=0):
        self.net_G = net_G
        self.net_D = net_D
        self.dataset = dataset
        self.output_dir = output_dir
        self.world_size = world_size
        self.rank = rank
        self.criterion_G = criterion

        #
        self.net_G.cuda()
        self.net_D.cuda()
        if world_size > 1:
            self.net_G_DDP = torch.nn.parallel.DistributedDataParallel(
                self.net_G,
                device_ids=[rank],
                output_device=rank
                )
            self.net_D_DDP = torch.nn.parallel.DistributedDataParallel(
                self.net_D,
                device_ids=[rank],
                output_device=rank
            )
        else:
            self.net_G_DDP = net_G
            self.net_D_DDP = net_D
        
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=1e-4)
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=1e-4)


        # training stage
        self.writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
        self.num_iter = 0
        self.num_epoch = 0


    def train_one_epoch(self, batch_size):
        self.net_G_DDP.train()
        self.net_D_DDP.train()

        batch_data = DataLoader(self.dataset, batch_size=batch_size, sampler=DistributedSampler(self.dataset),
                num_workers=1, collate_fn=CollateFn(), pin_memory=True,
                drop_last=True)

        max_iter = len(batch_data)

        for i_batch, (data_warp, data_fix) in enumerate(batch_data):

            self.num_iter += 1
            data_warp = data_warp.cuda()
            data_warp.requires_grad_()
            data_fix = data_fix.cuda()
            data_fix.requires_grad_()

            pred_warp, flow, delta_list = self.forward_G(data_warp, data_fix)

            # update D
            loss_D_real = self.update_D(data_warp, data_fix, pred_warp)
            self.optimizer_D.zero_grad()
            loss_D_real.backward()
            nn.utils.clip_grad_norm_(self.net_D.parameters(), 10.0)
            self.optimizer_D.step()

            # update G
            loss_D_fake, loss_smi, loss_smooth = self.update_G(data_fix, pred_warp, flow, delta_list)
            loss = loss_D_fake + loss_smi + loss_smooth
            self.optimizer_G.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net_G.parameters(), 10.0)
            self.optimizer_G.step()

            print('epoch %d, iter %d/%d' % (self.num_epoch, i_batch, max_iter))
            self.print_loss("loss_D_real", loss_D_real)
            self.print_loss("loss_D_fake", loss_D_fake)
            self.print_loss("loss_smi", loss_smi)
            self.print_loss("loss_smooth", loss_smooth)

        self.writer.file_writer.flush()
        self.num_epoch += 1

        return loss.item()
    

    def print_loss(self, name, loss,):
        self.writer.add_scalar(name, loss.item(), self.num_iter)
        print('\t%s: %f' % (name, loss.item()))
    

    def forward_G(self, data_warp, data_fix):
        data = torch.cat([data_warp, data_fix], dim=0)
        pred_warp, flow, delta_list = self.net_G_DDP(data)
        return pred_warp, flow, delta_list
    

    def update_D(self, data_warp, data_fix, pred_warp):
        batch_size = data_fix.size(0)
        data_dstb = 0.2*data_warp + 0.8*data_fix
        data_warp_dstb = 0.2*data_warp + 0.8*pred_warp
        data_neg = torch.cat([data_fix, data_warp_dstb], dim=1) 
        data_pos = torch.cat([data_fix, data_dstb], dim=1) 
        self.net_D_DDP.train()
        pred_D_fake = self.net_D_DDP(data_neg.detach())
        pred_D_real = self.net_D_DDP(data_pos.detach())
        target_D_fake = torch.tensor([0]*batch_size, 
            dtype=torch.float, device=data_fix.device)
        target_D_real = torch.tensor([1]*batch_size, 
            dtype=torch.float, device=data_fix.device)    
        loss_D_fake = F.binary_cross_entropy_with_logits(pred_D_fake.view(-1), target_D_fake)
        print('update_D loss_D_fake', loss_D_fake)
        loss_D_real = F.binary_cross_entropy_with_logits(pred_D_real.view(-1), target_D_real)
        return 0.5*loss_D_fake+0.5*loss_D_real
    

    def update_G(self, data_fix, pred_warp, flow, delta_list):
        batch_size = data_fix.size(0)
        loss_smi, loss_smooth = self.criterion_G(pred_warp, data_fix, flow, delta_list)
        data = torch.cat([data_fix, pred_warp], dim=1)
        self.net_D_DDP.eval()
        pred_D = self.net_D_DDP(data)
        target_D = torch.tensor([1]*batch_size,
            dtype=torch.float, device=data_fix.device)
        loss_D_fake = F.binary_cross_entropy_with_logits(pred_D.view(-1), target_D)
        print('update_G loss_D_fake', loss_D_fake)
        loss_D_fake = loss_D_fake * 0.01

        return loss_D_fake, loss_smi, loss_smooth


    def save_model(self):
        model_name = 'epoch_%04d.pt' % (self.num_epoch)
        save_path = os.path.join(self.output_dir, model_name)
        save_model = self.net_G
        torch.save(save_model.state_dict(), save_path)
        return save_path
