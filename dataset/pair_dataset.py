import os
import sys
import random

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset 


class PairDataset(Dataset):

    def __init__(self, backend_left, backend_right=None, trans=[], with_label=False, data_name=None):
        if backend_right is None:
            backend_right = backend_left
            self.symmetry = True
        else:
            self.symmetry = False
        self.backend_left = backend_left
        self.backend_right = backend_right
        self.trans = trans
        self.data_name = data_name
        self.data_len_left = len(backend_left)
        self.data_len_right = len(backend_right)
        self.with_label = with_label

  
    def __len__(self):
        if self.data_name == 'dirlab':
            return int(self.data_len_left*9)
        if self.symmetry:
            return self.data_len_left*(self.data_len_left-1)
        else:
            return self.data_len_left*self.data_len_right

    def to_pair_index(self, index):
        if self.data_name == "dirlab":
            index_left = int(index // 9)
            idx = index - index_left * 9
            index_right = (index_left//10)*10 + idx
            index_right = index_right if index_right < index_left else index_right+1
            return index_left, index_right
        if self.symmetry:
            index_left = int(index / (self.data_len_left-1))
            index_right = index % (self.data_len_left-1)
            index_right = index_right if index_right < index_left else index_right+1
        else:
            index_left = int(index/(self.data_len_right))
            index_right = index % self.data_len_right
        return index_left, index_right
    
    def to_pair_index_bk(self, index):
        if self.symmetry:
            index_left = int(index / (self.data_len_left-1))
            index_right = index % (self.data_len_left-1)
            index_right = index_right if index_right < index_left else index_right+1
        else:
            index_left = int(index/(self.data_len_right))
            index_right = index % self.data_len_right
        return index_left, index_right
    

    def __getitem__(self, index):
        index_left, index_right = self.to_pair_index(index)
        data_left = self.backend_left[index_left]
        data_right = self.backend_right[index_right]
        
        if self.with_label:
            data = data_left + data_right
        else:
            data = [data_left, data_right]
        for trans in self.trans:
            data = trans(data)
        data = [ torch.from_numpy(d).float() for d in data ]

        if self.with_label:
            data_left = data[:2]
            data_right = data[2:]
        else:
            data_left = data[0]
            data_right = data[1]

        return data_left, data_right