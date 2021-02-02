import os
import random
import sys

import cv2
import numpy as np

import SimpleITK as sitk
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


class DIRLabDataset(Dataset):

    def __init__(self, data_root, data_list, trans=[], with_label=False):
        self.data_root = data_root
        self.data_list = data_list
        self.trans = trans
        self.with_label = with_label
    

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        data_id, data_time = self.data_list[index]
        data_path = os.path.join(self.data_root, data_id, "%s_%s_RS1.mha" % (data_id, data_time))
        # label_path = os.path.join(self.data_root.replace('mha', 'points'), data_id, "%s_4D-75_%s_xyz_tr.txt" % (data_id, data_time))
        label_path = os.path.join(self.data_root.replace('mha/', ''), data_id.capitalize()+'Pack', 'ExtremePhases', "%s_300_%s_xyz.txt" % (data_id.capitalize(), data_time))
        print(label_path)
        # load raw data
        data = sitk.ReadImage(data_path)
        data = sitk.GetArrayFromImage(data)
        # normalization
        data = data.astype(np.float32)
        data = np.clip(data, -1000, 400)
        data = (data-data.min())/(data.max()-data.min())

        if self.with_label:
            label = np.loadtxt(label_path)
            # change to (D, H, W) as data shape
            label = np.stack([label[:, 2], label[:, 0], label[:, 1]], axis=1)
            for trans in self.trans:
                data, label = trans(data, label)
            data = [ data, label ]
        else:
            for trans in self.trans:
                data = trans(data)
        return data


if __name__ == '__main__':
    data_root = sys.argv[1]
    import sys
    sys.path.append('/data/guosheng/dual_prnet_plus_rev')
    from preprocess import *
    train_idx = [i for i in range(1, 11)]
    fold = 0
    train_idx.pop(0)
    data_list = [ ("case%d" % (i) , "T%02d" % j) for i in train_idx for j in range(0, 100, 10)]
    data_list.sort()
    trans_data = [
            CenterCrop(dst_size=(96,224,224), rnd_offset=2),
        ]
    dataset = DIRLabDataset(data_root, data_list, trans=trans_data)
    num_data = len(dataset)
    for i in range(num_data):
        data = dataset[i]
        print(data.shape)
        # for j in range(data.shape[0]):
        #     frame = data[j, :, :]
        #     frame = (255*frame).astype(np.uint8)
        #     cv2.imwrite('./temp/%s_%s.jpg' % (i, j), frame)
            # cv2.imshow("frame", frame)
            # cv2.waitKey(50)
