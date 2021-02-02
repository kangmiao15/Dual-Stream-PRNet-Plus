import os
import random
import sys

import cv2
import numpy as np

import SimpleITK as sitk
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

#FIAM
GROUP_CONFIG = [
    ("frontal_lobe", 21, 34),
    ("parietal_lobe", 41, 50),
    ("occipital_lobe", 61, 68),
    ("temporal_lobe", 81, 92),
    ("cingulate_lobe", 101, 122),
    ("putamen", 163, 166),
    ("hippocampus", 181, 182)
]

def group_label(label):
    label_merged = np.zeros(label.shape, dtype=np.int32)
    for i, (name, start, end) in enumerate(GROUP_CONFIG):
        region = np.logical_and(label>=start, label>=end)
        label_merged[region] = i+1
    return label_merged

class LPBA40Dataset(Dataset):

    def __init__(self, data_root, data_list, trans=[], with_label=False):
        self.data_root = data_root
        self.data_list = data_list
        self.trans = trans
        self.with_label = with_label
    

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        data_id = self.data_list[index]
        data_path = os.path.join(self.data_root, data_id, "%s.delineation.skullstripped.img.gz" % data_id)
        label_path = os.path.join(self.data_root, data_id, "%s.delineation.structure.label.hdr" % data_id)
        # load raw data
        data = sitk.ReadImage(data_path)
        data = sitk.GetArrayFromImage(data)

        # normalization
        data = data.astype(np.float32)
        data = data/data.max()

        if self.with_label:
            label = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(label)
            label = group_label(label)
            data = [ data, label ]

        # transform
        for trans in self.trans:
            data = trans(data)
        
        return data


if __name__ == '__main__':
    data_root = sys.argv[1]
    data_list = [ ("S%02d" % (i+1)) for i in range(40) ]

    dataset = LPBA40Dataset(data_root, data_list)

    num_data = len(dataset)

    for i in range(num_data):
        data = dataset[i]
        for j in range(data.shape[0]):
            frame = data[j, :, :]
            frame = (255*frame).astype(np.uint8)
            cv2.imshow("frame", frame)
            cv2.waitKey(50)
        print(data.shape)
