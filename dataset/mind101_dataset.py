import os
import sys
import random

import nibabel as nib
import cv2
import numpy as np
from scipy.ndimage import zoom

import torch
from torch.utils.data import Dataset 

#label 25
GROUP_CONFIG = {
        'parietal_lobe_1': ['postcentral', 'supramarginal', 'superior parietal',  'inferior parietal', ' precuneus'],
        'frontal_lobe_2': ['superior frontal', 'middle frontal', 'inferior frontal', 'lateral orbitofrontal', 'medial orbitofrontal', 'precentral', 'paracentral'],
        'occipital_lobe_3': ['lingual', 'pericalcarine', 'cuneus',  'lateral occipital'],
        'temporal_lobe_4': ['entorhinal', 'parahippocampal', 'fusiform', 'superior temporal', 'middle temporal',  'inferior temporal', 'transverse temporal'],
        'cingulate_lobe_5': ['cingulate', 'insula'],
    }
#label 31
'''
GROUP_CONFIG = {
        'parietal_lobe_1': ['postcentral', 'supramarginal', 'superior parietal',  'inferior parietal', ' precuneus'],
        'frontal_lobe_2': ['caudal middle frontal', 'lateral orbitofrontal', 'medial orbitofrontal', 'paracentral', 'pars opercularis', 'pars orbitalis', 'pars triangularis', 'precentral', 'rostral middle frontal', 'superior frontal'],
        'occipital_lobe_3': ['lingual', 'pericalcarine', 'cuneus',  'lateral occipital'],
        'temporal_lobe_4': ['entorhinal', 'parahippocampal', 'fusiform', 'superior temporal', 'middle temporal',  'inferior temporal', 'transverse temporal'],
        'cingulate_lobe_5': ['caudal anterior cingulate', 'insula', 'isthmus cingulate', 'posterior cingulate', 'rostral anterior cingulate'],
    }
'''

def group_label(label):
    label_name = './dataset/mind101_label_25.txt'
    d = {}
    with open(label_name) as f:
        for line in f:
            if line != '\r\n':
                (value, key) = line.strip().split(',')
                d[key.strip().strip('"')] = int(value)
    label_merged = np.zeros(label.shape, dtype=np.int32)
    region = np.zeros(label.shape, dtype=bool)
     
    for key in GROUP_CONFIG:
        for structure in GROUP_CONFIG[key]:
            left_num = d['left '+structure.strip()]
            right_num = d['right '+structure.strip()]
            region = np.logical_or(region, np.logical_or(label==left_num, label==right_num))
        label_num = int(key.split('_')[-1])
        label_merged[region] = label_num
        region = np.zeros(label.shape, dtype=bool)
    return label_merged

class Mind101Dataset(Dataset):

    def __init__(self, data_root, data_list, trans=[], with_label=False):
        self.data_root = data_root
        self.data_list = data_list
        self.trans = trans
        self.with_label = with_label
        self.count = 0
    

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        data_id = self.data_list[index]
        data_path = os.path.join(self.data_root, data_id, "t1weighted_brain.MNI152.nii.gz")
        label_path = os.path.join(self.data_root, data_id, "labels.DKT25.manual.MNI152.nii.gz")
        print(data_path)
        # load raw data
        data = nib.load(data_path)
        data = data.get_data()

        # normalization
        data = data.astype(np.float32)
        data = data/data.max()
        if self.with_label:
            label = nib.load(label_path)
            label = label.get_data()
            label = group_label(label)
            data = [ data, label ]

        # transform
        for trans in self.trans:
            data = trans(data)

        return data


if __name__ == '__main__':
    data_root = sys.argv[1]
    data_list = os.listdir(data_root) 

    dataset = Mind101Dataset(data_root, data_list)

    num_data = len(dataset)

    for i in range(num_data):
        data = dataset[i]
        for j in range(data.shape[-1]):
            frame = data[:, :, j]
            frame = (255*frame).astype(np.uint8)
            cv2.imshow("frame", frame)
            cv2.waitKey(50)
        print(data.shape)
