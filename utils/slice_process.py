import os

import numpy as numpy
from scipy.ndimage import zoom
import SimpleITK as sitk

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    data_root = '/home/miakang/code/deformable_registration_bk/data/LPBA40/delineation_space/S31/S31.delineation.skullstripped.img.gz'
    out_root = '/home/miakang/code/deformable_registration_bk/data/LPBA40/SliceSpace4/S31'
    slice_space = 4
    # load raw data
    data = sitk.ReadImage(data_root)
    data = sitk.GetArrayFromImage(data)  
    print('data', data.shape)
    data_reduce = data[:, ::slice_space, :]
    data_interp = zoom(data_reduce, (1.0, float(slice_space), 1.0), order=1)
    make_dir(out_root)
    print('data_reduce', data_reduce.shape)
    data_reduce = sitk.GetImageFromArray(data_reduce)
    sitk.WriteImage(data_reduce, os.path.join(out_root, 'S31.img.gz'))
    data_interp = sitk.GetImageFromArray(data_interp)
    sitk.WriteImage(data_interp, os.path.join(out_root, 'S31_interp.img.gz'))
