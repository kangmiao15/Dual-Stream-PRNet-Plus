import random
import math

import numpy as np 
import cv2
from scipy import ndimage


class CenterCrop:
    
    def __init__(self, dst_size, rnd_offset=-1):
        self.dst_size = dst_size
        self.rnd_offset = rnd_offset
    

    def __call__(self, volume):
        if type(volume) is list:
            org_D, org_H, org_W = volume[0].shape
        else:
            org_D, org_H, org_W = volume.shape
        dst_D, dst_H, dst_W = self.dst_size
        D_start, D_end = self.center_crop_idx(org_D, dst_D)
        H_start, H_end = self.center_crop_idx(org_H, dst_H)
        W_start, W_end = self.center_crop_idx(org_W, dst_W)
        if type(volume) is list:
            return [ v[D_start:D_end, H_start:H_end, W_start:W_end].copy() for v in volume ]
        else:
            return volume[D_start:D_end, H_start:H_end, W_start:W_end].copy()
        

    def center_crop_idx(self, org, dst):
        center = org/2.0
        if self.rnd_offset > 0:
            center = center + random.randint(-self.rnd_offset, self.rnd_offset)
        start = int(center - dst/2.0)
        end = int(center + dst/2.0)
        return start, end

class CenterCropDIR:
    
    def __init__(self, dst_size, rnd_offset=-1):
        self.dst_size = dst_size
        self.CropOffset = [[0,0,0],[0,-15,0],[0,0,0],[0,0,0],[0,10,0],
            [0,-20,20], [0,-20,10], [0,-65,0], [-16,-30,0], [-10,-20,0]]
        self.rnd_offset = rnd_offset
    

    def __call__(self, volume, case):
        dst_D, dst_H, dst_W = self.dst_size
        org_D, org_H, org_W = volume.shape
        if dst_D > org_D:
            pad = np.zeros((dst_D-org_D, volume.shape[1], volume.shape[2]), dtype=volume.dtype)
            volume = np.concatenate((volume, pad), axis=0)
        if dst_H > org_H:
            pad = np.zeros((volume.shape[0], dst_H-org_H, volume.shape[2]), dtype=volume.dtype)
            volume = np.concatenate((volume, pad), axis=1)
        if dst_W > org_W:
            pad = np.zeros((volume.shape[0], volume.shape[1], dst_W-org_W), dtype=volume.dtype)
            volume = np.concatenate((volume, pad), axis=2)
        org_D, org_H, org_W = volume.shape
        crop_offset = self.CropOffset[case]
        D_start, D_end = self.center_crop_idx(org_D, dst_D, crop_offset[0])
        H_start, H_end = self.center_crop_idx(org_H, dst_H, crop_offset[1])
        W_start, W_end = self.center_crop_idx(org_W, dst_W, crop_offset[2])
        if type(volume) is list:
            return [ v[D_start:D_end, H_start:H_end, W_start:W_end].copy() for v in volume ]
        else:
            return volume[D_start:D_end, H_start:H_end, W_start:W_end].copy()
        

    def center_crop_idx(self, org, dst, crop_offset):
        center = org/2.0 + crop_offset
        if (org - dst) > 2 * self.rnd_offset and self.rnd_offset > 0:
            rnd = random.randint(-self.rnd_offset, self.rnd_offset)
        else:
            rnd = 0
        print('rnd',rnd)
        center = center + rnd
        start = int(center - dst/2.0)
        end = int(center + dst/2.0)
        return start, end

class CenterCroplandmarks:
    
    def __init__(self, dst_size, rnd_offset=-1):
        self.dst_size = dst_size
        self.rnd_offset = rnd_offset
    

    def __call__(self, volume, landmark):
        org_D, org_H, org_W = volume.shape
        dst_D, dst_H, dst_W = self.dst_size
        D_start, D_end = self.center_crop_idx(org_D, dst_D)
        H_start, H_end = self.center_crop_idx(org_H, dst_H)
        W_start, W_end = self.center_crop_idx(org_W, dst_W)
        num_points = landmark.shape[0]
        new_landmarks = []

        for i in range(num_points):
            d = landmark[i, 0]
            h = landmark[i, 1]
            w = landmark[i, 2]
            cond1 = d < D_start or d >= D_end
            cond2 = h < H_start or h >= H_end
            cond3 = w < W_start or w >= W_end
            if cond1 or cond2 or cond3:
                print('elimilate the landmark:', (d, h, w))
            else:
                new_landmarks.append([d, h, w])
        new_landmarks = np.array(new_landmarks)
        new_landmarks = new_landmarks - np.array([D_start, H_start, W_start])
        return [ volume[D_start:D_end, H_start:H_end, W_start:W_end].copy(), new_landmarks]

    def center_crop_idx(self, org, dst):
        center = org/2.0
        if self.rnd_offset > 0:
            center = center + random.randint(-self.rnd_offset, self.rnd_offset)
        start = int(center - dst/2.0)
        end = int(center + dst/2.0)
        return start, end

class Padlandmarks:
    
    def __init__(self, dst_size):
        self.dst_size = dst_size
    
    def __call__(self, volume, landmarks):
        print('before', volume.shape)
        org_D, org_H, org_W = volume.shape
        dst_D, dst_H, dst_W = self.dst_size
        if dst_D > org_D:
            pad = np.zeros((dst_D-org_D, volume.shape[1], volume.shape[2]), dtype=volume.dtype)
            volume = np.concatenate((volume, pad), axis=0)
        if dst_H > org_H:
            pad = np.zeros((volume.shape[0], dst_H-org_H, volume.shape[2]), dtype=volume.dtype)
            volume = np.concatenate((volume, pad), axis=1)
        if dst_W > org_W:
            pad = np.zeros((volume.shape[0], volume.shape[1], dst_W-org_W), dtype=volume.dtype)
            volume = np.concatenate((volume, pad), axis=2)
        print('after', volume.shape)
        return volume,landmarks

class CropPadlandmarks:
    
    def __init__(self, dst_size, rnd_offset=-1):
        self.dst_size = dst_size
        self.rnd_offset = rnd_offset
    

    def __call__(self, volume):
        org_D, org_H, org_W = volume.shape
        dst_D, dst_H, dst_W = self.dst_size
        if dst_D > org_D:
            pad = np.zeros((dst_D-org_D, volume.shape[1], volume.shape[2]), dtype=volume.dtype)
            volume = np.concatenate((volume, pad), axis=0)
        if dst_H > org_H:
            pad = np.zeros((volume.shape[0], dst_H-org_H, volume.shape[2]), dtype=volume.dtype)
            volume = np.concatenate((volume, pad), axis=1)
        if dst_W > org_W:
            pad = np.zeros((volume.shape[0], volume.shape[1], dst_W-org_W), dtype=volume.dtype)
            volume = np.concatenate((volume, pad), axis=2)
        org_D, org_H, org_W = volume.shape
        D_start, D_end = self.center_crop_idx(org_D, dst_D) 
        H_start, H_end = self.center_crop_idx(org_H, dst_H)
        W_start, W_end = self.center_crop_idx(org_W, dst_W)
        # num_points = landmark.shape[0]
        # new_landmarks = []

        # for i in range(num_points):
        #     d = landmark[i, 0]
        #     h = landmark[i, 1]
        #     w = landmark[i, 2]
        #     cond1 = d < D_start or d >= (D_end-1)
        #     cond2 = h < H_start or h >= (H_end-1)
        #     cond3 = w < W_start or w >= (W_end-1)
        #     if cond1 or cond2 or cond3:
        #         print('elimilate the landmark:', (d, h, w))
        #     else:
        #         new_landmarks.append([d, h, w])
        # new_landmarks = np.array(new_landmarks)
        # new_landmarks = new_landmarks - np.array([D_start, H_start, W_start])
        print([D_start, H_start, W_start])
        return [ volume[D_start:D_end, H_start:H_end, W_start:W_end].copy(), [D_start, H_start, W_start]]

    def center_crop_idx(self, org, dst):
        center = org/2.0
        if self.rnd_offset > 0:
            center = center + random.randint(-self.rnd_offset, self.rnd_offset)
        start = int(center - dst/2.0)
        end = int(center + dst/2.0)
        return start, end

class RandomRotate3D:

    def __init__(self):
        pass

    def __call__(self, volume):
        axes = np.random.choice([0,1,2], size=2, replace=False)
        rotate = random.randint(0, 3)
        if rotate == 0:
            return volume
        if type(volume) is list:
            volume = [ np.rot90(v, k=rotate, axes=axes) for v in volume ]
            volume = [ v.copy() for v in volume ]
        else:
            volume = np.rot90(volume, k=rotate, axes=axes)
            volume = volume.copy()
        return volume
    

class RandomFlip:

    def __init__(self, axes):
        self.axes = axes + [-1]

    def __call__(self, volume):
        axis = np.random.choice(self.axes, replace=False)
        if axis < 0:
            return volume
        if type(volume) is list:
            volume = [ np.flip(v, axis=axis) for v in volume]
            volume = [ v.copy() for v in volume ]
        else:
            volume = np.flip(volume, axis=axis)
            volume = volume.copy()
        return volume


class RandomDisturb:

    def __init__(self, max_offset=2):
        pass


class ReColor:
    def __init__(self, alpha=0.05, beta=0.1):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, im):
        num_chns = im.shape[3]
        # random amplify each channel
        t = np.random.uniform(-1, 1, num_chns)
        im = im.astype(np.float32)
        im *= (1 + t * self.alpha)
        im = np.clip(im, 0, 1.0)
        # random gama
        gamma = random.uniform(-self.beta, self.beta) + 1.0
        im = np.power(im, gamma)
        return im



class SampleVolume:
    def __init__(self, dst_shape=[96, 96, 5], num_classes=-1, prob=None):
        self.dst_shape = dst_shape
        self.num_classes = num_classes
        self.prob = np.array(prob)

    def __call__(self, data, label):
        src_h,src_w,src_d,_ = data.shape
        dst_h,dst_w,dst_d = self.dst_shape
        if type(dst_d) is list:
            dst_d = random.choice(dst_d)
        if self.num_classes<=0:
            h = random.randint(0, src_h-dst_h)
            w = random.randint(0, src_w-dst_w)
            d = random.randint(0, src_d-dst_d)
        else:
            select = label == np.random.choice(self.num_classes,  p=self.prob)
            h, w, d = np.where(select)
            if len(h) > 0:
                select_idx = random.randint(0, len(h)-1)
                h = h[select_idx] - int(dst_h/2)
                w = w[select_idx] - int(dst_w/2)
                d = d[select_idx] - int(dst_d/2)
                h = min(max(h,0), src_h-dst_h)
                w = min(max(w,0), src_w-dst_w)
                d = min(max(d,0), src_d-dst_d)
            else:
                h = random.randint(0, src_h-dst_h)
                w = random.randint(0, src_w-dst_w)
                d = random.randint(0, src_d-dst_d)

        sub_volume = data[h:h+dst_h,w:w+dst_w,d:d+dst_d,:]
        sub_label = label[h:h+dst_h,w:w+dst_w,d:d+dst_d]
        return sub_volume,sub_label


class RandomNoise:
    def __init__(self, norm, mean=0):
        self.norm = norm
        self.mean = mean

    def __call__(self, im):
        mean = 2*random.random()*self.mean - self.mean
        noise = np.random.normal(loc=mean, scale=self.norm, size=im.shape)
        return im + noise


class RandomCrop:
    def __init__(self, crop_size, rotation=False):
        self.crop_size = crop_size

    def __call__(self, im, mask):
        x = random.randint(0, im.shape[1]-self.crop_size[0])
        y = random.randint(0, im.shape[0]-self.crop_size[1])
        crop_im = im[y:y+self.crop_size[1], x:x+self.crop_size[0], :]
        crop_mask = mask[y:y+self.crop_size[1], x:x+self.crop_size[0]]
        return crop_im, crop_mask


class RandomRotate:
    def __init__(self, random_hflip=True, random_dflip=True):
        self.random_hflip = random_hflip
        self.random_dflip = random_dflip

    def __call__(self, im, mask):
        H, W, D, C = im.shape
        if self.random_dflip and random.random() > 0.5:
            im = im[:, :, ::-1, :]
            mask = mask[:, :, ::-1]
        if self.random_hflip and random.random() > 0.5:
            im = im[:, ::-1, :, :]
            mask = mask[:, ::-1, :]
        rotate = random.randint(0, 3)
        if rotate > 0:
            im = np.rot90(im, rotate)
            mask = np.rot90(mask, rotate)
        return im.copy(), mask.copy()

def ReSample(image, old_spacing, new_spacing):
    '''
    resample the image from original spatial resolution to the given new_spacing.
    default: 3-order spline interpolation.
    :params:
        image: shape(height, width, channel)
        old_spacing: the old spacing of the input image, shape(depth,height,width),
        new_spacing: the new spacing of the output image, shape(depth,height,width).
    '''
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    image = ndimage.interpolation.zoom(image, real_resize_factor, order=4, mode='nearest')

    return image, new_spacing



