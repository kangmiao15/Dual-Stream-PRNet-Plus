import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def apply_offset(offset):
    '''
        convert offset grid to location grid
        offset: [N, 2, H, W] for 2D or [N, 3, D, H, W] for 3D
        output: [N, H, W, 2] for 2D or [N, D, H, W, 3] for 3D
    '''
    sizes = list(offset.size()[2:]) # [D, H, W] or [H, W]
    grid_list = torch.meshgrid([ torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)

    # apply offset
    grid_list = [ grid.float().unsqueeze(0) + offset[:, dim, ...]  
        for dim, grid in enumerate(grid_list) ]

    # normalize
    grid_list = [ grid / ((size-1.0)/2.0) - 1.0
        for grid, size in zip(grid_list, reversed(sizes))] 
    return torch.stack(grid_list, dim=-1)


def revert_offset(grid):
    '''
        convert location grid to offset grid
        grid: [N, 2, H, W] for 2D or [N, 3, D, H, W] for 3D
        output: [N, 2, H, W] for 2D or [N, 3, D, H, W] for 3D
    '''
    sizes = list(grid.size()[2:]) # [D, H, W] or [H, W]
    loc_list = torch.meshgrid([ torch.arange(size, device=grid.device) for size in sizes])
    loc_list = reversed(loc_list)

    grid_list = torch.split(grid, 1, dim=1)

    # unnormalize
    grid_list = [ (grid+1.0) * ((size-1.0)/2.0)
        for grid, size in zip(grid_list, reversed(sizes))] 

    # revert offset
    grid_list = [ grid - loc.float().unsqueeze(0)
        for grid, loc in zip(grid_list, loc_list) ]

    return torch.cat([grid.unsqueeze(1) for grid in grid_list], dim=1)

def add_offset(grid):
    '''
        add offset to grid
        grid: [N, 2, H, W] for 2D or [N, 3, D, H, W] for 3D
        output: [N, 2, H, W] for 2D or [N, 3, D, H, W] for 3D
    '''
    sizes = list(grid.size()[2:]) # [D, H, W] or [H, W]

    grid_list = torch.split(grid, 1, dim=1)

    # unnormalize
    grid_list = [ (grid+1.0) * ((size-1.0)/2.0)
        for grid, size in zip(grid_list, reversed(sizes))] 

    return torch.cat([grid.unsqueeze(1) for grid in grid_list], dim=1)

def warp_flow(src, flow, padding_mode="zeros"):
    '''
        bottom: [N, C, H, W] for 2D or [N, C, D, H, W] for 3D
        flow: [N, 2, H, W] for 2D or [N, 3, D, H, W] for 3D
    '''
    grid = apply_offset(flow)
    warp = F.grid_sample(src, grid, mode='bilinear', padding_mode=padding_mode)
    return warp


def scale_flow(flow, scale):
    return scale*F.interpolate(flow, scale_factor=scale, mode='bilinear')


def warp_landmarks(landmarks, offset):
    '''
        warp landmarks with offset 
        landmarks: [M, 2] for 2D or [M, 3] for 3D (N_batch * M_points * xyz)
        offset: [N, 2, H, W] for 2D or [N, 3, D, H, W] for 3D
        output: [N, M, 2] for 2D or [N, M, 3] for 3D
    '''
    assert(landmarks.shape[-1] == offset.shape[1]) 
    landmarks = landmarks.unsqueeze(0)

    sizes = list(offset.size()[2:]) # [D, H, W] or [H, W]
    offset_list = torch.split(offset, 1, dim=1)

    # unnormalize
    offset_list = [ (offs+1.0) * ((size-1.0)/2.0)
        for offs, size in zip(offset_list, reversed(sizes))] 
    # revert offset
    offset = torch.cat([offs for offs in offset_list], dim=1)
    
    new_landmarks = torch.zeros(landmarks.shape)
    num = landmarks.shape[1]
    if landmarks.shape[-1] == 3:
        for i in range(num):
            x = int(landmarks[:, i, 0])
            y = int(landmarks[:, i, 1])
            z = int(landmarks[:, i, 2])
            new_landmarks[:, i, :] = offset[:, :, x, y, z]
            # new_landmarks[:, i, 0] = offset[:, 0, x, y, z]
            # new_landmarks[:, i, 1] = offset[:, 1, x, y, z]
            # new_landmarks[:, i, 2] = offset[:, 2, x, y, z]
    else:
        print("offset channels != 3")
        raise NotImplementedError
    new_landmarks = torch.stack([new_landmarks[:, :, 2], new_landmarks[:, :, 1], new_landmarks[:, :, 0]], dim=2)
    return new_landmarks.squeeze(0)

if __name__ == '__main__':
    '''
        test unit
    '''
    import sys
    import random
    import cv2
    import numpy as np

    offset = torch.tensor(np.random.random(2, 3, 4, 5, 7))
    points = torch.tensor(np.random.randint(0, 5, (2, 6, 3)))
    new_points = warp_landmarks(points, offset)
    print(new_points.shape)
    '''
    org_im = cv2.imread(sys.argv[1])

    # to tensor
    im = torch.from_numpy(org_im).float()
    im = im.permute(2, 0, 1) # H, W, C - > C, H, W
    im = im.unsqueeze(0)
    N, C, H, W = im.size()

    while True:
        offset_x = random.random() * 64 - 32
        offset_y = random.random() * 64 - 32
        flow = torch.zeros((1, 2, H, W), device=im.device)
        flow[:, 0, :, :] = flow[:, 0, :, :] + offset_x
        flow[:, 1, :, :] = flow[:, 1, :, :] + offset_y
        print( "dx: %.3f, dy: %.3f" % (offset_x, offset_y))

        warp_im = warp_flow(im, flow)

        # to numpy
        warp_im = warp_im.squeeze()
        warp_im = warp_im.permute(1, 2, 0) # C, H, W -> H, W, C
        warp_im = warp_im.cpu().numpy()
        warp_im = warp_im.astype(np.uint8)

        cv2.imshow("org", org_im)
        cv2.imshow("warp", warp_im)
        cv2.waitKey()
        '''