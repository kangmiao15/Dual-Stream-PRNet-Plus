import tensorflow as tf
import itertools
import numpy as np
import keras
import keras.backend as K
import os
tf.compat.v1.disable_eager_execution()

def prod_n(lst):
    # lst.shape = [3, 160, 192, 224]
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod

def sub2ind(siz, subs, **kwargs):
    """
    assumes column-order major
    """
    # subs is a list
    # siz = [160, 192, 224]
    # subs.shape = [3, 160, 192, 224]
    assert len(siz) == len(subs), 'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])
    # siz[::-1] 由[160, 192, 224]变为[224, 192, 160]，k=[224,43008,6881280]

    # 在上层函数interpn中，我们需要得到idx也就是本函数中的ndx去得到偏移后的vol_val, 在上一层函数中，我们将vol展开得到_vol(shape=[6881280,1])
    # 上层函数中vol_val = tf.gather(_vol, idx)，因此我们需要idx作为索引得到偏移后的vol_val。vol[0][0][0] = _vol[idx[0][0][0]]
    # idx为偏移矩阵，其每个元素idx[i][j][k]都对应一个key，例如：idx[0][0][0]=12, 那么vol[0][0][0]=_vol[12]. 
    # 那么如何得到偏移矩阵。 首先我们看vol是如何展开的，shape=[nx, ny, nz], e.g. [[[1,2],[3,4]][[5,6],[7,8]]] -> shape=(2,2,2),展开时先展Z方向，再展Y方向，最后是X方向。
    # 得到[[1],[2],[3],[4],[5],[6],[7],[8]]，因此我们求偏移矩阵时也要根据这个特点，先是z，再y，最后x


    ndx = subs[-1]
    # ndx.shape=[160, 192, 224], ndx是z方向的shift


    # print(len(subs[:-1][::-1]))
    # print(subs[0][::-1].shape)
    # subs[:-1][::-1] -> size [2, 160, 192, 224]
    # subs[:-1]得到的shape=[2， 160， 192， 224], subs[:-1][0]是x, subs[:-1][1]是y，所以subs[:-1]对应的是[x,y], subs[:-1][::-1]对应的是[y,x]
    
    for i, v in enumerate(subs[:-1][::-1]):
        # 根据上面所说的求偏移矩阵，第一个ndx对应的是z方向，然后添加y方向(224 * y), 最后添加x方向(224*192*x).
        # 举例，subs[:,0,0,0] = [11,13,15], 分别对应x,y,z, 这就是说[0][0][0]点对应的实际是vol中的[11,13,15],因为vol被展开成_vol(shape=[6881280,1])
        # 因此将[11,13,15]转成15+13*224+11*224*192=476015，也就是说ndx[0][0][0]对应的是_vol[476015], --->最后得到偏移矩阵
        ndx = ndx + v * k[i]
    # ndx.shape = [160, 192, 224]
    return ndx

def interpn(vol, loc, interp_method='linear'):
    """
    N-D gridded interpolation in tensorflow
    vol can have more dimensioas_listns than loc[i], in which case loc[i] acts as a slice 
    for the first dimensions
    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
            each tensor has to have the same size (but not nec. same size as vol)
            or a tensor of size [*new_vol_shape, D]
        interp_method: interpolation type 'linear' (default) or 'nearest'
    Returns:
        new interpolated volume of the same size as the entries in loc
    TODO:
        enable optional orig_grid - the original grid points.
        check out tf.contrib.resampler, only seems to work for 2D data
    """
    if isinstance(loc, (list, tuple)):
        loc = tf.stack(loc, -1)

    # since loc can be a list, nb_dims has to be based on vol.
    nb_dims = loc.shape[-1]
    if nb_dims != len(vol.shape[:-1]):
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = K.expand_dims(vol, -1)

    # flatten and float location Tensors
    loc = tf.cast(loc, 'float32')
    print(vol.shape)
    # if isinstance(vol.shape, (tf.Dimension, tf.TensorShape)):
    #     volshape = vol.shape.as_list()
    # else:
    volshape = vol.shape

    # interpolate
    if interp_method == 'linear':
        loc0 = tf.floor(loc)

        # clip values
        max_loc = [d - 1 for d in vol.get_shape().as_list()]
        clipped_loc = [tf.clip_by_value(loc[...,d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [tf.clip_by_value(loc0[...,d], 0, max_loc[d]) for d in range(nb_dims)]

        # get other end of point cube
        loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0] # note reverse ordering since weights are inverse of diff.

        # go through all the cube corners, indexed by a ND binary vector 
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0
        
        for c in cube_pts:
            
            # get nd values
            # note re: indices above volumes via https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's too
            #   expensive. Instead we fill the output with zero for the corresponding value. The CPU
            #   version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
            # indices = tf.stack(subs, axis=-1)
            # vol_val = tf.gather_nd(vol, indices)
            # faster way to gather than gather_nd, because the latter needs tf.stack which is slow :(
            idx = sub2ind(vol.shape[:-1], subs)
            vol_val = tf.gather(tf.reshape(vol, [-1, volshape[-1]]), idx)

            # get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            # tf stacking is slow, we we will use prod_n()
            # wlm = tf.stack(wts_lst, axis=0)
            # wt = tf.reduce_prod(wlm, axis=0)
            wt = prod_n(wts_lst)
            wt = K.expand_dims(wt, -1)
            
            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val
        
    else:
        assert interp_method == 'nearest'
        roundloc = tf.cast(tf.round(loc), 'int32')

        # clip values
        max_loc = [tf.cast(d - 1, 'int32') for d in vol.shape]
        roundloc = [tf.clip_by_value(roundloc[...,d], 0, max_loc[d]) for d in range(nb_dims)]

        # get values
        # tf stacking is slow. replace with gather
        # roundloc = tf.stack(roundloc, axis=-1)
        # interp_vol = tf.gather_nd(vol, roundloc)
        idx = sub2ind(vol.shape[:-1], roundloc)
        interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx) 

    return interp_vol

def point_spatial_transformer(surface_points, trf, single=False):
    """
    surface_points is a N x D or a N x (D+1) Tensor
    trf is a *volshape x D Tensor
    """
    surface_pts_D = surface_points.get_shape().as_list()[-1]
    trf_D = trf.get_shape().as_list()[-1]
    assert surface_pts_D in [trf_D, trf_D + 1]

    if surface_pts_D == trf_D + 1:
        li_surface_pts = K.expand_dims(surface_points[..., -1], -1)
        surface_points = surface_points[..., :-1]
    # just need to interpolate.
    # at each location determined by surface point, figure out the trf...
    # Note: if surface_points are on the grid, gather_nd should work as well
    fn = lambda x:interpn(x[0], x[1])
    diff = tf.map_fn(fn, [trf, surface_points], dtype=tf.float32)
    ret = surface_points + diff
    # ret = diff

    if surface_pts_D == trf_D + 1:
        ret = tf.concat((ret, li_surface_pts), -1)

    return ret

def warp_landmarks(fixed_image_landmarks, flow):
    '''
    warp the fixed_landmarks to moving_image space with the flow. According to https://github.com/voxelmorph/voxelmorph/issues/84
    : fixed_image_landmarks -> (M, 3) array for 3D volume for z,x,y
    : flow -> (N, 3, D, H, W) array for 3D volume for z,x,y
    return the moved landmarks 
    '''
    # moving_image_landmarks = np.loadtxt('./data/DIR-Lab/4DCT/Case2Pack/ExtremePhases/Case2_300_T00_xyz.txt')
    # fixed_image_landmarks = np.loadtxt( './data/DIR-Lab/4DCT/Case2Pack/ExtremePhases/Case2_300_T50_xyz.txt')
    # flow = np.load('./flow.npy')
    flow = flow.transpose(0,2,3,4,1)
    fixed_image_landmarks = fixed_image_landmarks[np.newaxis, ...]
    fixed_image_landmarks = tf.Variable(fixed_image_landmarks, dtype=tf.float32)
    flow = tf.Variable(flow)

    # obtain moved_image_landmarks
    moved_image_landmarks = point_spatial_transformer(fixed_image_landmarks , flow)
    # moved_image_landmarks convert into ndarray
    # os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    # sess = tf.Session()
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    moved_image_landmarks = moved_image_landmarks.eval(session=sess)
    moved_image_landmarks = np.squeeze(moved_image_landmarks )
    return moved_image_landmarks
    # moved is warped to moving space, so the diff = moved - moving 
    # difference = moved_image_landmarks - moving_image_landmarks 
    # sum_difference = 0
    # # Landmarks include 300 points
    # for i in range(0,300,1):
    #     sum_difference += np.linalg.norm(difference[i])
    # # TRE
    # print(sum_difference/300)