import os 
import numpy as np
import pdb

if __name__ == "__main__":
    s = [1, 1, 5, 5, 5]
    b, c, d, h ,w = s
    input1 = np.ones(s)
    input2 = np.ones(s)
    displacement = 3
    kernel = 1
    pad_size = 3
    stride1 = 1
    stride2 = 2

    dis_rad = int(displacement/stride2)
    dis_size = dis_rad *2 + 1
    k_rad = int( (kernel-1) / 2 )

    pad_d = d + 2 * pad_size
    pad_h = h + 2 * pad_size
    pad_w = w + 2 * pad_size
    pad_s = [b, c, pad_d, pad_h, pad_w]

    pad1 = np.zeros(pad_s)
    pad2 = np.zeros(pad_s)
    pad1[:, :, pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size] = input1
    pad2[:, :, pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size] = input2
    
    out_c = (int(displacement/stride2)*2 + 1)*(int(displacement/stride2)*2 + 1)*(int(displacement/stride2)*2 + 1)
    # out_d = d + displacement/
    output = np.zeros([b, out_c, d, h, w])
    
    start = displacement + k_rad
    end_d = pad_d - start
    end_h = pad_h - start
    end_w = pad_w -start

    for i in range(start, end_d):
        for j in range(start, end_h):
            for k in range(start, end_w):
                # pdb.set_trace()
                p_sum = 0
                patch_1 = pad1[0, 0, i-k_rad:i+k_rad, j-k_rad: j+k_rad, k-k_rad:k+k_rad].copy()
                for ti in range(-dis_rad, dis_rad):
                    for tj in range(-dis_rad, dis_rad):
                        for tk in range(-dis_rad, dis_rad):
                            i1 = i + ti*stride2
                            j1 = j + tj*stride2
                            k1 = j + tk*stride2
                            patch_2 = pad2[0, 0, i1-k_rad:i1+k_rad, j1-k_rad: j1+k_rad, k1-k_rad:k1+k_rad].copy()
                            mp = np.mean(patch_1 * patch_2)
                            out_dim = (ti+dis_rad)*displacement*displacement + (tj+dis_rad)*displacement + tk + dis_rad
                            output[0, out_dim, i, j, k] = mp





            