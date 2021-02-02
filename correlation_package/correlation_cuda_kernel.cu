#include <stdio.h>

#include "correlation_cuda_kernel.cuh"

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

using at::Half;

template <typename scalar_t>
__global__ void channels_first(const scalar_t* __restrict__ input, scalar_t* rinput, int channels, int depth, int height, int width, int pad_size)
{

	// n (batch size), c (num of channels), z (depth), y (height), x (width)
	int z = blockIdx.x;
    int y = blockIdx.y;
	int x = blockIdx.z;

	int n = threadIdx.x ;
	int ch_off = threadIdx.y;
	scalar_t value;

    int dimczyx = channels * depth * height * width;
    int dimzyx = depth * height * width;
    int dimyx = height * width;

    int p_dimx = (width + 2 * pad_size);
    int p_dimy = (height + 2 * pad_size);
    int p_dimz = (depth + 2 * pad_size);
    int p_dimzyxc = channels * p_dimz * p_dimy * p_dimx;
    int p_dimyxc = channels * p_dimy * p_dimx;
    int p_dimxc = p_dimx * channels;

	for (int c = ch_off; c < channels; c += THREADS_PER_BLOCK) {
      value = input[n * dimczyx + c * dimzyx + z * dimyx + y * width + x];
      rinput[n * p_dimzyxc + (z + pad_size) * p_dimyxc + (y + pad_size) * p_dimxc + (x + pad_size) * channels + c] = value;
	}
}

template <typename scalar_t>
__global__ void correlation_forward(scalar_t*  output, int nOutputChannels, int outputDepth, int outputHeight, int outputWidth,
	const scalar_t* __restrict__ rInput1, int nInputChannels, int inputDepth, int inputHeight, int inputWidth,
	const scalar_t* __restrict__ rInput2,
	int pad_size,
	int kernel_size,
	int max_displacement,
	int stride1,
	int stride2)
{
	// n (batch size), c (num of channels), z(depth), y (height), x (width)

    int pInputDepth = inputDepth + 2 * pad_size;
	int pInputWidth = inputWidth + 2 * pad_size;
	int pInputHeight = inputHeight + 2 * pad_size;

	int kernel_rad = (kernel_size - 1) / 2;
	int displacement_rad = max_displacement / stride2;
	int displacement_size = 2 * displacement_rad + 1;

	int n = threadIdx.x;

	int z1 = blockIdx.x * stride1 + max_displacement + kernel_rad;
    int y1 = blockIdx.y * stride1 + max_displacement + kernel_rad;
    int x1 = blockIdx.z * stride1 + max_displacement + kernel_rad;
    int c = threadIdx.y;

    int pdimzyxc = pInputDepth * pInputHeight * pInputWidth * nInputChannels;
    int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
    int pdimxc = pInputWidth * nInputChannels;
    int pdimc = nInputChannels;

    int tdimczyx = nOutputChannels * outputDepth * outputHeight * outputWidth;
    int tdimzyx = outputDepth * outputHeight * outputWidth;
    int tdimyx = outputHeight * outputWidth;
    int tdimx = outputWidth;

	scalar_t nelems = kernel_size * kernel_size * kernel_size * pdimc;

    __shared__ float prod_sum[THREADS_PER_BLOCK];

    // no significant speed-up in using chip memory for input1 sub-data, 
    // not enough chip memory size to accomodate memory per block for input2 sub-data
    // instead i've used device memory for both 

    // element-wise product along channel axis
    for (int tk = -displacement_rad; tk <= displacement_rad; ++tk ) {
        for (int tj = -displacement_rad; tj <= displacement_rad; ++tj ) {
            for (int ti = -displacement_rad; ti <= displacement_rad; ++ti ) {
                prod_sum[c] = 0;
                int x2 = x1 + ti*stride2;
                int y2 = y1 + tj*stride2;
                int z2 = z1 + tk*stride2;

                for (int k = -kernel_rad; k <= kernel_rad; ++k) {
                    for (int j = -kernel_rad; j <= kernel_rad; ++j) {
                        for (int i = -kernel_rad; i <= kernel_rad; ++i) {
                            for (int ch = c; ch < pdimc; ch += THREADS_PER_BLOCK) {
                                int indx1 = n * pdimzyxc + (z1+k)*pdimyxc + (y1+j) * pdimxc + (x1 + i) * pdimc + ch;
                                int indx2 = n * pdimzyxc + (z2+k)*pdimyxc + (y2+j) * pdimxc + (x2 + i) * pdimc + ch;
                                prod_sum[c] += rInput1[indx1] * rInput2[indx2];
                            }
                        }
                    }
                }

                // accumulate 
                __syncthreads();
                if (c == 0) {
                    float reduce_sum = 0;
                    for (int index = 0; index < THREADS_PER_BLOCK; ++index) {
                        reduce_sum += prod_sum[index];
                    }
                    int tc =(tk + displacement_rad) * displacement_size * displacement_size + (tj + displacement_rad) * displacement_size + (ti + displacement_rad);
                    const int tindx = n * tdimczyx + tc * tdimzyx + blockIdx.x * tdimyx + blockIdx.y * tdimx + blockIdx.z;
                    output[tindx] = reduce_sum / nelems;
                }
      }
    }
    }
}


template <typename scalar_t>
__global__ void correlation_backward_input1(int item, scalar_t* gradInput1, int nInputChannels, int inputDepth, int inputHeight, int inputWidth,
	const scalar_t* __restrict__ gradOutput, int nOutputChannels, int outputDepth, int outputHeight, int outputWidth,
	const scalar_t* __restrict__ rInput2,
	int pad_size,
	int kernel_size,
	int max_displacement,
	int stride1,
	int stride2)
{
	// n (batch size), c (num of channels), z(depth), y (height), x (width)

	int n = item;
    int z = blockIdx.x * stride1 + pad_size;
    int y = blockIdx.y * stride1 + pad_size;
    int x = blockIdx.z * stride1 + pad_size;
    int c = threadIdx.x;
    int tch_off = threadIdx.y;


	int kernel_rad = (kernel_size - 1) / 2;
	int displacement_rad = max_displacement / stride2;
	int displacement_size = 2 * displacement_rad + 1;

    int xmin = (x - kernel_rad - max_displacement) / stride1;
    int ymin = (y - kernel_rad - max_displacement) / stride1;
    int zmin = (z - kernel_rad - max_displacement) / stride1;

    int xmax = (x + kernel_rad - max_displacement) / stride1;
    int ymax = (y + kernel_rad - max_displacement) / stride1;
    int zmax = (z + kernel_rad - max_displacement) / stride1;

    if (xmax < 0 || ymax < 0 || zmax < 0 || xmin >= outputWidth || ymin >= outputHeight || zmax >= outputDepth) {
        // assumes gradInput1 is pre-allocated and zero filled
      return;
    }

    if (xmin > xmax || ymin > ymax || zmin > zmax) {
        // assumes gradInput1 is pre-allocated and zero filled
        return;
    }


    xmin = max(0,xmin);
    xmax = min(outputWidth-1,xmax);

    ymin = max(0,ymin);
    ymax = min(outputHeight-1,ymax);

    zmin = max(0,zmin);
    zmax = min(outputDepth-1,zmax);

    int pInputWidth = inputWidth + 2 * pad_size;
    int pInputHeight = inputHeight + 2 * pad_size;
    int pInputDepth = inputDepth + 2 * pad_size;

    int pdimzyxc = pInputDepth * pInputHeight * pInputWidth * nInputChannels;
    int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
    int pdimxc = pInputWidth * nInputChannels;
    int pdimc = nInputChannels;

    int tdimczyx = nOutputChannels * outputDepth * outputHeight * outputWidth;
    int tdimzyx = outputDepth * outputHeight * outputWidth;
    int tdimyx = outputHeight * outputWidth;
    int tdimx = outputWidth;

    int odimczyx = nInputChannels * inputDepth * inputHeight* inputWidth;
    int odimzyx = inputDepth * inputHeight* inputWidth;
    int odimyx = inputHeight * inputWidth;
    int odimx = inputWidth;

	scalar_t nelems = kernel_size * kernel_size * kernel_size * nInputChannels;

	__shared__ scalar_t prod_sum[THREADS_PER_BLOCK];
	prod_sum[tch_off] = 0;

    for (int tc = tch_off; tc < nOutputChannels; tc += THREADS_PER_BLOCK) {

      int i2 = (tc % displacement_size - displacement_rad) * stride2;
      int j2 = ((tc / displacement_size) % displacement_size - displacement_rad) * stride2;
      int k2 = ((tc / (displacement_size*displacement_size)) % displacement_size  - displacement_rad) * stride2;

      int indx2 = n * pdimzyxc + (z + k2) * pdimyxc + (y + j2)* pdimxc + (x + i2) * pdimc + c;
      
      float val2 = rInput2[indx2];

      for (int k = zmin; k <= zmax; ++k) {
      for (int j = ymin; j <= ymax; ++j) {
        for (int i = xmin; i <= xmax; ++i) {
          int tindx = n * tdimczyx + tc * tdimzyx + k * tdimyx + j * tdimx + i;
          prod_sum[tch_off] += gradOutput[tindx] * val2;
        }
      }
      }
    }
	__syncthreads();

	if (tch_off == 0) {
		scalar_t reduce_sum = 0;
		for(int idx = 0; idx < THREADS_PER_BLOCK; idx++) {
			reduce_sum += prod_sum[idx];
		}
		const int indx1 = n * odimczyx + c * odimzyx + (z - pad_size) * odimyx + (y - pad_size) * odimx + (x - pad_size);
		gradInput1[indx1] = reduce_sum / nelems;
	}

}

template <typename scalar_t>
__global__ void correlation_backward_input2(int item, scalar_t*  gradInput2, int nInputChannels, int inputDepth, int inputHeight, int inputWidth,
	const scalar_t* __restrict__ gradOutput, int nOutputChannels, int outputDepth, int outputHeight, int outputWidth,
	const scalar_t* __restrict__ rInput1,
	int pad_size,
	int kernel_size,
	int max_displacement,
	int stride1,
	int stride2)
{
	// n (batch size), c (num of channels), z(depth), y (height), x (width)

    int n = item;
    int z = blockIdx.x * stride1 + pad_size;
    int y = blockIdx.y * stride1 + pad_size;
    int x = blockIdx.z * stride1 + pad_size;
    int c = threadIdx.x;

    int tch_off = threadIdx.y;

    int kernel_rad = (kernel_size - 1) / 2;
    int displacement_rad = max_displacement / stride2;
    int displacement_size = 2 * displacement_rad + 1;

    int pInputDepth = inputDepth + 2 * pad_size;
    int pInputWidth = inputWidth + 2 * pad_size;
    int pInputHeight = inputHeight + 2 * pad_size;

    int pdimzyxc = pInputDepth * pInputHeight * pInputWidth * nInputChannels;
    int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
    int pdimxc = pInputWidth * nInputChannels;
    int pdimc = nInputChannels;

    int tdimczyx = nOutputChannels * outputDepth * outputHeight * outputWidth;
    int tdimzyx = outputDepth * outputHeight * outputWidth;
    int tdimyx = outputHeight * outputWidth;
    int tdimx = outputWidth;

    int odimczyx = nInputChannels * inputDepth * inputHeight* inputWidth;
    int odimzyx = inputDepth * inputHeight* inputWidth;
    int odimyx = inputHeight * inputWidth;
    int odimx = inputWidth;

	scalar_t nelems = kernel_size * kernel_size * kernel_size * nInputChannels;

	__shared__ scalar_t prod_sum[THREADS_PER_BLOCK];
	prod_sum[tch_off] = 0;

    for (int tc = tch_off; tc < nOutputChannels; tc += THREADS_PER_BLOCK) {
		int i2 = (tc % displacement_size - displacement_rad) * stride2;
		int j2 = ((tc / displacement_size) % displacement_size - displacement_rad) * stride2;
		int k2 = ((tc / (displacement_size*displacement_size)) % displacement_size  - displacement_rad) * stride2;

		int xmin = (x - kernel_rad - max_displacement - i2) / stride1;
		int ymin = (y - kernel_rad - max_displacement - j2) / stride1;
		int zmin = (z - kernel_rad - max_displacement - k2) / stride1;

		int xmax = (x + kernel_rad - max_displacement - i2) / stride1;
		int ymax = (y + kernel_rad - max_displacement - j2) / stride1;
		int zmax = (z + kernel_rad - max_displacement - k2) / stride1;

		if (xmax < 0 || ymax < 0 || zmax < 0 || xmin >= outputWidth || ymin >= outputHeight || zmin >= outputDepth) {
			// assumes gradInput2 is pre-allocated and zero filled
			continue;
		}

		if (xmin > xmax || ymin > ymax || zmin > zmax) {
			// assumes gradInput2 is pre-allocated and zero filled
			continue;
		}

		xmin = max(0,xmin);
		xmax = min(outputWidth-1,xmax);

		ymin = max(0,ymin);
		ymax = min(outputHeight-1,ymax);

		zmin = max(0,zmin);
		zmax = min(outputDepth-1,zmax);
		
		int indx1 = n * pdimzyxc + (z - k2)*pdimyxc + (y - j2)* pdimxc + (x - i2) * pdimc + c;
		
		scalar_t val1 = rInput1[indx1];

	for (int k = zmin; k <= zmax; ++k) {
	for (int j = ymin; j <= ymax; ++j) {
		for (int i = xmin; i <= xmax; ++i) {
			int tindx = n * tdimczyx + tc * tdimzyx + k * tdimyx + j * tdimx + i;
			prod_sum[tch_off] += gradOutput[tindx] * val1;
			}
		}
	}

	__syncthreads();

    if(tch_off == 0) {
		float reduce_sum = 0;
		for(int idx = 0; idx < THREADS_PER_BLOCK; idx++) {
			reduce_sum += prod_sum[idx];
		}
		const int indx2 = n * odimczyx + c * odimzyx + (z - pad_size) * odimyx + (y - pad_size) * odimx + (x - pad_size);
		gradInput2[indx2] = reduce_sum / nelems;
	}
	}
}

int correlation_forward_cuda_kernel(at::Tensor& output,
	int ob,
	int oc,
	int od,
	int oh,
	int ow,
	int osb,
	int osc,
	int osh,
	int osw,

	at::Tensor& input1,
	int ic,
	int id,
	int ih,
	int iw,
	int isb,
	int isc,
	int ish,
	int isw,

	at::Tensor& input2,
	int gc,
	int gsb,
	int gsc,
	int gsh,
	int gsw,

	at::Tensor& rInput1,
	at::Tensor& rInput2,
	int pad_size,
	int kernel_size,
	int max_displacement,
	int stride1,
	int stride2,
	int corr_type_multiply,
	cudaStream_t stream)
{

	int batchSize = ob;

	int nInputChannels = ic;
	int inputWidth = iw;
	int inputHeight = ih;
	int inputDepth = id;

	int nOutputChannels = oc;
	int outputWidth = ow;
	int outputHeight = oh;
	int outputDepth = od;

	dim3 blocks_grid(inputDepth, inputHeight, inputWidth);
	dim3 threads_block(batchSize, THREADS_PER_BLOCK);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.type(), "channels_first_fwd_1", ([&] {

		channels_first<scalar_t> << <blocks_grid, threads_block, 0, stream >> >(
			input1.data<scalar_t>(), rInput1.data<scalar_t>(), nInputChannels, inputDepth, inputHeight, inputWidth, pad_size);

	}));

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.type(), "channels_first_fwd_2", ([&] {

		channels_first<scalar_t> << <blocks_grid, threads_block, 0, stream >> > (
			input2.data<scalar_t>(), rInput2.data<scalar_t>(), nInputChannels, inputDepth, inputHeight, inputWidth, pad_size);

	}));

	dim3 threadsPerBlock(batchSize, THREADS_PER_BLOCK);
	dim3 totalBlocksCorr(outputDepth, outputHeight, outputWidth);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.type(), "correlation_forward", ([&] {

		correlation_forward<scalar_t> << <totalBlocksCorr, threadsPerBlock, 0, stream >> >
			(output.data<scalar_t>(), nOutputChannels, outputDepth, outputHeight, outputWidth,
			rInput1.data<scalar_t>(), nInputChannels, inputDepth, inputHeight, inputWidth,
			rInput2.data<scalar_t>(),
			pad_size,
			kernel_size,
			max_displacement,
			stride1,
			stride2);

	}));

	cudaError_t err = cudaGetLastError();


	// check for errors
	if (err != cudaSuccess) {
		printf("error in correlation_forward_cuda_kernel: %s\n", cudaGetErrorString(err));
		return 0;
	}

	return 1;
}


int correlation_backward_cuda_kernel(
	at::Tensor& gradOutput,
	int gob,
	int goc,
	int god,
	int goh,
	int gow,
	int gosb,
	int gosc,
	int gosh,
	int gosw,

	at::Tensor& input1,
	int ic,
	int id,
	int ih,
	int iw,
	int isb,
	int isc,
	int ish,
	int isw,

	at::Tensor& input2,
	int gsb,
	int gsc,
	int gsh,
	int gsw,

	at::Tensor& gradInput1,
	int gisb,
	int gisc,
	int gish,
	int gisw,

	at::Tensor& gradInput2,
	int ggc,
	int ggsb,
	int ggsc,
	int ggsh,
	int ggsw,

	at::Tensor& rInput1,
	at::Tensor& rInput2,
	int pad_size,
	int kernel_size,
	int max_displacement,
	int stride1,
	int stride2,
	int corr_type_multiply,
	cudaStream_t stream)
{

	int batchSize = gob;
	int num = batchSize;

	int nInputChannels = ic;
	int inputDepth = id;
	int inputWidth = iw;
	int inputHeight = ih;

	int nOutputChannels = goc;
    int outputDepth = god;
	int outputWidth = gow;
	int outputHeight = goh;

	dim3 blocks_grid(inputDepth, inputHeight, inputWidth);
	dim3 threads_block(batchSize, THREADS_PER_BLOCK);


	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.type(), "lltm_forward_cuda", ([&] {

		channels_first<scalar_t> << <blocks_grid, threads_block, 0, stream >> >(
			input1.data<scalar_t>(),
			rInput1.data<scalar_t>(),
			nInputChannels,
			inputDepth,
			inputHeight,
			inputWidth,
			pad_size
			);
	}));

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.type(), "lltm_forward_cuda", ([&] {

		channels_first<scalar_t> << <blocks_grid, threads_block, 0, stream >> >(
			input2.data<scalar_t>(),
			rInput2.data<scalar_t>(),
			nInputChannels,
			inputDepth,
			inputHeight,
			inputWidth,
			pad_size
			);
	}));

	dim3 threadsPerBlock(nInputChannels, THREADS_PER_BLOCK);
	dim3 totalBlocksCorr(inputDepth, inputHeight, inputWidth);

	for (int n = 0; n < num; ++n) {

		AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.type(), "lltm_forward_cuda", ([&] {


			correlation_backward_input1<scalar_t> << <totalBlocksCorr, threadsPerBlock, 0, stream >> > (
				n, gradInput1.data<scalar_t>(), nInputChannels, inputDepth, inputHeight, inputWidth,
				gradOutput.data<scalar_t>(), nOutputChannels, outputDepth, outputHeight, outputWidth,
				rInput2.data<scalar_t>(),
				pad_size,
				kernel_size,
				max_displacement,
				stride1,
				stride2);
		}));
	}

	for (int n = 0; n < batchSize; n++) {

		AT_DISPATCH_FLOATING_TYPES_AND_HALF(rInput1.type(), "lltm_forward_cuda", ([&] {

			correlation_backward_input2<scalar_t> << <totalBlocksCorr, threadsPerBlock, 0, stream >> >(
				n, gradInput2.data<scalar_t>(), nInputChannels, inputDepth, inputHeight, inputWidth,
				gradOutput.data<scalar_t>(), nOutputChannels, outputDepth, outputHeight, outputWidth,
				rInput1.data<scalar_t>(),
				pad_size,
				kernel_size,
				max_displacement,
				stride1,
				stride2);

		}));
	}

	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("error in correlation_backward_cuda_kernel: %s\n", cudaGetErrorString(err));
		return 0;
	}

	return 1;
}
