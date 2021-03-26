/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 5/2014
*/
/*
This file is part of diffmicro.

    Diffmicro is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Diffmicro is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Diffmicro.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef _CUDA_INIT_H_
#define _CUDA_INIT_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda_texture_types.h>

#include <iostream>

#include "global_define.h"

/*! This structure contains the relevant data of the GPU card*/
extern cudaDeviceProp deviceProp;

/*! This structure is used to store the number of blocks and threads to execute kernels on GPU.*/
struct cuda_exec
{
  //! number of blocks
	dim3 nbk;
  //! number of threads
	dim3 nth;
};

/*!This function should be called at the beginning of the program to prepare CUDA execution.*/
bool cuda_init(bool print);
/*!This function should be called at the end of the program to free the memory used by CUDA.*/
void cuda_end();

cudaError_t my_cudafree(void *ptr);

/*!This function can be used to calculate cuda_exec given the total ammount of threads that the user would like to execute and
the maximuma ThreadsPerBlock that the user would like to execute in the kernels. The final number of threads to be executed will
be exec.nbk*exec.nth >= n_total_threads. */
void calc_cuda_exec(INDEX n_total_threads, INDEX ThreadsPerBlock, cuda_exec *exec);

/*!This kernel is left here for example. This kernel multiplies the input array by a constant.
template<typename FLOAT>
__global__ void gain_ary_device(FLOAT gain, INDEX dim, FLOAT *in, FLOAT *out)
{
	INDEX i = (blockDim.x * blockIdx.x + threadIdx.x);
	if (i < dim)
	{
		out[i] = gain*in[i];
	}
}*/

#endif
