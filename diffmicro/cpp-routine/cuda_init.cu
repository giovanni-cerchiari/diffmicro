/*
Author: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 5/2014

This functions are written for diffmicro.exe application.

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
#include "cuda_init.h"

cudaDeviceProp deviceProp;

int cutGetMaxGflopsDeviceId() 
{ 
	int current_device = 0, sm_per_multiproc = 0; 
	int max_compute_perf = 0, max_perf_device = 0; 
	int device_count = 0, best_SM_arch = 0; 
	int arch_cores_sm[3] = { 1, 8, 32 }; 
 cudaGetDeviceCount( &device_count );

	// Find the best major SM Architecture GPU device 
	while ( current_device < device_count )
	{ 
		cudaGetDeviceProperties( &deviceProp, current_device ); 
		if (deviceProp.major > 0 && deviceProp.major < 9999)
		{ 
			best_SM_arch = max(best_SM_arch, deviceProp.major); 
		} 
		current_device++; 
	} 
	// Find the best CUDA capable GPU device 
		current_device = 0; 
	while( current_device < device_count )
	{ 
		cudaGetDeviceProperties( &deviceProp, current_device ); 
		if (deviceProp.major == 9999 && deviceProp.minor == 9999)
		{ 
			sm_per_multiproc = 1; 
		}
		else if (deviceProp.major <= 2)
		{ 
			sm_per_multiproc = arch_cores_sm[deviceProp.major]; 
		}
		else
		{ // Device has SM major > 2
			sm_per_multiproc = arch_cores_sm[2]; 
		} 

		int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate; 
		if( compute_perf > max_compute_perf )
		{ 
			// If we find GPU of SM major > 2, search only these 
			if ( best_SM_arch > 2 )
			{ 
				// If device==best_SM_arch, choose this, or else pass 
				if (deviceProp.major == best_SM_arch)
				{ 
					max_compute_perf = compute_perf; 
					max_perf_device = current_device; 
				} 
			}
			else
			{ 
				max_compute_perf = compute_perf; 
				max_perf_device = current_device;
		} 
	} 
	++current_device; 
	} 
		cudaGetDeviceProperties(&deviceProp, max_compute_perf); 
		printf("\nDevice %d: \"%s\"\n", max_perf_device, deviceProp.name); 
		printf("Compute Capability : %d.%d\n", deviceProp.major, deviceProp.minor);

	return max_perf_device; 
} 



void cuda_print_device_prop()
{
	std::cout <<"---------------------------------------------"<<std::endl;
	std::cout <<"CUDA DEVICE PROPERTIES"<<std::endl<<std::endl;
	// 	Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer.
	std::cout <<"canMapHostMemory = "<<deviceProp.canMapHostMemory<<std::endl;
	// Clock frequency in kilohertz.
	std::cout <<"clockRate [KHz] = "<<deviceProp.clockRate<<std::endl;
 // Compute mode (See cudaComputeMode).
	std::cout <<"computeMode = "<<deviceProp.computeMode<<std::endl;
 // Device can concurrently copy memory and execute a kernel.
	std::cout <<"deviceOverlap = "<<deviceProp.deviceOverlap<<std::endl;
 // Device is integrated as opposed to discrete.	
	std::cout <<"integrated = "<<deviceProp.integrated<<std::endl;
 // Specified whether there is a run time limit on kernels.	
	std::cout <<"kernelExecTimeoutEnabled = "<<deviceProp.kernelExecTimeoutEnabled<<std::endl;
 // Major compute capability.	
	std::cout <<"major = "<<deviceProp.major<<std::endl;
	// Maximum size of each dimension of a grid.
	std::cout <<"maxGridSize [3] = ("<<*(deviceProp.maxGridSize)<<" ; "<<*(deviceProp.maxGridSize + 1)<<" ; "<<*(deviceProp.maxGridSize + 2)<<")"<<std::endl;
 // Maximum size of each dimension of a block.
	std::cout <<"maxThreadsDim [3] = ("<<*(deviceProp.maxThreadsDim)<<" ; "<<*(deviceProp.maxThreadsDim + 1)<<" ; "<<*(deviceProp.maxThreadsDim + 2)<<")"<<std::endl;
 // Maximum number of threads per block.	
	std::cout <<"maxThreadsPerBlock = "<<deviceProp.maxThreadsPerBlock<<std::endl;
 // Maximum pitch in bytes allowed by memory copies.	
	std::cout <<"memPitch = "<<deviceProp.memPitch<<std::endl;
 // Minor compute capability.	
	std::cout <<"minor = "<<deviceProp.minor<<std::endl;
 // Number of multiprocessors on device.	
	std::cout <<"multiProcessorCount = "<<deviceProp.multiProcessorCount<<std::endl;
 // ASCII string identifying device.	
	std::cout <<"name [256] = "<<deviceProp.name<<std::endl;
 // 32-bit registers available per block	
	std::cout <<"regsPerBlock = "<<deviceProp.regsPerBlock<<std::endl;
 // Shared memory available per block in bytes.	
	std::cout <<"sharedMemPerBlock [bytes] = "<<deviceProp.sharedMemPerBlock<<std::endl;
 // Alignment requirement for textures.	
	std::cout <<"textureAlignment = "<<deviceProp.textureAlignment<<std::endl;
 // Constant memory available on device in bytes.	
	std::cout <<"totalConstMem [bytes] = "<<deviceProp.totalConstMem<<std::endl;
 // Global memory available on device in bytes.	
	std::cout <<"totalGlobalMem [bytes] = "<<deviceProp.totalGlobalMem<<std::endl;
 // Warp size in threads. 	
	std::cout <<"warpSize = "<<deviceProp.warpSize<<std::endl;
	std::cout <<"---------------------------------------------"<<std::endl;
}

bool cuda_init(bool print)
{
	int device_id;
	cudaError_t set_ret;
	bool ret;
//	cudaGetDeviceProperties(&deviceProp, 0);
	device_id = cutGetMaxGflopsDeviceId();
	set_ret = cudaSetDevice(device_id);

	if(print == true) cuda_print_device_prop();

	switch (set_ret)
	{
		case cudaSuccess:
			std::cout << "gpu device correctly initialized" << std::endl;
			ret = true;
			break;

		case cudaErrorInvalidDevice:
			std::cout << "invalid gpu device" << std::endl;
			ret = false;
			break;
		

		case cudaErrorDeviceAlreadyInUse:
			std::cout << "gpu device already in use" << std::endl;
			ret = false;
			break;

		default:
			std::cout << "error initializing gpu device" << std::endl;
			ret = false;
			break;
	}
	

	std::cout <<"---------------------------------------------"<<std::endl;
	//printf("deviceProp.maxThreadsPerBlock = %i \r\n\r\n", deviceProp.maxThreadsPerBlock);

	return ret;
}


void cuda_end()
{
	
}


cudaError_t my_cudafree(void *ptr)
{
	cudaError_t ret;
	if(NULL != ptr)
	{
		ret = cudaFree(ptr);
		ptr = NULL;
	}
	else
	{
		ret = cudaSuccess;
	}
	return ret;
}

void calc_cuda_exec(INDEX n_total_threads, INDEX ThreadsPerBlock, cuda_exec *exec)
{
	if (0 == ThreadsPerBlock) return;
	INDEX rem;
	lldiv_t d;

	exec->nbk.y = 1;
	exec->nbk.z = 1;
	exec->nth.y = 1;
	exec->nth.z = 1;

	d = div((long long)(n_total_threads), ThreadsPerBlock);

	// if there is remanence or not
	if (0 == d.rem)
		exec->nbk.x = 0;
	else
		exec->nbk.x = 1;

	exec->nbk.x += d.quot;

	// maximize the number of threads
	exec->nth.x = ThreadsPerBlock;

	// in case the threads are really few, then it is only remanence of the division
	if (n_total_threads < ThreadsPerBlock)
		exec->nth.x = n_total_threads;

}
