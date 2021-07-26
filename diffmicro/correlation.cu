/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 8/2011
*/

/*
Copyright: Mojtaba Norouzisadeh, Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
e-mail: norouzi.mojtaba.sade@gmail.com

update: 05/2020 - 09/2020
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


/*!
This functions are written for diffmicro.exe application.
*/

#include "correlation.h"
#include "figure_opencv.h"
#include "device_launch_parameters.h"

window_display_opencv *fft_window(NULL);
window_display_opencv *image_window(NULL);
double *image_l(NULL);
double *module_fft(NULL);
double *phase_fft(NULL);
unsigned __int16 *tmp_display_i16(NULL);
CUFFT_COMPLEX *tmp_display_cpx(NULL);

sizes s_load_image;
sizes s_radial_lut;
sizes s_fft;
sizes s_fft_images;
sizes s_power_spectra;
sizes s_time_series;
sizes s_fft_time;


unsigned short *dev_im_gpu(NULL);
unsigned int *dev_radial_lut_gpu(NULL);
CUFFT_COMPLEX *dev_fft_gpu(NULL);
CUFFT_COMPLEX* dev_fft_time_gpu(NULL);
CUFFT_REAL* dev_corr_gpu(NULL);
CUFFT_COMPLEX *dev_images_gpu(NULL);
CUFFT_COMPLEX *dev_image_sot_gpu(NULL);
STORE_REAL *dev_power_spectra_gpu(NULL);
//Mohammed 
//STORE_REAL* test_pw(NULL);





//! this is the inverse norm of the FFT to have the operation normalized
CUFFT_REAL one_over_fft_norm;
//! CUDA variable necessary to calculate a FFT
cufftHandle plan;
cufftHandle tplan;


/*!
from image in unsigned short format to a CUFFT_COMPLEX memory area where the FFT will be calculated
*/
__global__ void short_to_real_with_gain(INDEX dim, unsigned short in[], CUFFT_REAL gain, CUFFT_COMPLEX out[])
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dim)
	{
		out[i].x = gain * (CUFFT_REAL)(in[i]);
		out[i].y = 0.;
	}
}

/*!
Gain of a complex array by a real coefficient
*/
__global__ void gain_complex(CUFFT_REAL gain, INDEX dim, CUFFT_COMPLEX in[], CUFFT_COMPLEX out[])
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dim)
	{
		out[i].x = gain * in[i].x;
		out[i].y = gain * in[i].y;
	}
}

__global__ void gaincomplex_gpu_2d(INDEX dimfft, INDEX dimt, INDEX dim, CUFFT_COMPLEX in[], FFTW_REAL gain, CUFFT_COMPLEX out[])
{
	INDEX j = blockDim.y * blockIdx.y + threadIdx.y;

	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;

	if ((i < dim) && (j < dimt))
	{
		out[i + j * dimfft].x = gain * in[i + j * dim].x;
		out[i + j * dimfft].y = gain * in[i + j * dim].y;
	}
}
/*!This kernel copy with real gain and lut*/
__global__ void gain_complex_lut(CUFFT_REAL gain, INDEX dim, unsigned int *lut, CUFFT_COMPLEX in[], CUFFT_COMPLEX out[])
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dim)
	{
		out[i].x = gain * in[lut[i]].x;
		out[i].y = gain * in[lut[i]].y;
	}
}


/*!
This kernel executes the difference between the upper half of two Fourier transforms, calulate the power spectrum
and stores the result refreshing the values o a power spectrum out[]. coef1 and coef2 are the refreshing coefficient
used to make a linear combination.
*/
__global__ void diff_power_spectrum_to_avg_gpu
  (INDEX dim, CUFFT_COMPLEX min[], CUFFT_COMPLEX sot[], CUFFT_REAL coef1, CUFFT_REAL coef2, STORE_REAL out[])
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;
	CUFFT_REAL difx, dify;

	if (i < dim)
	{
		difx = min[i].x - sot[i].x;
		dify = min[i].y - sot[i].y;
		out[i] = coef1 * out[i] + coef2 * (difx * difx + dify * dify);
	}
}

__global__ void averagesabs2_array_gpu(INDEX dim, INDEX dim_t, CUFFT_COMPLEX* _in, CUFFT_REAL* out)
{
	INDEX j = blockDim.x * blockIdx.x + threadIdx.x;
	// Does the time series exists?
	if (j < dim_t)
	{
		CUFFT_COMPLEX* in;
		// selection of the time series
		in = &(_in[j * dim]);

		FFTW_REAL avg = 0.0;
		FFTW_REAL coef1, coef2, abs2_fromstart, abs2_fromend;
		INDEX i, ii;
		for (i = 0; i < dim; ++i)
		{
			// next absolute value from the beginning of the array
			abs2_fromstart = in[i ].x * in[i ].x + in[i ].y * in[i ].y;

			// next absolute value from the end of the array
			ii = dim - 1 - i;
			abs2_fromend = in[ii ].x * in[ii ].x + in[ii ].y * in[ii ].y;

			// in-place average
			coef2 = (FFTW_REAL)(1.0) / (FFTW_REAL)(i + 1);
			coef1 = (FFTW_REAL)(i)*coef2;
			avg = coef1 * avg + coef2 * (abs2_fromstart + abs2_fromend);

			// save the result in the output array.
			// This operation must be done inside the for loop.
			// ATTENTION! note the index
			out[ii + j * dim] = avg;
		}
	}
}

__global__ void gaincomplex_gpu(INDEX dim, CUFFT_COMPLEX in[], FFTW_REAL gain, CUFFT_COMPLEX out[])
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;

	if ( i < dim)
	{
		out[i].x = gain * in[i].x;
		out[i].y = gain * in[i].y;
	}
}
__global__ void gaincomplex_gpu_test(INDEX dim, STORE_REAL in[], FFTW_REAL gain, STORE_REAL out[])
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dim)
	{
		out[i] = gain * in[i];
		
	}
}

__global__ void complexabs2_gpu(INDEX dim, CUFFT_COMPLEX* in, CUFFT_COMPLEX* out)
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;

	if  ( i < dim)
	{
		out[i].x = in[i].x * in[i].x + in[i].y * in[i].y;
		out[i].y = 0.0;
	}
}

__global__ void updatewithdivrebyramp_gpu(INDEX dim, INDEX ramp_start, CUFFT_COMPLEX* in, FFTW_REAL* update)
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dim)
		update[i] -= (2. / (FFTW_REAL)(ramp_start - i)) * in[i].x;
}

__global__ void copyfrom_gpu(INDEX dim, CUFFT_COMPLEX* tseries, FFTW_REAL* corr_memory)
{

	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dim)
	{
		
		tseries[i ].x = corr_memory[i ];
		tseries[i ].y = 0.0;
	}
}


__global__ void cpx_col2row_gain_gpu(INDEX dimcopy, INDEX dimx_in, INDEX i_col_in, CUFFT_COMPLEX in[], CUFFT_REAL gain, INDEX dimx_out, INDEX i_row_out, CUFFT_COMPLEX out[])
{

	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;
	INDEX i_in, i_out;

	if (i < dimcopy)
	{
		i_in = i * dimx_in + i_col_in;
		i_out = i_row_out * dimx_out + i;

		out[i_out].x = gain * in[i_in].x;
		out[i_out].y= gain * in[i_in].y;
	}
}

__global__ void cpx_row2col_gain_lut_gpu(CUFFT_REAL gain0, unsigned int* lut, INDEX dim, INDEX dimx_in, INDEX i_row_in, CUFFT_COMPLEX in[],
	FFTW_REAL gain, INDEX dimx_out, INDEX i_col_out, CUFFT_COMPLEX out[])
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;
	INDEX i_in, i_out;
	if (i < dim)
	{
		i_in = i_row_in * dimx_in + i;
		i_out = i * dimx_out + i_col_out;

		out[i_out].x = gain0 * gain * in[lut[i_in]].x;
		out[i_out].y = gain0 * gain * in[lut[i_in]].y;
	}
}

__global__ void cpx_row2col_gain_gpu(INDEX dim, INDEX dimx_in, INDEX i_row_in, CUFFT_COMPLEX in[], FFTW_REAL gain, INDEX dimx_out, INDEX i_col_out, CUFFT_COMPLEX out[])
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;
	INDEX i_in, i_out;
	if (i < dim)
	{
		i_in = i_row_in * dimx_in + i;
		i_out = i * dimx_out + i_col_out;

		out[i_out].x = gain * in[i_in].x;
		out[i_out].y = gain * in[i_in].y;
	}
}


__global__ void complextorealwithgain_gpu(INDEX dim, CUFFT_COMPLEX vets[], CUFFT_REAL gain, CUFFT_REAL vetc[])
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dim)
	{
		vetc[i] = gain * vets[i].x;
	}
}

__global__ void updatewithdivrebyramp_gpu_2d(INDEX ii,unsigned int* lut, INDEX dimxy2,INDEX id_group,INDEX dimfft, INDEX dimt, INDEX dim, INDEX ramp_start,
	CUFFT_COMPLEX* in, FFTW_REAL* update, CUFFT_COMPLEX* tseries)//,STORE_REAL* test),INDEX numerosity, INDEX nbthreads
{
	INDEX j = blockDim.y * blockIdx.y + threadIdx.y;

	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;

	if ((i < dim) && (i !=0) && (j < dimt)) {
		tseries[i + j * dim].x = update[i + j * dim] - (2. / (FFTW_REAL)(ramp_start - i)) * in[i + j * dimfft].x;
		//test[lut[j + id_group * nbthreads+ii* numerosity] + i * dimxy2] = (STORE_REAL)(update[i + j * dim] - (2. / (FFTW_REAL)(ramp_start - i)) * in[i + j * dimfft].x);
		//tseries[i + j * dim].x = update[i + j * dim];
		//tseries[i + j * dim].y = 0.0;
	}
}

void time_series_analysis_gpu_2D(INDEX ii) {

	cuda_exec mycuda_dim_t, mycuda_dim, mycuda_dim_dim_t;

	lldiv_t group;
	INDEX i, n_group, group_rem;

	time_time_correlation.start();

	calc_cuda_exec(s_time_series.dim, deviceProp.maxThreadsPerBlock, &mycuda_dim);
	calc_cuda_exec(useri.nthread_gpu, deviceProp.maxThreadsPerBlock, &mycuda_dim_t);
	calc_cuda_exec(s_time_series.dim * useri.nthread_gpu, deviceProp.maxThreadsPerBlock, &mycuda_dim_dim_t);


	group = std::div((long long)(s_time_series.numerosity), (long long)(useri.nthread_gpu));
	n_group = (INDEX)(group.quot);
	group_rem = (INDEX)(group.rem);

	//-----------------------------------------
	// NO GROUP SPLIT, IT IS ASSUMED THAT MEMORY WILL BE ENOUGH... TO BE CHANGED

	for (i = 0; i < n_group; ++i)
	{
		//std::cout << i << std::endl;
		timeseriesanalysis_gpu_2D(ii,i,s_time_series.dim, useri.nthread_gpu, &dev_images_gpu[i * useri.nthread_gpu * s_time_series.dim],
			s_fft_time.dim, dev_fft_time_gpu, &tplan, dev_corr_gpu,
			mycuda_dim_t, mycuda_dim, mycuda_dim_dim_t);

	}
	if (0 != group_rem)
	{
		calc_cuda_exec(group_rem, deviceProp.maxThreadsPerBlock, &mycuda_dim_t);
		calc_cuda_exec(s_time_series.dim * group_rem, deviceProp.maxThreadsPerBlock, &mycuda_dim_dim_t);
		timeseriesanalysis_gpu_2D(ii,i,s_time_series.dim, group_rem, &dev_images_gpu[i * useri.nthread_gpu * s_time_series.dim],
			s_fft_time.dim, dev_fft_time_gpu, &tplan, dev_corr_gpu,
			mycuda_dim_t, mycuda_dim, mycuda_dim_dim_t);

	}

	time_time_correlation.stop();

	time_from_device_to_host.start();
	cudaMemcpy(&dev_images_cpu[ii * (s_time_series.dim * s_time_series.numerosity)], dev_images_gpu, s_time_series.memory_tot, cudaMemcpyDeviceToHost);
	time_from_device_to_host.stop();

	/*timeseriesanalysis_gpu(s_time_series.dim, s_time_series.numerosity, dev_images_gpu, s_fft_time.dim, dev_fft_time_gpu, &tplan, dev_corr_gpu,
		mycuda_dim_t, mycuda_dim, mycuda_dim_dim_t);*/

		//timeseriesanalysis_gpu(dim, dim_tseries, yinputg2, dimp, yg2, &plan, out_f2_g, mycuda_dim_t, mycuda_dim, mycuda_dim_dim_t);

}

void CUDA_free() {

	if (NULL != dev_im_gpu)
	{
		cudaFree(dev_im_gpu);
		dev_im_gpu = NULL;
	}
	if (NULL != dev_fft_gpu)
	{
		cudaFree(dev_fft_gpu);
		dev_fft_gpu = NULL;
	}
	if (NULL != dev_images_gpu)
	{
		cudaFree(dev_images_gpu);
		dev_images_gpu = NULL;
	}
	if (NULL != dev_power_spectra_gpu)
	{
		cudaFree(dev_power_spectra_gpu);
		dev_power_spectra_gpu = NULL;
	}
	/*if (NULL != dev_radial_lut_gpu)
	{
		cudaFree(dev_radial_lut_gpu);
		dev_radial_lut_gpu = NULL;
	}*/

	if (NULL != dev_fft_time_gpu)
	{
		cudaFree(dev_fft_time_gpu);
		dev_fft_time_gpu = NULL;
	}

	if (NULL != dev_corr_gpu)
	{
		cudaFree(dev_corr_gpu);
		dev_corr_gpu = NULL;
	}

	if (NULL != dev_image_sot_gpu)
	{
		cudaFree(dev_image_sot_gpu);
		dev_image_sot_gpu = NULL;
	}
}

__global__ void dev_images_TO_ram_power_spectra(INDEX dimxdimy2,INDEX npw, INDEX dimpw,
	CUFFT_COMPLEX in[], unsigned int* lut, STORE_REAL out[])
{
	INDEX i = blockDim.y * blockIdx.y + threadIdx.y;

	INDEX j = blockDim.x * blockIdx.x + threadIdx.x;

	if ((i < npw) && (j < dimpw))
	{
		//out[lut[j]] = (STORE_REAL)in[i + j * 1].x;
		out[lut[j]+ i* dimxdimy2] = (STORE_REAL)in[i + j * npw].x;

	}

}
void test_pw_gpu(STORE_REAL* ram_power_spectra_) {

	/*int cudaStatus = cudaMemcpy(ram_power_spectra_, test_pw, s_time_series.dim * s_load_image.dim / 2 * sizeof(STORE_REAL), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {

		std::cout << "failed cudaMemcpy ram_power_spectra_" << std::endl;

	}*/

}
void pw_azth_avg2_gpu(unsigned int* lut, INDEX npw, STORE_REAL* ram_pw, CUFFT_COMPLEX* dev_images) {



	int threads2 = 32;
	int blocksx2 = (s_power_spectra.dim + threads2 - 1) / threads2;
	int blocksy2 = (npw + threads2 - 1) / threads2;

	dim3 THREADS3(threads2, threads2);
	dim3 BLOCKS3(blocksx2, blocksy2);

	//for (i = 0; i < dim_t; ++i)
	//{
	dev_images_TO_ram_power_spectra << <BLOCKS3, THREADS3 >> > (s_load_image.dim / 2,npw, s_power_spectra.dim, 
		dev_images, dev_radial_lut_gpu, ram_pw);

	STORE_REAL* ram_power_spectra_(NULL);

	ram_power_spectra_ = new STORE_REAL[s_time_series.dim * 512 * 512 / 2];

	
	int cudaStatus = cudaMemcpy(ram_power_spectra_,ram_pw, s_time_series.dim * 512 * 512 / 2 * sizeof(STORE_REAL), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {

		std::cout << "failed cudaMemcpy ram_power_spectra_" << std::endl;

	}
	/*for (int i = 1; i < npw; i++) {

		memset(ram_power_spectra, 0, sizeof(STORE_REAL) * s_load_image.dim / 2);

		for (int j = 0; j < s_power_spectra.dim; j++) {
			ram_power_spectra[lut[j]] = (STORE_REAL)dev_images_cpu[i + j * npw][0];
			//std::cout << ram_power_spectra[lut[j]] << std::endl;
		}
	}*/


}
void time_series_analysis_gpu() {

	cuda_exec mycuda_dim_t, mycuda_dim, mycuda_dim_dim_t;

	lldiv_t group;
	INDEX i, n_group, group_rem;

	time_time_correlation.start();
	
	calc_cuda_exec(s_time_series.dim, deviceProp.maxThreadsPerBlock, &mycuda_dim);
	calc_cuda_exec(useri.nthread_gpu, deviceProp.maxThreadsPerBlock, &mycuda_dim_t);
	calc_cuda_exec(s_time_series.dim * useri.nthread_gpu, deviceProp.maxThreadsPerBlock, &mycuda_dim_dim_t);


	group = std::div((long long)(s_time_series.numerosity), (long long)(useri.nthread_gpu));
	n_group = (INDEX)(group.quot);
	group_rem = (INDEX)(group.rem);

	//-----------------------------------------
	// NO GROUP SPLIT, IT IS ASSUMED THAT MEMORY WILL BE ENOUGH... TO BE CHANGED

	for (i = 0; i < n_group; ++i)
	{
		//std::cout << i << std::endl;
		timeseriesanalysis_gpu(s_time_series.dim, useri.nthread_gpu , &dev_images_gpu[i*useri.nthread_gpu* s_time_series.dim], s_fft_time.dim, dev_fft_time_gpu, &tplan, dev_corr_gpu,
			mycuda_dim_t, mycuda_dim, mycuda_dim_dim_t);

		
		
	}
	if (0 != group_rem)
	{
		calc_cuda_exec(group_rem, deviceProp.maxThreadsPerBlock, &mycuda_dim_t);
		calc_cuda_exec(s_time_series.dim * group_rem, deviceProp.maxThreadsPerBlock, &mycuda_dim_dim_t);
		timeseriesanalysis_gpu(s_time_series.dim, group_rem, &dev_images_gpu[i * useri.nthread_gpu * s_time_series.dim], s_fft_time.dim, dev_fft_time_gpu, &tplan, dev_corr_gpu,
			mycuda_dim_t, mycuda_dim, mycuda_dim_dim_t);

	}

	time_time_correlation.stop();

	time_from_device_to_host.start();
	cudaMemcpy(dev_images_cpu, dev_images_gpu, s_time_series.memory_tot, cudaMemcpyDeviceToHost);
	time_from_device_to_host.stop();

	/*timeseriesanalysis_gpu(s_time_series.dim, s_time_series.numerosity, dev_images_gpu, s_fft_time.dim, dev_fft_time_gpu, &tplan, dev_corr_gpu,
		mycuda_dim_t, mycuda_dim, mycuda_dim_dim_t);*/

	//timeseriesanalysis_gpu(dim, dim_tseries, yinputg2, dimp, yg2, &plan, out_f2_g, mycuda_dim_t, mycuda_dim, mycuda_dim_dim_t);
	
}

void timeseries_to_lutpw_gpu(INDEX dimcopy, CUFFT_REAL gain, INDEX t, INDEX starting_freq, STORE_REAL* ram_power_spectra) {

	cuda_exec mycuda_dim_ii;
	calc_cuda_exec(dimcopy, deviceProp.maxThreadsPerBlock, &mycuda_dim_ii);

	timeseries_to_lutfft_gpu(dimcopy, gain, t, starting_freq);
	//complex_to_real_with_gain_cpu(useri.nthread, dimcopy, dev_image_sot_cpu,
	//		(FFTW_REAL)(1.0), dev_power_spectra_cpu);
	complextorealwithgain_gpu<<<mycuda_dim_ii.nbk, mycuda_dim_ii.nth >>>
		( dimcopy, dev_image_sot_gpu,(FFTW_REAL)(1.0), dev_power_spectra_gpu);

	cudaMemcpy(ram_power_spectra, dev_power_spectra_gpu, sizeof(CUFFT_REAL)* dimcopy, cudaMemcpyDeviceToHost);



}
void timeseries_to_lutfft_gpu(INDEX dimcopy, CUFFT_REAL gain, INDEX t, INDEX starting_freq) {

	cuda_exec mycuda_dim_ii;
	calc_cuda_exec(dimcopy, deviceProp.maxThreadsPerBlock, &mycuda_dim_ii);

	cpx_col2row_gain_gpu<<<mycuda_dim_ii.nbk, mycuda_dim_ii.nth >>>(dimcopy, s_time_series.dim, t, dev_images_gpu, gain,
		(INDEX)(1), starting_freq, dev_image_sot_gpu);


}


void lutfft_to_timeseries_gpu(INDEX dimcopy, CUFFT_REAL gain, INDEX t, INDEX starting_freq) {

	cuda_exec mycuda_dim_i;
	calc_cuda_exec(dimcopy, deviceProp.maxThreadsPerBlock, &mycuda_dim_i);

	cpx_row2col_gain_gpu<<<mycuda_dim_i.nbk, mycuda_dim_i.nth >>>(dimcopy, (INDEX)(1), starting_freq, dev_image_sot_gpu, gain,
		s_time_series.dim, t, dev_images_gpu);

}

/*
void number_bk_th(INDEX dim, unsigned int n_max_th, unsigned int &nbk, unsigned int &nth)
{

	if(dim > n_max_th)
		{
			nth = n_max_th;
			nbk = dim / n_max_th;
		}
	else
		{
			nth = dim;
			nbk = 1;
		}
}*/

void calc_sizes(INDEX dimy, INDEX dimx, INDEX n_tot, INDEX size_of_element, sizes &s)
{
	s.dimx = dimx;
	s.dimy = dimy;
	s.numerosity = n_tot;
	
	s.dim = dimx * dimy;
	s.memory_one = s.dim * size_of_element;
	s.memory_tot = n_tot * s.memory_one;
	calc_cuda_exec(s.dim, deviceProp.maxThreadsPerBlock, &s.cexe);
}


int gpu_allocation(int flg_mode, INDEX &nimages, INDEX &dimy, INDEX &dimx, INDEX &dim_power_spectrum, unsigned int *ram_radial_lut)
{
	int alloc_status_li, alloc_status_fft, alloc_status_pw, alloc_status_im, alloc_status_plan , alloc_status_plan_time, alloc_status_rlut, alloc_status_imsot;
	int alloc_status_fftime, alloc_status_corr_g;
	INDEX i, free_video_memory, image_p_spectrum_memory;
	INDEX capacity;

	INDEX dimtimeseries_exponent, dimtimeseries_zeropadding;
	double capacity_d;

	calc_sizes(dimy, dimx, 1, sizeof(unsigned short), s_load_image);
	calc_sizes(1, dim_power_spectrum, 1, sizeof(unsigned int), s_radial_lut);
	calc_sizes(dimy, dimx, 1, sizeof(CUFFT_COMPLEX), s_fft);
	calc_sizes(1, dim_power_spectrum, 1, sizeof(CUFFT_COMPLEX), s_fft_images);
	calc_sizes(1, dim_power_spectrum, 1, sizeof(STORE_REAL), s_power_spectra);
	calc_sizes(1, nimages, 1, sizeof(CUFFT_COMPLEX), s_time_series);
	// Calculating the closes power of 2 that allows for zero-padding
	dimtimeseries_exponent = (INDEX)(std::ceil(std::log((FFTW_REAL)(nimages)) / std::log(2.0))) + 1;
	dimtimeseries_zeropadding = 1;
	for (i = 0; i < dimtimeseries_exponent; ++i) dimtimeseries_zeropadding *= 2;
	calc_sizes(1, dimtimeseries_zeropadding, 1, sizeof(CUFFT_COMPLEX), s_fft_time);

	// extimating video card capability of storing the same number of power spectra and
	// FFT of different images
	free_video_memory = (INDEX)(deviceProp.totalGlobalMem) - (s_load_image.memory_tot + s_fft.memory_tot + s_radial_lut.memory_tot + s_fft_images.memory_one);
	

	switch (useri.execution_mode)
	{
	case DIFFMICRO_MODE_FIFO:
		image_p_spectrum_memory = s_power_spectra.memory_one + s_fft_images.memory_one;

		capacity = free_video_memory / image_p_spectrum_memory;
		if (capacity > nimages) capacity = nimages;
		break;
	case DIFFMICRO_MODE_TIMECORRELATION:
		//memory.tot
		free_video_memory -= s_fft_time.memory_one * useri.nthread_gpu;
		image_p_spectrum_memory = s_time_series.memory_one;
		capacity = free_video_memory / image_p_spectrum_memory;
		if (capacity > s_fft_images.dim) capacity = s_fft_images.dim;
		break;
	default:
		std::cerr << "invalid diffmicro mode" << std::endl;
		return 1;
		break;
	}



	if(capacity == 0)
		{
			std::cerr <<"not enough video card memory for this task"<<std::endl;
			return false;
		}

		
	one_over_fft_norm = (CUFFT_REAL)(1./(sqrt((CUFFT_REAL)(dimx * dimy))));
	gpu_free_pointers();

	//----------------------------------------------------------------------------
	// ALLOCATION
	
	// trial and error allocation

	//----------------------------------------------------------
	// CUFFT initialization
#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
	alloc_status_plan = cufftPlan2d(&plan, dimy, dimx, CUFFT_C2C);
	//cufftExecC2C(plan, dev_fft, dev_fft, CUFFT_FORWARD);
#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
	alloc_status_plan = cufftPlan2d(&plan, dimy, dimx, CUFFT_Z2Z);
	//cufftExecZ2Z(plan, dev_fft, dev_fft, CUFFT_FORWARD);
#else
#error Unknown CUDA type selected
#endif

	if (cudaSuccess != alloc_status_plan)
	{
		std::cerr << "cuda error in inizializing plan for FFT" << std::endl;
		return 1;
	}

	switch (useri.execution_mode)
	{
	case DIFFMICRO_MODE_FIFO:

		

		alloc_status_im = cudaMalloc(&dev_images_gpu, s_fft_images.memory_one * capacity );
		alloc_status_pw = cudaMalloc(&dev_power_spectra_gpu, s_power_spectra.memory_one * capacity );
		alloc_status_fft = cudaMalloc(&dev_fft_gpu, s_fft.memory_tot );
		alloc_status_li = cudaMalloc(&dev_im_gpu, s_load_image.memory_tot );
		alloc_status_rlut = cudaMalloc(&dev_radial_lut_gpu, s_radial_lut.memory_tot );
		alloc_status_imsot = cudaMalloc(&dev_image_sot_gpu, s_fft_images.memory_one);
		capacity_d = (double)(capacity);
		while((cudaSuccess != alloc_status_pw) || (cudaSuccess != alloc_status_im) ||
					 (cudaSuccess != alloc_status_li) || (cudaSuccess != alloc_status_fft) ||
								(CUFFT_SUCCESS != alloc_status_plan) || (cudaSuccess != alloc_status_rlut) || (cudaSuccess != alloc_status_imsot))
			{
				//printf("capacity %u\r\n", capacity);
				capacity_d *= 0.95;
				capacity_d = std::floor( capacity_d * 0.95);
				capacity = (INDEX)(capacity_d);
				if(CUFFT_SUCCESS == alloc_status_plan) cufftDestroy(plan);
				if(cudaSuccess == alloc_status_pw) cudaFree(dev_power_spectra_gpu);
				if(cudaSuccess == alloc_status_im) cudaFree(dev_images_gpu);
				if(cudaSuccess == alloc_status_li) cudaFree(dev_im_gpu);
				if(cudaSuccess == alloc_status_fft) cudaFree(dev_fft_gpu);
				if(cudaSuccess == alloc_status_rlut) cudaFree(dev_radial_lut_gpu);
				if(cudaSuccess == alloc_status_imsot) cudaFree(dev_image_sot_gpu);
				//----------------------------------------------------------
				// CUFFT initialization
				#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
					alloc_status_plan = cufftPlan2d(&plan, dimy, dimx, CUFFT_C2C);
					//cufftExecC2C(plan, dev_fft, dev_fft, CUFFT_FORWARD);
				#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
					alloc_status_plan = cufftPlan2d(&plan, dimy, dimx, CUFFT_Z2Z);
					//cufftExecZ2Z(plan, dev_fft, dev_fft, CUFFT_FORWARD);
				#else
					#error Unknown CUDA type selected
				#endif

				alloc_status_im = cudaMalloc(&dev_images_gpu, s_fft_images.memory_one * capacity );
				alloc_status_pw = cudaMalloc(&dev_power_spectra_gpu, s_power_spectra.memory_one * capacity );
				alloc_status_fft = cudaMalloc(&dev_fft_gpu, s_fft.memory_tot );
				alloc_status_li = cudaMalloc(&dev_im_gpu, s_load_image.memory_tot );
				alloc_status_rlut = cudaMalloc(&dev_radial_lut_gpu, s_radial_lut.memory_tot);
				alloc_status_imsot = cudaMalloc(&dev_image_sot_gpu, s_fft_images.memory_one);

			}
		


		//----------------------------------------------------------
		//----------------------------------------------------------
		//----------------------------------------------------------
		// this operation that seems unuseful in fact reset the plan in a such a way that, if ever some allocation errors
		// as occured, the program will work properly
		// this fact is purely experimental and I cannot explain why it happens!!!
				// CUFFT initialization
				#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
					cufftExecC2C(plan, dev_fft, dev_fft, CUFFT_FORWARD);
				#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
					cufftExecZ2Z(plan, dev_fft_gpu, dev_fft_gpu, CUFFT_FORWARD);
				#else
					#error Unknown CUDA type selected
				#endif
		cudaDeviceSynchronize();

		//----------------------------------------------------------
		//----------------------------------------------------------
		//----------------------------------------------------------


		calc_sizes(1, dim_power_spectrum, capacity, sizeof(CUFFT_COMPLEX), s_fft_images);
		tot_memory_fft_images = s_fft_images.memory_tot;

		calc_sizes(1, dim_power_spectrum, capacity, sizeof(STORE_REAL), s_power_spectra);
		tot_calculation_memory = s_power_spectra.memory_tot;
		cudaMemcpy(dev_radial_lut_gpu, ram_radial_lut, s_radial_lut.memory_one, cudaMemcpyHostToDevice);
		break;

	case DIFFMICRO_MODE_TIMECORRELATION:
	{
		

		int n[1] = { s_fft_time.dim };

#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
		alloc_status_plan_time = cufftPlanMany(&tplan, 1, n,
			NULL, 1, s_fft_time.dim,  //advanced data layout, NULL shuts it off
			NULL, 1, s_fft_time.dim,  //advanced data layout, NULL shuts it off
			CUFFT_C2C, useri.nthread_gpu);
		//cufftExecC2C(tplan, dev_fft, dev_fft, CUFFT_FORWARD);
#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
		alloc_status_plan_time = cufftPlanMany(&tplan, 1, n,
			NULL, 1, s_fft_time.dim,  //advanced data layout, NULL shuts it off
			NULL, 1, s_fft_time.dim,  //advanced data layout, NULL shuts it off
			CUFFT_Z2Z, useri.nthread_gpu);
		//cufftExecZ2Z(tplan, dev_fft, dev_fft, CUFFT_FORWARD);
#else
#error Unknown CUDA type selected
#endif
		cudaDeviceSynchronize();

		if (cudaSuccess != alloc_status_plan_time)
		{
			std::cerr << "cuda error in inizializing plan for FFT" << std::endl;
			return 1;
		}

		
	

		alloc_status_im = cudaMalloc(&dev_images_gpu, s_time_series.memory_one * capacity);
		alloc_status_fftime = cudaMalloc(&dev_fft_time_gpu, s_fft_time.memory_one * useri.nthread_gpu);
		alloc_status_corr_g = cudaMalloc(&dev_corr_gpu, s_time_series.memory_one * useri.nthread_gpu);
		alloc_status_pw = cudaMalloc(&dev_power_spectra_gpu, s_power_spectra.memory_one );
		alloc_status_fft = cudaMalloc(&dev_fft_gpu, s_fft.memory_tot);
		alloc_status_li = cudaMalloc(&dev_im_gpu, s_load_image.memory_tot);
		alloc_status_rlut = cudaMalloc(&dev_radial_lut_gpu, s_radial_lut.memory_tot);
		alloc_status_imsot = cudaMalloc(&dev_image_sot_gpu, s_fft_images.memory_one);

		/*int alloc_status_testpw = cudaMalloc(&test_pw, nimages*dimx*dimy/2*sizeof(STORE_REAL));
		if (cudaSuccess != alloc_status_testpw) {

			std::cout << "ERROR CudaMalloc Test" << std::endl;
		}*/
		



		
		

		capacity_d = (double)(capacity);

		while ((cudaSuccess != alloc_status_pw) || (cudaSuccess != alloc_status_im) ||
			(cudaSuccess != alloc_status_li) || (cudaSuccess != alloc_status_fft) ||
			(CUFFT_SUCCESS != alloc_status_plan) || (cudaSuccess != alloc_status_rlut) || (cudaSuccess != alloc_status_imsot) ||
			 (cudaSuccess != alloc_status_fftime) || (cudaSuccess != alloc_status_corr_g) )
		{
			//printf("capacity %u\r\n", capacity);
			capacity_d *= 0.95;
			capacity_d = std::floor(capacity_d * 0.95);
			capacity = (INDEX)(capacity_d);
			if (CUFFT_SUCCESS == alloc_status_plan) cufftDestroy(tplan);
			if (cudaSuccess == alloc_status_pw) cudaFree(dev_power_spectra_gpu);
			if (cudaSuccess == alloc_status_im) cudaFree(dev_images_gpu);
			if (cudaSuccess == alloc_status_li) cudaFree(dev_im_gpu);
			if (cudaSuccess == alloc_status_fft) cudaFree(dev_fft_gpu);
			if (cudaSuccess == alloc_status_rlut) cudaFree(dev_radial_lut_gpu);
			if (cudaSuccess == alloc_status_imsot) cudaFree(dev_image_sot_gpu);
			if (cudaSuccess == alloc_status_fftime) cudaFree(dev_fft_time_gpu);
			if (cudaSuccess == alloc_status_corr_g) cudaFree(dev_corr_gpu);
			//if (cudaSuccess == alloc_status_testpw) cudaFree(test_pw);


			//----------------------------------------------------------
			// CUFFT initialization
#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
			alloc_status_plan_time = cufftPlanMany(&tplan, 1, n,
				NULL, 1, s_fft_time.dim,  //advanced data layout, NULL shuts it off
				NULL, 1, s_fft_time.dim,  //advanced data layout, NULL shuts it off
				CUFFT_C2C, useri.nthread_gpu);
			//cufftExecC2C(tplan, dev_fft, dev_fft, CUFFT_FORWARD);
#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
			alloc_status_plan_time = cufftPlanMany(&tplan, 1, n,
				NULL, 1, s_fft_time.dim,  //advanced data layout, NULL shuts it off
				NULL, 1, s_fft_time.dim,  //advanced data layout, NULL shuts it off
				CUFFT_Z2Z, useri.nthread_gpu);
			//cufftExecZ2Z(tplan, dev_fft, dev_fft, CUFFT_FORWARD);
#else
#error Unknown CUDA type selected
#endif
			cudaDeviceSynchronize();

			alloc_status_im = cudaMalloc(&dev_images_gpu, s_time_series.memory_one * capacity);
			alloc_status_corr_g = cudaMalloc(&dev_corr_gpu, s_time_series.memory_one * useri.nthread_gpu);
			alloc_status_fftime = cudaMalloc(&dev_fft_time_gpu, s_fft_time.memory_one * useri.nthread_gpu);
			alloc_status_pw = cudaMalloc(&dev_power_spectra_gpu, s_power_spectra.memory_one );
			alloc_status_fft = cudaMalloc(&dev_fft_gpu, s_fft.memory_tot);
			alloc_status_li = cudaMalloc(&dev_im_gpu, s_load_image.memory_tot);
			alloc_status_rlut = cudaMalloc(&dev_radial_lut_gpu, s_radial_lut.memory_tot);
			alloc_status_imsot = cudaMalloc(&dev_image_sot_gpu, s_fft_images.memory_one);
			
			//alloc_status_testpw = cudaMalloc(&test_pw, nimages * dimx * dimy / 2 * sizeof(STORE_REAL));


		}



		//----------------------------------------------------------
		//----------------------------------------------------------
		//----------------------------------------------------------
		// this operation that seems unuseful in fact reset the plan in a such a way that, if ever some allocation errors
		// as occured, the program will work properly
		// this fact is purely experimental and I cannot explain why it happens!!!
				// CUFFT initialization
//#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
//		cufftExecC2C(plan, dev_fft, dev_fft, CUFFT_FORWARD);
//#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
//		cufftExecZ2Z(plan, dev_fft_gpu, dev_fft_gpu, CUFFT_FORWARD);
//#else
//#error Unknown CUDA type selected
//#endif
		//cudaDeviceSynchronize();

		//----------------------------------------------------------
		//----------------------------------------------------------
		//----------------------------------------------------------
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		//calc_sizes(1, dim_power_spectrum, capacity, sizeof(CUFFT_COMPLEX), s_fft_images);
		//calc_sizes(1, dim_power_spectrum, capacity, sizeof(STORE_REAL), s_power_spectra);

		// CUFFT initialization
#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
		cufftExecC2C(plan, dev_fft, dev_fft, CUFFT_FORWARD);
#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
		cufftExecZ2Z(plan, dev_fft_gpu, dev_fft_gpu, CUFFT_FORWARD);
#else
#error Unknown CUDA type selected
#endif
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));

		calc_sizes(1, nimages, capacity, sizeof(CUFFT_COMPLEX), s_time_series);

		tot_memory_fft_images = s_time_series.memory_tot;
		
		calc_sizes(1, dimtimeseries_zeropadding, useri.nthread_gpu, sizeof(CUFFT_COMPLEX), s_fft_time);
		tot_calculation_memory = s_fft_time.memory_tot;

		cudaMemcpy(dev_radial_lut_gpu, ram_radial_lut, s_radial_lut.memory_one, cudaMemcpyHostToDevice);

		err = cudaGetLastError();
		if (err != cudaSuccess)
			printf("Error: %s\n", cudaGetErrorString(err));


		lldiv_t group = std::div((long long)(s_fft_images.dim), (long long)(s_time_series.numerosity));
		INDEX n_group = (INDEX)(group.quot);
		if (group.rem != 0) n_group = n_group + 1;

		dev_images_cpu = new FFTW_COMPLEX[n_group*s_time_series.dim * capacity];
		dev_image_sot_cpu = new FFTW_COMPLEX[s_fft_images.dim];
		dev_power_spectra_cpu = new FFTW_REAL[s_power_spectra.dim];

		break;
	}
	default:
		break;
	}
	/*dev_radial_lut_gpu = new unsigned int[s_radial_lut.dim];
	for (i = 0; i < s_radial_lut.dim; ++i)
		dev_radial_lut_gpu[i] = ram_radial_lut[i];*/

	n_capacity = capacity;
	

	if(capacity == 0)	return 1;
	return 0;
}

void gpu_free_pointers()
{
	
	if(NULL != dev_im_gpu)
		{
			cudaFree(dev_im_gpu);
			dev_im_gpu = NULL;
		}
	if(NULL != dev_fft_gpu)
		{
			cudaFree(dev_fft_gpu);
			dev_fft_gpu = NULL;
		}
	if(NULL != dev_images_gpu)
		{
			cudaFree(dev_images_gpu);
			dev_images_gpu = NULL;
		}
	if(NULL != dev_power_spectra_gpu )
		{
			cudaFree(dev_power_spectra_gpu);
			dev_power_spectra_gpu = NULL;
		}
	if (NULL != dev_radial_lut_gpu)
	{
		cudaFree(dev_radial_lut_gpu);
		dev_radial_lut_gpu = NULL;
	}

	if (NULL != dev_fft_time_gpu)
	{
		cudaFree(dev_fft_time_gpu);
		dev_fft_time_gpu = NULL;
	}

	if (NULL != dev_corr_gpu)
	{
		cudaFree(dev_corr_gpu);
		dev_corr_gpu = NULL;
	}

	if (NULL != dev_image_sot_gpu)
	{
		cudaFree(dev_image_sot_gpu);
		dev_image_sot_gpu = NULL;
	}

	if (NULL != dev_image_sot_cpu)
	{
		delete[] dev_image_sot_cpu;
		dev_image_sot_cpu = NULL;
	}

	if (NULL != dev_images_cpu)
	{
		delete [] dev_images_cpu;
		dev_images_cpu = NULL;
	}

	if (NULL != dev_power_spectra_cpu)
	{
		delete[] dev_power_spectra_cpu;
		dev_power_spectra_cpu = NULL;
	}

}

void gpu_deallocation()
{
	gpu_free_pointers();
	cufftDestroy(plan);
	cufftDestroy(tplan);
}


int image_to_dev_gpu(SIGNED_INDEX ind_fifo, STORE_REAL &mean, unsigned short *im, bool flg_debug)
{
	CUFFT_COMPLEX* dev_store_ptr;

	// selecting the correct memory area
	if (0 > ind_fifo)
		dev_store_ptr = dev_image_sot_gpu;
	else
		dev_store_ptr = &(dev_images_gpu[ind_fifo * s_fft_images.dim]);

	if (true == flg_debug)
	{
		double *img[2];
		if (NULL == image_l) image_l = new double[s_load_image.dim];
		if ((NULL == module_fft) && (NULL == phase_fft) && (NULL == tmp_display_i16) && (NULL == tmp_display_cpx))
		{
			module_fft = new double[s_load_image.dim];
			phase_fft = new double[s_load_image.dim];
			img[0] = module_fft; img[1] = phase_fft;
			tmp_display_i16 = new unsigned __int16[s_load_image.dim];
			tmp_display_cpx = new CUFFT_COMPLEX[s_load_image.dim];
		}
		if ((NULL == fft_window) && (NULL == image_window))
		{
			bool flg_colormap = true;
			fft_window = new_figure(s_load_image.dimx, s_load_image.dimy, 2, img);
			image_window = new_figure(s_load_image.dimx, s_load_image.dimy, image_l);
			fft_window->control[ID_BUTTON_COLORMAP]->set_value(&flg_colormap);
			fft_window->colormap = cv::COLORMAP_HSV;
		}
	}

	INDEX i;
	int ret = 0;
	CUFFT_REAL mean_tmp;

	// memory copy from RAM to video card
	time_from_host_to_device.start();
	cudaMemcpy( dev_im_gpu, im, s_load_image.memory_one, cudaMemcpyHostToDevice);
	time_from_host_to_device.stop();

	

	if (true == flg_debug)
	{
		cudaMemcpy(tmp_display_i16, dev_im_gpu, s_load_image.memory_one, cudaMemcpyDeviceToHost);
		for (i = 0; i < s_load_image.dim; ++i) image_l[i] = (double)(tmp_display_i16[i]);
		image_window->show();
		waitkeyboard();
	}

 time_fft_norm.start();



	// from image to complex matrix
	short_to_real_with_gain<<<s_load_image.cexe.nbk, s_load_image.cexe.nth>>>
		                      (s_load_image.dim ,dev_im_gpu, (CUFFT_REAL)(one_over_fft_norm / 65536.), dev_fft_gpu);
	cudaDeviceSynchronize();
	if (true == flg_debug)
	{
		cudaMemcpy(tmp_display_cpx, dev_fft_gpu, s_fft.memory_one, cudaMemcpyDeviceToHost);
		for (i = 0; i < s_load_image.dim; ++i)	image_l[i] = tmp_display_cpx[i].x;
		complex_to_module_phase(s_load_image.dim, tmp_display_cpx, module_fft, phase_fft);
		image_window->show();
		fft_window->show();
		waitkeyboard();
	}



	// FFT execution
	#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
	 cufftExecC2C(plan, dev_fft, dev_fft, CUFFT_FORWARD);
//		cufftExecC2C(plan, dev_fft, dev_fft, CUFFT_FORWARD);
	#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
	 cufftExecZ2Z(plan, dev_fft_gpu, dev_fft_gpu, CUFFT_FORWARD);
	#else
		#error Unknown CUDA type selected
	#endif


	cudaDeviceSynchronize();



	// normalization
	cudaMemcpy( &mean_tmp, dev_fft_gpu, sizeof(CUFFT_REAL), cudaMemcpyDeviceToHost);

	mean = mean_tmp;
	if(mean < 0.000000000000001)
		{
			mean = 1.;
			mean_tmp = 1.;
			ret = 1;
			waitkeyboard(0);
		}
	mean_tmp =	(CUFFT_REAL)(1./mean_tmp);
	
	//mean_tmp = 1.0;
	//gain_complex<<<s_fft_images.cexe.nbk, s_fft_images.cexe.nth>>>(mean_tmp,s_fft_images.dim, dev_fft, dev_store_ptr);
	gain_complex_lut<<<s_fft_images.cexe.nbk, s_fft_images.cexe.nth >>>
		(mean_tmp, s_fft_images.dim, dev_radial_lut_gpu, dev_fft_gpu, dev_store_ptr);
	cudaDeviceSynchronize();
	time_fft_norm.stop();

	

	if (true == flg_debug)
	{
		cudaMemcpy(tmp_display_cpx, dev_fft_gpu, s_fft.memory_one, cudaMemcpyDeviceToHost);
		complex_to_module_phase(s_load_image.dim, tmp_display_cpx, module_fft, phase_fft);
		module_fft[0] = 0.0;
		image_window->show();
		fft_window->show();
		waitkeyboard(5);

		cudaMemcpy(tmp_display_cpx, dev_store_ptr, s_fft_images.memory_one, cudaMemcpyDeviceToHost);
		memset(module_fft, 0, s_fft.dim * sizeof(double));
		memset(phase_fft, 0, s_fft.dim * sizeof(double));
		for (i = 0; i < s_fft_images.dim; ++i)
		{
			module_fft[ram_radial_lut[i]] = sqrt(tmp_display_cpx[i].x*tmp_display_cpx[i].x + tmp_display_cpx[i].y*tmp_display_cpx[i].y);
			phase_fft[ram_radial_lut[i]] = atan2(tmp_display_cpx[i].y, tmp_display_cpx[i].x);
			module_fft[ram_radial_lut[i]] = tmp_display_cpx[i].x;
		}
		/*
		for (i = 0; i < s_fft_images.dim; ++i)
		{
			module_fft[ram_radial_lut[i]] = i+0.1;
			phase_fft[ram_radial_lut[i]] = 0;
		}
		cudaMemcpy(ram_radial_lut, dev_radial_lut, s_fft_images.dim*sizeof(unsigned int), cudaMemcpyDeviceToHost);
		for (i = 0; i < s_fft_images.dim; ++i)
		{
			module_fft[ram_radial_lut[i]] -= i;
	//		phase_fft[ram_radial_lut[i]] = i;
		}*/
		fft_window->show();
		waitkeyboard(5);
	}

	++n_computed_fft;

	

	return ret;
}

void copy_power_spectra_from_dev_gpu(STORE_REAL *power_spectrum_r)
{
	cudaMemcpy(power_spectrum_r, dev_power_spectra_gpu, s_power_spectra.memory_tot, cudaMemcpyDeviceToHost);
}


void diff_power_spectrum_to_avg_gpu_gpu(CUFFT_REAL coef1, CUFFT_REAL coef2, INDEX j, INDEX ind_dist)
{
	diff_power_spectrum_to_avg_gpu << <s_power_spectra.cexe.nbk, s_power_spectra.cexe.nth >> >
		(s_power_spectra.dim, &(dev_images_gpu[j * s_fft_images.dim]), dev_image_sot_gpu, coef1, coef2, &(dev_power_spectra_gpu[ind_dist * s_power_spectra.dim]));

	//sync
	cudaDeviceSynchronize();
}

void timeseriesanalysis_gpu_2D(INDEX ii, INDEX id_group,INDEX dimtimeseries, INDEX dim_t, CUFFT_COMPLEX* tseries, INDEX dimfft, CUFFT_COMPLEX* fft_memory, cufftHandle* tplan, CUFFT_REAL* corr_memory, cuda_exec mycuda_dim_t, cuda_exec mycuda_dim, cuda_exec mycuda_dim_dim_t)
{
	INDEX i;

	cuda_exec mycuda_dim_p;
	cuda_exec mycuda_dim_dim_p;

	calc_cuda_exec(dimfft, deviceProp.maxThreadsPerBlock, &mycuda_dim_p);
	calc_cuda_exec(dimfft * dim_t, deviceProp.maxThreadsPerBlock, &mycuda_dim_dim_p);



	//----------------------------------------------
	// Calculating the average of the absolute squares
	averagesabs2_array_gpu << < mycuda_dim_t.nbk, mycuda_dim_t.nth >> > (dimtimeseries, dim_t, tseries, corr_memory);

	/*CUFFT_REAL* dev_corr_cpu1(NULL);
	dev_corr_cpu1 = new CUFFT_REAL[s_time_series.dim * useri.nthread_gpu];

	cudaMemcpy(dev_corr_cpu1, corr_memory, s_time_series.dim * useri.nthread_gpu *sizeof(CUFFT_REAL), cudaMemcpyDeviceToHost);
	FILE* version2;
	version2 = fopen("v2_abs2.txt", "a");
	for (int ii = 0; ii < useri.nthread_gpu * s_time_series.dim; ++ii)
		//fprintf()
		fprintf(version2, "%f \n", dev_corr_cpu1[ii]);

	fclose(version2);*/


	//----------------------------------------------
	// preparing Fourier variables

	//-----------------------------------------------
	// zeroing all fft memory for zero padding and copying data with normalization to fft memory
	/*for (i = 0; i < dimfft; ++i)
	{
		fft_memory[i].x = 0.0;
		fft_memory[i].y = 0.0;
	}*/

	gaincomplex_gpu << <mycuda_dim_dim_p.nbk, mycuda_dim_dim_p.nth >> > (dimfft * dim_t, fft_memory, 0.0, fft_memory);

	cudaDeviceSynchronize();

	//std::cout << cudaGetLastError() << std::endl;

	int threads2 = 32;
	int blocksx2 = (dimtimeseries + threads2 - 1) / threads2;
	int blocksy2 = (dim_t + threads2 - 1) / threads2;

	dim3 THREADS3(threads2, threads2);
	dim3 BLOCKS3(blocksx2, blocksy2);

	//for (i = 0; i < dim_t; ++i)
	//{
	gaincomplex_gpu_2d << <BLOCKS3, THREADS3 >> > (dimfft, dim_t, dimtimeseries, tseries,
		(FFTW_REAL)(1. / std::sqrt((FFTW_REAL)(dimfft))), fft_memory);
	//----------------------------------------------

	cudaDeviceSynchronize();
	//std::cout << cudaGetLastError() << std::endl;
//}

/*CUFFT_COMPLEX* dev_corr_cpu1(NULL);
dev_corr_cpu1 = new CUFFT_COMPLEX[dimfft * useri.nthread_gpu];

cudaMemcpy(dev_corr_cpu1, fft_memory, dimfft * useri.nthread_gpu * sizeof(CUFFT_COMPLEX), cudaMemcpyDeviceToHost);
FILE* version2;
version2 = fopen("v2_gaincomplex_gpu.txt", "a");
for (int ii = 0; ii < useri.nthread_gpu * dimfft; ++ii)
	//fprintf()
	fprintf(version2, "%f    %f \n", dev_corr_cpu1[ii].x, dev_corr_cpu1[ii].y);

fclose(version2);*/


// FFT execution
// FFT execution
#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
	cufftExecC2C(tplan[0], fft_memory, fft_memory, CUFFT_FORWARD);
	//		cufftExecC2C(plan, dev_fft, dev_fft, CUFFT_FORWARD);
#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
	cufftExecZ2Z(tplan[0], fft_memory, fft_memory, CUFFT_FORWARD);
#else
#error Unknown CUDA type selected
#endif

	cudaDeviceSynchronize();
	//std::cout << cudaGetLastError() << std::endl;

	/*CUFFT_COMPLEX* dev_corr_cpu1(NULL);
	dev_corr_cpu1 = new CUFFT_COMPLEX[dimfft * useri.nthread_gpu];

	cudaMemcpy(dev_corr_cpu1, fft_memory, dimfft * useri.nthread_gpu * sizeof(CUFFT_COMPLEX), cudaMemcpyDeviceToHost);
	FILE* version2;
	version2 = fopen("v21.txt", "a");
	for (int ii = 0; ii < dimfft * useri.nthread_gpu; ++ii)
		//fprintf()
		fprintf(version2, "%d   %f    %f \n", ii, dev_corr_cpu1[ii].x, dev_corr_cpu1[ii].y);

	fclose(version2);*/


	//fftw_execute(tplan[0]);

	// evaluating abs^2
	// change my cuda to bigger one
	complexabs2_gpu << <mycuda_dim_dim_p.nbk, mycuda_dim_dim_p.nth >> > (dimfft * dim_t, fft_memory, fft_memory);
	cudaDeviceSynchronize();
	//std::cout << cudaGetLastError() << std::endl;
	// FFT execution. Given the conditions
	// - we are only interested in the real part
	// - we start from a real function generated by the absolute value
	// Then the direct and inverse FFT are equivalent. We can re-use plan_direct!

#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
	cufftExecC2C(tplan[0], fft_memory, fft_memory, CUFFT_FORWARD);
	//		cufftExecC2C(plan, dev_fft, dev_fft, CUFFT_FORWARD);
#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
	cufftExecZ2Z(tplan[0], fft_memory, fft_memory, CUFFT_FORWARD);
#else
#error Unknown CUDA type selected
#endif

	cudaDeviceSynchronize();
	//std::cout << cudaGetLastError() << std::endl;

	//int threads2 = 32;
	//int blocksx2 = (dimtimeseries + threads2 - 1) / threads2;
	//int blocksy2 = (dim_t + threads2 - 1) / threads2;

	//dim3 THREADS3(threads2, threads2);
	//dim3 BLOCKS3(blocksx2, blocksy2);
	
	/*if ((id_group == 0)&&(ii==0)) {
		std::cout << "oooooooooooooooooooooooooooooooooooo" << std::endl;
		gaincomplex_gpu_test << <mycuda_dim_dim_p.nbk, mycuda_dim_dim_p.nth >> > (4000 * s_load_image.dim / 2, test_pw, 0.0, test_pw);
	}*/

	updatewithdivrebyramp_gpu_2d << <BLOCKS3, THREADS3 >> > (ii, dev_radial_lut_gpu,
		s_load_image.dim / 2, id_group, dimfft, dim_t, dimtimeseries, dimtimeseries, fft_memory, corr_memory, tseries);//,test_pw);,s_time_series.numerosity,useri.nthread_gpu,
	// copy the result  back to original memory area

	cudaDeviceSynchronize();
	//std::cout << cudaGetLastError() << std::endl;
	//for (i = 0; i < dim_t; ++i)
	//{
	//copyfrom_gpu << <mycuda_dim.nbk, mycuda_dim.nth >> > (dimtimeseries, &tseries[i * dimtimeseries], &corr_memory[i * dimtimeseries]);
	//cudaDeviceSynchronize();
	//std::cout << cudaGetLastError() << std::endl;
	/*for (INDEX	j = 0; j < dimtimeseries; ++j)
	{
		tseries[j+ i * dimtimeseries].x = corr_memory[j + i * dimtimeseries];
		tseries[j + i * dimtimeseries].y = 0.0;
	}*/
	//}

/*CUFFT_REAL* dev_corr_cpu1(NULL);
dev_corr_cpu1 = new CUFFT_REAL[s_time_series.dim * useri.nthread_gpu];

cudaMemcpy(dev_corr_cpu1, corr_memory, s_time_series.dim * useri.nthread_gpu * sizeof(CUFFT_REAL), cudaMemcpyDeviceToHost);
FILE* version2;
version2 = fopen("v22.txt", "a");
for (int ii = 0; ii < useri.nthread_gpu * s_time_series.dim; ++ii)
	//fprintf()
	fprintf(version2, "%d   %.10f \n", ii, dev_corr_cpu1[ii]);

fclose(version2);*/


}


void timeseriesanalysis_gpu(INDEX dimtimeseries, INDEX dim_t, CUFFT_COMPLEX* tseries, INDEX dimfft, CUFFT_COMPLEX* fft_memory, cufftHandle* tplan, CUFFT_REAL* corr_memory, cuda_exec mycuda_dim_t, cuda_exec mycuda_dim, cuda_exec mycuda_dim_dim_t)
{
	INDEX i;

	
	cuda_exec mycuda_dim_p;
	cuda_exec mycuda_dim_dim_p;

	calc_cuda_exec(dimfft, deviceProp.maxThreadsPerBlock, &mycuda_dim_p);
	calc_cuda_exec(dimfft* dim_t, deviceProp.maxThreadsPerBlock, &mycuda_dim_dim_p);

	

	//----------------------------------------------
	// Calculating the average of the absolute squares
	averagesabs2_array_gpu << < mycuda_dim_t.nbk, mycuda_dim_t.nth >> > (dimtimeseries,dim_t, tseries, corr_memory);
	

	//----------------------------------------------
	// preparing Fourier variables

	//-----------------------------------------------
	// zeroing all fft memory for zero padding and copying data with normalization to fft memory
	/*for (i = 0; i < dimfft; ++i)
	{
		fft_memory[i].x = 0.0;
		fft_memory[i].y = 0.0;
	}*/

	gaincomplex_gpu << <mycuda_dim_dim_p.nbk, mycuda_dim_dim_p.nth >> > (dimfft * dim_t, fft_memory, 0.0, fft_memory);

	cudaDeviceSynchronize();

	//std::cout << cudaGetLastError() << std::endl;
	for (i = 0; i < dim_t; ++i)
	{
		gaincomplex_gpu << <mycuda_dim.nbk, mycuda_dim.nth >> > (dimtimeseries, &tseries[i * dimtimeseries], (FFTW_REAL)(1. / std::sqrt((FFTW_REAL)(dimfft))), &fft_memory[i * dimfft]);
		//----------------------------------------------

		cudaDeviceSynchronize();
		//std::cout << cudaGetLastError() << std::endl;
	}
		// FFT execution
		// FFT execution
		#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
				cufftExecC2C(tplan[0], fft_memory, fft_memory, CUFFT_FORWARD);
				//		cufftExecC2C(plan, dev_fft, dev_fft, CUFFT_FORWARD);
		#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
				cufftExecZ2Z(tplan[0], fft_memory, fft_memory, CUFFT_FORWARD);
		#else
		#error Unknown CUDA type selected
		#endif

		cudaDeviceSynchronize();
		//std::cout << cudaGetLastError() << std::endl;


		//fftw_execute(tplan[0]);

		// evaluating abs^2
		// change my cuda to bigger one
		complexabs2_gpu << <mycuda_dim_dim_p.nbk, mycuda_dim_dim_p.nth >> > (dimfft * dim_t, fft_memory, fft_memory);
		cudaDeviceSynchronize();
		//std::cout << cudaGetLastError() << std::endl;
		// FFT execution. Given the conditions
		// - we are only interested in the real part
		// - we start from a real function generated by the absolute value
		// Then the direct and inverse FFT are equivalent. We can re-use plan_direct!

		#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
				cufftExecC2C(tplan[0], fft_memory, fft_memory, CUFFT_FORWARD);
				//		cufftExecC2C(plan, dev_fft, dev_fft, CUFFT_FORWARD);
		#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
				cufftExecZ2Z(tplan[0], fft_memory, fft_memory, CUFFT_FORWARD);
		#else
		#error Unknown CUDA type selected
		#endif

		cudaDeviceSynchronize();
		//std::cout << cudaGetLastError() << std::endl;
for (i = 0; i < dim_t; ++i)
{
		updatewithdivrebyramp_gpu << <mycuda_dim.nbk, mycuda_dim.nth >> > (dimtimeseries, dimtimeseries, &fft_memory[i * dimfft], &corr_memory[i * dimtimeseries]);
		// copy the result  back to original memory area

		cudaDeviceSynchronize();
		//std::cout << cudaGetLastError() << std::endl;

		copyfrom_gpu << <mycuda_dim.nbk, mycuda_dim.nth >> > (dimtimeseries, &tseries[i * dimtimeseries], &corr_memory[i * dimtimeseries]);
		cudaDeviceSynchronize();
		//std::cout << cudaGetLastError() << std::endl;
		/*for (INDEX	j = 0; j < dimtimeseries; ++j)
		{
			tseries[j+ i * dimtimeseries].x = corr_memory[j + i * dimtimeseries];
			tseries[j + i * dimtimeseries].y = 0.0;
		}*/
	}
}

void Image_to_complex_matrix3(INDEX dimfr, INDEX ifr, int i, INDEX nimages) {

	//CUFFT_COMPLEX* dev_store_ptr;
	//dev_store_ptr = &(dev_images_gpu[i * s_fft_images.dim]);
	//int alloc_status_im = cudaMalloc(&dev_images_gpu1, nimages * s_power_spectra.dim * sizeof(CUFFT_COMPLEX));

	time_fft_norm.start();

	short_to_real_with_gain << <s_load_image.cexe.nbk, s_load_image.cexe.nth >> >
		(s_load_image.dim, dev_im_gpu, (CUFFT_REAL)(one_over_fft_norm), dev_fft_gpu);

	cufftExecZ2Z(plan, dev_fft_gpu, dev_fft_gpu, CUFFT_FORWARD);

	cudaDeviceSynchronize();


	CUFFT_REAL mean_tmp;
	STORE_REAL mean;
	// normalization
	cudaMemcpy(&mean_tmp, dev_fft_gpu, sizeof(CUFFT_REAL), cudaMemcpyDeviceToHost);

	mean = mean_tmp;
	if (mean < 0.000000000000001)
	{
		mean = 1.;
		mean_tmp = 1.;
		//ret = 1;
		waitkeyboard(0);
	}
	mean_tmp = (CUFFT_REAL)(1. / mean_tmp);

	//mean_tmp = (CUFFT_REAL)1;
	/*int threads = 1024;
	int blocksx = (s_power_spectra.dim + threads - 1) / threads;
	//int blocksx = (fft_size + threads - 1) / threads;
	//int blocksy1 = (nb_fft + threads1 - 2) / threads1;
	dim3 THREADS(threads);
	dim3 BLOCKS(blocksx);

	gain_complex_lut_timeSeries << <BLOCKS, THREADS >> >
		(s_fft_time.dim, (FFTW_REAL)(1. / std::sqrt((FFTW_REAL)(s_fft_time.dim))), i, nimages, mean_tmp,
			s_fft_images.dim, dev_radial_lut_gpu, dev_fft_gpu, dev_fft_time_gpu1);*/

	cuda_exec mycuda_dim_i;
	calc_cuda_exec(dimfr, deviceProp.maxThreadsPerBlock, &mycuda_dim_i);

	cpx_row2col_gain_lut_gpu << <mycuda_dim_i.nbk, mycuda_dim_i.nth >> > (mean_tmp, dev_radial_lut_gpu, dimfr,
		(INDEX)(1), ifr, dev_fft_gpu, (FFTW_REAL)(1.0), s_time_series.dim, i, dev_images_gpu);

	time_fft_norm.stop();

	cudaDeviceSynchronize();
	/*CUFFT_COMPLEX* tmp_display_cpx_(NULL);

	tmp_display_cpx_ = new CUFFT_COMPLEX[s_power_spectra.dim * nimages];

	cudaMemcpy(tmp_display_cpx_, dev_images_gpu, nimages*s_fft_images.memory_one, cudaMemcpyDeviceToHost);
	for (int ii = 0; ii < s_fft_images.dim* nimages; ++ii)
		std::cout << tmp_display_cpx_[ii].x << "  + i " << tmp_display_cpx_[ii].y << std::endl;

	FILE* version3;
	version3 = fopen("v4_timeSeries.txt", "w");
	for (int ii = 0; ii < nimages * s_power_spectra.dim; ++ii)
		//fprintf()
		fprintf(version3, "%d   %f    %f\n", ii, tmp_display_cpx_[ii].x, tmp_display_cpx_[ii].y);

	fclose(version3);*/

}

void power_spectra_from_dev(INDEX n_pw, STORE_REAL ram_power_spectra[])
{
	time_from_device_to_host.start();
	cudaMemcpy(ram_power_spectra, dev_power_spectra_gpu, s_power_spectra.memory_one * n_pw, cudaMemcpyDeviceToHost);
	time_from_device_to_host.stop();
}


