/*
Copyright: Mojtaba Norouzisadeh, Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
e-mail: norouzi.mojtaba.sade@gmail.com

date: 05/2020 - 09/2020
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

#include "timeavg_by_correlation.h"
#include "device_launch_parameters.h"
#include "figure_opencv.h"
#include "correlation.h"
#include <cmath>
#include <iostream>
#include "timecorr_demo.cu"
#include "cuda_init.h"
__global__ void create_input_test(INDEX dim, INDEX dim_t, FFTW_REAL _t[], FFTW_REAL _yr[], FFTW_REAL _yi[])
{
	INDEX j = blockDim.x * blockIdx.x + threadIdx.x;

	if (j < dim_t)
	{
		FFTW_REAL *t, *yr, *yi;
		t = &(_t[j * dim]);
		yr = &(_yr[j * dim]);
		yi = &(_yi[j * dim]);

		for (INDEX i = 0; i < dim; ++i)
		{
			t[i] = (FFTW_REAL)(i);
			yr[i] = t[i] * t[i] * std::exp(-0.1 * t[i]) * std::cos(0.5 * t[i]) / 100;
			yi[i] = t[i] * t[i] * std::exp(-0.05 * (t[i] - dim / 4.) * (t[i] - dim / 4.))
				* std::sin(0.3 * t[i]) / 100;
		}

	}
}


__global__ void init_cufft_var(INDEX dim, CUFFT_COMPLEX yinputg[], CUFFT_COMPLEX yinputg2[], FFTW_REAL yi[], FFTW_REAL yr[])
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < dim)
	{
		
		yinputg2[i].x = yinputg[i].x = yr[i];
		yinputg2[i].y = yinputg[i].y = yi[i];

	}
}

__global__ void reg_calculation(INDEX dim, INDEX dim_t, CUFFT_COMPLEX _yinput[], FFTW_REAL out_r[])
{
	INDEX kk = blockDim.x * blockIdx.x + threadIdx.x;

	if (kk < dim_t)
	{
		INDEX k, i,j;
		FFTW_REAL coef1, coef2;
		FFTW_REAL dif_re, dif_im;
		
		CUFFT_COMPLEX * yinput;
		yinput = &(_yinput[kk * dim]);

		
		for (j = 0; j < dim; ++j)
		{
			out_r[j+kk * dim] = 0.0;
			k = 0;
			for (i = j; i < dim; ++i, ++k)
			{
				dif_re = yinput[i - j].x - yinput[i].x;
				dif_im = yinput[i - j].y - yinput[i].y;
				// in place average
				coef2 = (FFTW_REAL)(1.0) / (FFTW_REAL)(k + 1);
				coef1 = (FFTW_REAL)(k)*coef2;
				out_r[j + kk * dim] = coef1 * out_r[kk * dim+j] + coef2 * (dif_re * dif_re + dif_im * dif_im);
			}
		}

		
	}
}

//__global__ void averagesabs2_array_gpu(INDEX dim, INDEX dim_t, CUFFT_COMPLEX* _in, FFTW_REAL* out)
//{
//	INDEX j = blockDim.x * blockIdx.x + threadIdx.x;
//	
//	if (j < dim_t)
//	{
//		CUFFT_COMPLEX* in;
//		in = &(_in[j * dim]);
//		
//		FFTW_REAL avg = 0.0;
//		FFTW_REAL coef1, coef2, abs2_fromstart, abs2_fromend;
//		INDEX i, ii;
//
//		for (i = 0; i < dim; ++i)
//		{
//			// next absolute value from the beginning of the array
//			abs2_fromstart = in[i].x * in[i].x + in[i].y * in[i].y;
//
//			// next absolute value from the end of the array
//			ii = dim - 1 - i;
//			abs2_fromend = in[ii].x * in[ii].x + in[ii].y * in[ii].y;
//
//			// in-place average
//			coef2 = (FFTW_REAL)(1.0) / (FFTW_REAL)(i + 1);
//			coef1 = (FFTW_REAL)(i)*coef2;
//			avg = coef1 * avg + coef2 * (abs2_fromstart + abs2_fromend);
//
//			// save the result in the output array
//			// ATTENTION! note the index
//			out[ii+ j*dim] = avg;
//		}
//	}
//}

int sdsdf_main(int argc, char* argv[])
{
	INDEX i, j, k;
	INDEX dim, dimp;
	INDEX dim_tseries;
	FFTW_REAL* t;
	FFTW_REAL* yr, * yi;
	FFTW_REAL* out_r, * out_f, * out_f2,*out_r_temp, *out_f2_g_temp;
	FFTW_REAL coef1, coef2;
	FFTW_REAL abs2_fromstart, abs2_fromend;
	FFTW_REAL dif_re, dif_im;
	FFTW_COMPLEX* y, * y2,* yinput, * yinput2;
	CUFFT_COMPLEX* yinputg, * yinputg2;
	CUFFT_COMPLEX* yinputc, * yinputc2;
	CUFFT_COMPLEX* yg, * yg2;
	int rankdim[1];

	fftw_plan plan_direct, plan_direct2;

	//----------------------------------------------
	// some input
	dim = 90;
	dim_tseries=32;
	t = new FFTW_REAL[dim_tseries * dim];
	yr = new FFTW_REAL[dim_tseries * dim];
	yi = new FFTW_REAL[dim_tseries * dim];

	FFTW_REAL* t_g, *yr_g, *yi_g;

	FFTW_REAL* out_r_g;
	
	cuda_init(true);
	cuda_exec mycuda_dim;
	cuda_exec mycuda_dim_t;
	cuda_exec mycuda_dim_dim_t;

	calc_cuda_exec(dim, deviceProp.maxThreadsPerBlock, &mycuda_dim);
	calc_cuda_exec(dim_tseries, deviceProp.maxThreadsPerBlock, &mycuda_dim_t);
	calc_cuda_exec(dim_tseries * dim, deviceProp.maxThreadsPerBlock, &mycuda_dim_dim_t);


	std::cout << cudaMalloc(&out_r_g, dim_tseries*dim * sizeof(FFTW_REAL)) << std::endl;

	cudaMalloc(&t_g, dim_tseries * dim * sizeof(FFTW_REAL));
	cudaMalloc(&yr_g, dim_tseries * dim * sizeof(FFTW_REAL));
	cudaMalloc(&yi_g, dim_tseries * dim * sizeof(FFTW_REAL));

	
	create_input_test << < mycuda_dim_t.nbk, mycuda_dim_t.nth >> > (dim, dim_tseries,t_g,yr_g,yi_g);


	cudaDeviceSynchronize();

	std::cout << cudaMemcpy(t, t_g, dim * sizeof(FFTW_REAL), cudaMemcpyDeviceToHost) << std::endl;
	cudaMemcpy(yr, yr_g, dim_tseries * dim * sizeof(FFTW_REAL), cudaMemcpyDeviceToHost);
	cudaMemcpy(yi, yi_g, dim_tseries * dim * sizeof(FFTW_REAL), cudaMemcpyDeviceToHost);

	
	
	cudaMalloc(&yinputg, dim_tseries * dim * sizeof(CUFFT_COMPLEX));
	cudaMalloc(&yinputg2, dim_tseries * dim * sizeof(CUFFT_COMPLEX));

	
	
	
	init_cufft_var << <mycuda_dim_dim_t.nbk, mycuda_dim_dim_t.nth >> > (dim_tseries * dim, yinputg, yinputg2, yi_g, yr_g);

	

	cudaDeviceSynchronize();
	//cudaError_t err = cudaGetLastError();
	//if (err != cudaSuccess)
	//	printf("Error: %s\n", cudaGetErrorString(err));

	yinput = (FFTW_COMPLEX*)(fftw_malloc(sizeof(FFTW_COMPLEX) * dim * dim_tseries));
	yinput2 = (FFTW_COMPLEX*)(fftw_malloc(sizeof(FFTW_COMPLEX) * dim * dim_tseries));

	yinputc = (CUFFT_COMPLEX*)(fftw_malloc(sizeof(CUFFT_COMPLEX) * dim));
	yinputc2 = (CUFFT_COMPLEX*)(fftw_malloc(sizeof(CUFFT_COMPLEX) * dim));

	std::cout << cudaMemcpy(yinput, yinputg, dim * dim_tseries * sizeof(FFTW_COMPLEX), cudaMemcpyDeviceToHost) << std::endl;
	std::cout << cudaMemcpy(yinput2, yinputg2, dim * dim_tseries * sizeof(FFTW_COMPLEX), cudaMemcpyDeviceToHost) << std::endl;


	//----------------------------------------------
	//----------------------------------------------
	// REGULAR CALCULATION
	out_r_temp = new FFTW_REAL[dim * dim_tseries];
	out_r = new FFTW_REAL[dim * dim_tseries];

	for (INDEX kk = 0; kk <  dim_tseries; ++kk)
	{
		for (j = 0; j < dim; ++j)
		{
			out_r[j+ kk * dim] = 0.0;
			k = 0;
			for (i = j; i < dim; ++i, ++k)
			{
				dif_re = yinput[(i - j)+ kk * dim][0] - yinput[i+ kk * dim][0];
				dif_im = yinput[(i - j)+ kk * dim][1] - yinput[i+ kk * dim][1];
				// in place average
				coef2 = (FFTW_REAL)(1.0) / (FFTW_REAL)(k + 1);
				coef1 = (FFTW_REAL)(k)*coef2;
				out_r[j + kk * dim] = coef1 * out_r[kk * dim + j] + coef2 * (dif_re * dif_re + dif_im * dif_im);
			}
		}
	}



	reg_calculation<<<mycuda_dim_t.nbk, mycuda_dim_t.nth >>>(dim, dim_tseries, yinputg, out_r_g);
	
	cudaDeviceSynchronize();
	std::cout << cudaMemcpy(out_r_temp, out_r_g, dim * dim_tseries * sizeof(FFTW_REAL), cudaMemcpyDeviceToHost) << std::endl;


	/*FFTW_REAL diff_error{};
	for (INDEX kk = 0; kk < dim * dim_tseries; ++kk)
	{
		diff_error += out_r_temp[kk] - out_r[kk];
		std::cout << kk <<"GPU is : " <<out_r_temp[kk] << "CPU is : " << out_r[kk] << std::endl;
	}
	std::cout << "Error is :  " << diff_error << std::endl;*/


	//----------------------------------------------
	//----------------------------------------------
	// FFT based calculation

	//----------------------------------------------
	// Calculating the average of the absolute squares
	out_f = new FFTW_REAL[dim * dim_tseries];
	out_f2 = new FFTW_REAL[dim * dim_tseries];
	FFTW_REAL* out_f_g, *out_f2_g;

	std::cout << cudaMalloc(&out_f_g, dim * dim_tseries * sizeof(FFTW_REAL)) << std::endl;
	std::cout << cudaMalloc(&out_f2_g, dim * dim_tseries * sizeof(FFTW_REAL)) << std::endl;

	averagesabs2_array_gpu<<<mycuda_dim_t.nbk, mycuda_dim_t.nth >>>(dim,dim_tseries, yinputg, out_f_g);

	averagesabs2_array_cpu  (dim, dim_tseries, yinput, out_f);

	cudaDeviceSynchronize();
	std::cout << cudaMemcpy(out_f2, out_f_g, dim * dim_tseries * sizeof(FFTW_REAL), cudaMemcpyDeviceToHost) << std::endl;



	//FFTW_REAL diff_error{};
	//for (INDEX kk = 0; kk < dim * dim_tseries; ++kk)
	//{
	//	diff_error += out_f2[kk] - out_f[kk];
	//	std::cout << kk <<"GPU is : " << out_f2[kk] << "CPU is : " << out_f[kk] << std::endl;
	//}
	//std::cout << "Error is :  " << diff_error << std::endl;


	//----------------------------------------------
	// preparing Fourier variables

	// Calculating the closes power of 2 that allows for zero-padding
	dimp = (INDEX)(std::ceil(std::log((FFTW_REAL)(dim)) / std::log(2.0))) + 1;
	rankdim[0] = 1;
	for (i = 0; i < dimp; ++i) rankdim[0] *= 2;
	dimp = rankdim[0];

	// allocating memory
	y = (FFTW_COMPLEX*)(fftw_malloc(sizeof(FFTW_COMPLEX) * dimp));
	y2 = (FFTW_COMPLEX*)(fftw_malloc(sizeof(FFTW_COMPLEX) * dimp));

	cudaMalloc(&yg, dim_tseries * dimp * sizeof(CUFFT_COMPLEX));
	cudaMalloc(&yg2, dim_tseries * dimp* sizeof(CUFFT_COMPLEX));

	// preparing plan
	fftw_init_threads();
	fftw_plan_with_nthreads(2);
	plan_direct = fftw_plan_dft(1, rankdim, y, y, FFTW_FORWARD, FFTW_ESTIMATE);
	plan_direct2 = fftw_plan_dft(1, rankdim, y2, y2, FFTW_FORWARD, FFTW_ESTIMATE);

	int alloc_status_plan;
	cufftHandle plan;

	//#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
	//	alloc_status_plan = cufftPlan1d(&plan, dimp, CUFFT_Z2Z);
	//	//cufftExecC2C(plan, dev_fft, dev_fft, CUFFT_FORWARD);
	//#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
	//	alloc_status_plan = cufftPlan1d(&plan, dimp, CUFFT_Z2Z,1);
	//	//cufftExecZ2Z(plan, dev_fft, dev_fft, CUFFT_FORWARD);
	//#else
	//#error Unknown CUDA type selected
	//#endif

	int n[1] = { dimp };

	cufftResult res = cufftPlanMany(&plan, 1, n,
		NULL, 1, dimp,  //advanced data layout, NULL shuts it off
		NULL, 1, dimp,  //advanced data layout, NULL shuts it off
		CUFFT_Z2Z, dim_tseries);
	if (res != CUFFT_SUCCESS) { printf("plan create fail\n"); return 1; }


	timeseriesanalysis_cpu(dim,dim_tseries, yinput2, dimp, y2, &plan_direct2, out_f2);

	timeseriesanalysis_gpu(dim, dim_tseries, yinputg2, dimp, yg2, &plan, out_f2_g, mycuda_dim_t, mycuda_dim, mycuda_dim_dim_t);
	cudaDeviceSynchronize();

	out_f2_g_temp = new FFTW_REAL[dim * dim_tseries];

	std::cout << cudaMemcpy(out_f2_g_temp, out_f2_g, dim * dim_tseries * sizeof(FFTW_REAL), cudaMemcpyDeviceToHost) << std::endl;

//	FFTW_REAL diff_error{};
//for (INDEX kk = 0; kk < dim * dim_tseries; ++kk)
//{
//	diff_error += out_f2_g_temp[kk] - out_f2[kk];
//	std::cout << kk <<"GPU is : " << out_f2_g_temp[kk] << "CPU is : " << out_f2[kk] << std::endl;
//}
//std::cout << "Error is :  " << diff_error << std::endl;


	//-----------------------------------------------
	// copying data to the execution memory with normalization
	for (i = 0; i < dimp; ++i)
	{
		y[i][0] = 0.0;
		y[i][1] = 0.0;
	}
	gain_complex_cpu(4, dim, yinput, (FFTW_REAL)(1. / std::sqrt((FFTW_REAL)(dimp))), y);
	//----------------------------------------------

	// FFT execution
	fftw_execute(plan_direct);
	// evaluating abs^2
	complex_abs2_cpu(4, dimp, y, y);
	// FFT execution. Given the conditions
	// - we are only interested in the real part
	// - we start from a real function generated by the absolute value
	// Then the direct and inverse FFT are equivalent. We can re-use plan_direct!
	fftw_execute(plan_direct);
	update_with_divrebyramp_cpu(4, dim, y, out_f);

	//------------------------------------------------
	//------------------------------------------------
	// DISPLAY
	window_display_opencv* fig;
	scatter_opencv* sc;
	sc = new scatter_opencv;
	sc->dimpoint = dim;
	sc->x = t;
	sc->y = yr;
	fig = plot(-1, sc);
	sc->color_normal = cv::Scalar(0, 0, 255);
	sc = new scatter_opencv;
	sc->dimpoint = dim;
	sc->x = t;
	sc->y = yi;
	sc->set_marker_type(MARKER_CIRCLE_OPENCV);
	sc->color_normal = cv::Scalar(255, 100, 100);
	fig = plot(fig->id, sc);
	sc = new scatter_opencv;
	sc->dimpoint = dim;
	sc->x = t;
	sc->y = out_r;
	fig = plot(-1, sc);
	sc->set_marker_type(MARKER_CIRCLE_OPENCV);
	sc->color_normal = cv::Scalar(0, 0, 255);
	sc = new scatter_opencv;
	sc->dimpoint = dim;
	sc->x = t;
	sc->y = out_f;
	fig = plot(fig->id, sc);
	for (i = 0; i < dim_tseries; i++)
	{
		sc = new scatter_opencv;
		sc->dimpoint = dim;
		sc->x = t;
		sc->y = &out_f2_g_temp[i*dim];
		sc->set_marker_type(MARKER_NONE_OPENCV);
		sc->color_normal = cv::Scalar(0, 128, 255);
		fig = plot(fig->id, sc);

		waitkeyboard(0);
	}

	//------------------------------------------------
	delete[] t, yr, yi;
	delete[] out_r, out_f, out_f2,out_r_temp, out_f2_g_temp;
	fftw_free(y);
	fftw_free(y2);
	fftw_free(yinput);
	fftw_free(yinput2);
	fftw_destroy_plan(plan_direct);
	fftw_destroy_plan(plan_direct2);
	fftw_cleanup_threads();

	cudaFree(t_g);
	cudaFree(yr_g);
	cudaFree(yi_g);
	cudaFree(out_r_g);
	cudaFree(out_f_g);
	cudaFree(out_f2_g);
	cudaFree(yinputg);
	cudaFree(yinputg2);
	cudaFree(yg);
	cudaFree(yg2);

	cuda_end();

	return 0;
}


