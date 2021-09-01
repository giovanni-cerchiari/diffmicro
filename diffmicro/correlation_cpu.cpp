/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 08/2011
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
These functions are written for diffmicro.exe application.
*/

#include "stdafx.h"
#include "correlation.h"
#include <thread>

unsigned short* dev_im_cpu(NULL);
unsigned int* dev_radial_lut_cpu(NULL);
FFTW_COMPLEX *dev_fft_cpu(NULL);
FFTW_COMPLEX* dev_fft_time_cpu(NULL);
FFTW_REAL* dev_corr_cpu(NULL);
FFTW_COMPLEX *dev_images_cpu(NULL);
FFTW_COMPLEX* dev_image_sot_cpu(NULL);
STORE_REAL *dev_power_spectra_cpu(NULL);
FFTW_REAL one_over_fft_norm_cpu(0);

#if (FFTW_TYPE == FFTW_TYPE_FLOAT)
	fftwf_plan plan;
#elif (FFTW_TYPE == FFTW_TYPE_DOUBLE)
	fftw_plan plan;
#else
	#error Unknown FFTW type selected
#endif


#if (FFTW_TYPE == FFTW_TYPE_FLOAT)
fftwf_plan *plantime(NULL);
#elif (FFTW_TYPE == FFTW_TYPE_DOUBLE)
fftw_plan *plantime(NULL);
#else
#error Unknown FFTW type selected
#endif

void shorttorealwithgain_cpu(INDEX dim, unsigned short vets[], FFTW_REAL gain, FFTW_COMPLEX vetc[])
{
	for(INDEX i=0; i<dim; ++i)
		{
			vetc[i][0] = gain * (FFTW_REAL)(vets[i]);
			vetc[i][1] = 0.;
		}
}

void short_to_real_with_gain_cpu(INDEX nth, INDEX dim, unsigned short vets[], FFTW_REAL gain, FFTW_COMPLEX vetc[])
{
	INDEX i;
	lldiv_t d;
	std::thread** th;
	th = new std::thread * [nth];

	d = div((long long)dim, nth);

	// starting all threads
	for (i = 0; i < nth; ++i)
		th[i] = new std::thread(shorttorealwithgain_cpu, d.quot, &(vets[i * d.quot]), gain, &(vetc[i * d.quot]));
	if (0 < d.rem)
		shorttorealwithgain_cpu(d.rem, & (vets[i * d.quot]), gain, & (vetc[i * d.quot]));

	// waiting for all threads to finish
	for (i = 0; i < nth; ++i)
		th[i]->join();

	delete[] th;
}

void complextocomplexwithgain_cpu(INDEX dim, FFTW_COMPLEX vets[], FFTW_REAL gain, FFTW_COMPLEX vetc[])
{
	for (INDEX i = 0; i < dim; ++i)
	{
		vetc[i][0] = gain * (FFTW_REAL)(vets[i][0]);
		vetc[i][1] = gain * (FFTW_REAL)(vets[i][1]);
	}
}

void complex_to_complex_with_gain_cpu(INDEX nth, INDEX dim, FFTW_COMPLEX vets[], FFTW_REAL gain, FFTW_COMPLEX vetc[])
{
	INDEX i;
	lldiv_t d;
	std::thread** th;
	th = new std::thread * [nth];

	d = div((long long)dim, nth);

	// starting all threads
	for (i = 0; i < nth; ++i)
		th[i] = new std::thread(complextocomplexwithgain_cpu, d.quot, &(vets[i * d.quot]), gain, &(vetc[i * d.quot]));
	if (0 < d.rem)
		complextocomplexwithgain_cpu(d.rem, &(vets[i * d.quot]), gain, &(vetc[i * d.quot]));

	// waiting for all threads to finish
	for (i = 0; i < nth; ++i)
		th[i]->join();

	delete[] th;
}

void complextorealwithgain_cpu(INDEX dim, FFTW_COMPLEX vets[], FFTW_REAL gain, FFTW_REAL vetc[])
{
	for (INDEX i = 0; i < dim; ++i)
	{
		vetc[i] = gain * vets[i][0];
	}
}

void complex_to_real_with_gain_cpu(INDEX nth, INDEX dim, FFTW_COMPLEX vets[], FFTW_REAL gain, FFTW_REAL vetc[])
{
	INDEX i;
	lldiv_t d;
	std::thread** th;
	th = new std::thread * [nth];

	d = div((long long)dim, nth);

	// starting all threads
	for (i = 0; i < nth; ++i)
		th[i] = new std::thread(complextorealwithgain_cpu, d.quot, &(vets[i * d.quot]), gain, &(vetc[i * d.quot]));
	if (0 < d.rem)
		complextorealwithgain_cpu(d.rem, &(vets[i * d.quot]), gain, &(vetc[i * d.quot]));

	// waiting for all threads to finish
	for (i = 0; i < nth; ++i)
		th[i]->join();

	delete[] th;
}



void cpx_row2col_gain_cpu(INDEX dim, INDEX dimx_in, INDEX i_row_in, FFTW_COMPLEX in[], FFTW_REAL gain, INDEX dimx_out, INDEX i_col_out, FFTW_COMPLEX out[])
{
	INDEX i_in, i_out;
	for (INDEX i = 0; i < dim; ++i)
	{
		i_in = i_row_in * dimx_in + i;
		i_out = i * dimx_out + i_col_out;

		out[i_out][0] = gain * in[i_in][0];
		out[i_out][1] = gain * in[i_in][1];
	}
}

void cpx_col2row_gain_cpu(INDEX dimcopy, INDEX dimx_in, INDEX i_col_in, FFTW_COMPLEX in[], FFTW_REAL gain, INDEX dimx_out, INDEX i_row_out, FFTW_COMPLEX out[])
{
	INDEX i_in, i_out;
	for (INDEX i = 0; i < dimcopy; ++i)
	{
		i_in = i * dimx_in + i_col_in;
		i_out = i_row_out * dimx_out + i;

		out[i_out][0] = gain * in[i_in][0];
		out[i_out][1] = gain * in[i_in][1];
	}
}

void lutfft_to_timeseries_cpu(INDEX dimcopy, FFTW_REAL gain, INDEX t, INDEX starting_freq)
{
	cpx_row2col_gain_cpu(dimcopy, (INDEX)(1), starting_freq, dev_image_sot_cpu, gain,
		                   s_time_series.dim, t, dev_images_cpu);
}

void timeseries_to_lutfft_cpu(INDEX dimcopy, FFTW_REAL gain, INDEX t, INDEX starting_freq)
{
	cpx_col2row_gain_cpu(dimcopy, s_time_series.dim, t, dev_images_cpu, gain,
		(INDEX)(1), starting_freq, dev_image_sot_cpu);
}

void timeseries_to_lutpw_cpu(INDEX dimcopy, FFTW_REAL gain, INDEX t, INDEX starting_freq, STORE_REAL *ram_power_spectra)
{
	timeseries_to_lutfft_cpu(dimcopy, gain, t, starting_freq);
	//complex_to_real_with_gain_cpu(useri.nthread, dimcopy, dev_image_sot_cpu,
	//		(FFTW_REAL)(1.0), dev_power_spectra_cpu);
	complex_to_real_with_gain_cpu(useri.nthread, dimcopy, dev_image_sot_cpu,
		(FFTW_REAL)(1.0), ram_power_spectra);
}

void timeseriesanalysis_cpu(INDEX dimtimeseries, FFTW_COMPLEX *tseries, INDEX dimfft, FFTW_COMPLEX *fft_memory, fftw_plan *tplan, FFTW_REAL *corr_memory)
{
	INDEX i;
	//----------------------------------------------
	// Calculating the average of the absolute squares
	averagesabs2_array_cpu(dimtimeseries, tseries, corr_memory);

	//----------------------------------------------
	// preparing Fourier variables

	//-----------------------------------------------
	// zeroing all fft memory for zero padding and copying data with normalization to fft memory
	for (i = 0; i < dimfft; ++i)
	{
		fft_memory[i][0] = 0.0;
		fft_memory[i][1] = 0.0;
	}
	gaincomplex_cpu(dimtimeseries, tseries, (FFTW_REAL)(1. / std::sqrt((FFTW_REAL)(dimfft))), fft_memory);
	//----------------------------------------------

	// FFT execution
	fftw_execute(tplan[0]);
	// evaluating abs^2
	complexabs2_cpu(dimfft, fft_memory, fft_memory);
	// FFT execution. Given the conditions
	// - we are only interested in the real part
	// - we start from a real function generated by the absolute value
	// Then the direct and inverse FFT are equivalent. We can re-use plan_direct!
	fftw_execute(tplan[0]);
	updatewithdivrebyramp_cpu(dimtimeseries, dimtimeseries, fft_memory, corr_memory);
	// copy the result  back to original memory area
	for (i = 0; i < dimtimeseries; ++i)
	{
		tseries[i][0] = corr_memory[i];
		tseries[i][1] = 0.0;
	}
}

void timeseriesanalysis_cpu(INDEX dimtimeseries, INDEX dim_t, FFTW_COMPLEX* tseries, INDEX dimfft, FFTW_COMPLEX* fft_memory, fftw_plan* tplan, FFTW_REAL* corr_memory)
{
	INDEX i;
	//----------------------------------------------
	// Calculating the average of the absolute squares
	averagesabs2_array_cpu(dimtimeseries,dim_t, tseries, corr_memory);


	for (INDEX j = 0; j < dim_t; ++j)
	{
		//----------------------------------------------
		// preparing Fourier variables

		//-----------------------------------------------
		// zeroing all fft memory for zero padding and copying data with normalization to fft memory
		for (i = 0; i < dimfft; ++i)
		{
			fft_memory[i][0] = 0.0;
			fft_memory[i][1] = 0.0;
		}
		gaincomplex_cpu(dimtimeseries, tseries, (FFTW_REAL)(1. / std::sqrt((FFTW_REAL)(dimfft))), fft_memory);
		//----------------------------------------------

		// FFT execution
		fftw_execute(tplan[0]);
		// evaluating abs^2
		complexabs2_cpu(dimfft, fft_memory, fft_memory);
		// FFT execution. Given the conditions
		// - we are only interested in the real part
		// - we start from a real function generated by the absolute value
		// Then the direct and inverse FFT are equivalent. We can re-use plan_direct!
		fftw_execute(tplan[0]);
		updatewithdivrebyramp_cpu(dimtimeseries, dimtimeseries, fft_memory, corr_memory);
		// copy the result  back to original memory area
		for (i = 0; i < dimtimeseries; ++i)
		{
			tseries[i][0] = corr_memory[i];
			tseries[i][1] = 0.0;
		}
	}
}

void timeseriesanalysisgroup_cpu(INDEX dimgroup, INDEX dimtimeseries, FFTW_COMPLEX* tseries, INDEX dimfft, FFTW_COMPLEX* fft_memory, fftw_plan *tplan, FFTW_REAL* corr_memory)
{
	INDEX i;
	for (i = 0; i < dimgroup; ++i)
		timeseriesanalysis_cpu(dimtimeseries, &(tseries[i*dimtimeseries]), dimfft, fft_memory, tplan, corr_memory);
}

void timeseries_analysis_cpu(INDEX nth)
{
	INDEX i;
	lldiv_t d;
	std::thread** th;
	th = new std::thread * [nth];

	d = div((long long)(s_time_series.numerosity), nth);

	// starting all threads
	// note that of memory areas dev_fft_time_cpu and dev_corr_cpu there are only as many as threads to be started!
	// the memory jump should NOT be multiplied by d.quot
	for (i = 0; i < nth; ++i)
		th[i] = new std::thread(timeseriesanalysisgroup_cpu, d.quot, s_time_series.dim, &(dev_images_cpu[i * d.quot * s_time_series.dim]), s_fft_time.dim,
			      &(dev_fft_time_cpu[i * s_fft_time.dim]), &(plantime[i]), &(dev_corr_cpu[i * s_time_series.dim]) );
	// waiting for all threads to finish
	for (i = 0; i < nth; ++i)
		th[i]->join();

	// doing the remanence.
	// This must be done after the join to use the frit existing fftw plan and memory areas
	if (0 < d.rem)
		timeseriesanalysisgroup_cpu(d.rem, s_time_series.dim, &(dev_images_cpu[i * d.quot * s_time_series.dim]), s_fft_time.dim,
						dev_fft_time_cpu, plantime, dev_corr_cpu );

	delete[] th;
}



/*! WARNING: Do not use this function in multithreading!!!*/
void gaincomplexlut_cpu(INDEX dim, unsigned int* lut, FFTW_COMPLEX in[], FFTW_REAL gain, FFTW_COMPLEX out[])
{
	for (INDEX i = 0; i < dim; ++i)
	{
		out[i][0] = gain * in[lut[i]][0];
		out[i][1] = gain * in[lut[i]][1];
	}
}


//	cThread *th_diffpw ;
//	diffpwtavg_arg<STORE_REAL, FFT_COMPLEX> *args_diffpw;

//--------------------------------------------------------------------
// vector addition

void diffpwtavg_cpu(INDEX dim, FFTW_COMPLEX min[], FFTW_COMPLEX sot[], FFTW_REAL coef1, FFTW_REAL coef2, FFTW_REAL pw[])
{
	FFTW_REAL difx, dify;
	for(INDEX i=0; i<dim; ++i)
		{
			difx = min[i][0] - sot[i][0];
			dify = min[i][1] - sot[i][1];
			pw[i] = coef1 * pw[i] + coef2 * (difx * difx + dify * dify);
		}
}


void diff_power_spectrum_to_avg_cpu(INDEX nth, INDEX dim, FFTW_COMPLEX min[], FFTW_COMPLEX sot[], FFTW_REAL coef1, FFTW_REAL coef2, FFTW_REAL pw[])
{
	INDEX i;
	lldiv_t d;
	std::thread** th;
	th = new std::thread * [nth];

	d = div((long long)dim, nth);

	// starting all threads
	for (i = 0; i < nth; ++i)
		th[i] = new std::thread(diffpwtavg_cpu, d.quot, &(min[i * d.quot]), &(sot[i * d.quot]), coef1, coef2, &(pw[i * d.quot]));
	if (0 < d.rem)
		diffpwtavg_cpu(d.rem, & (min[i * d.quot]), & (sot[i * d.quot]), coef1, coef2, & (pw[i * d.quot]));

	// waiting for all threads to finish
	for (i = 0; i < nth; ++i)
		th[i]->join();

	delete[] th;
}


int cpu_allocation(int size_freq,int flg_mode, INDEX& nimages, INDEX& dimy, INDEX& dimx, INDEX& dim_power_spectrum, unsigned int* ram_radial_lut)
{
//	int alloc_status_li, alloc_status_fft, alloc_status_pw, alloc_status_im, alloc_status_plan;
	INDEX i,free_video_memory, image_p_spectrum_memory;
	INDEX capacity;
	INDEX dimtimeseries_exponent, dimtimeseries_zeropadding;
//	double capacity_d;

	// unitary size to make allocation
	calc_sizes(dimy, dimx, 1, sizeof(unsigned short), s_load_image);
	calc_sizes(1, dim_power_spectrum, 1, sizeof(unsigned int), s_radial_lut);
	calc_sizes(dimy, dimx, 1, sizeof(FFTW_COMPLEX), s_fft);
	calc_sizes(1, dim_power_spectrum, 1, sizeof(FFTW_COMPLEX), s_fft_images);
	calc_sizes(1, dim_power_spectrum, 1, sizeof(STORE_REAL), s_power_spectra);
	calc_sizes(1, nimages, 1, sizeof(FFTW_COMPLEX), s_time_series);
	// Calculating the closes power of 2 that allows for zero-padding
	dimtimeseries_exponent = (INDEX)(std::ceil(std::log((FFTW_REAL)(nimages)) / std::log(2.0))) + 1;
	dimtimeseries_zeropadding = 1;
	for (i = 0; i < dimtimeseries_exponent; ++i) dimtimeseries_zeropadding *= 2;
	calc_sizes(1, dimtimeseries_zeropadding, 1, sizeof(FFTW_COMPLEX), s_fft_time);

	// extimating video card capability of storing the same number of power spectra and
	// FFT of different images
	if ((INDEX)(useri.RAM) <= (s_load_image.memory_tot + s_fft.memory_tot + s_fft_images.memory_one))
	{
		std::cerr << "not enough dedicated RAM memory for this task" << std::endl;
		return 2;
	}
	free_video_memory = (INDEX)(useri.RAM) - (s_load_image.memory_tot + s_fft.memory_tot + s_fft_images.memory_one);
	switch (useri.execution_mode)
	{
	case DIFFMICRO_MODE_FIFO:
		image_p_spectrum_memory = s_power_spectra.memory_one + s_fft_images.memory_one;
		capacity = free_video_memory / image_p_spectrum_memory;
		if (capacity > nimages) capacity = nimages;
		break;
	case DIFFMICRO_MODE_TIMECORRELATION:
		free_video_memory -= s_fft_time.memory_tot * useri.nthread;
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
			std::cerr <<"not enough dedicated RAM memory for this task"<<std::endl;
			return 2;
		}
	
		
	one_over_fft_norm_cpu = (FFTW_REAL)(1./(sqrt((FFTW_REAL)(dimx * dimy))));
	cpu_free_pointers();

	//----------------------------------------------------------------------------
	// ALLOCATION
	
	// trial and error allocation

	//----------------------------------------------------------
	// FFT initialization
	int rankdim[2];
	rankdim[0] = (int)(dimy); rankdim[1] = (int)(dimx); 
	// this one must  be allocated like this
	dev_fft_cpu = (FFTW_COMPLEX*)(fftw_malloc(s_fft.memory_tot ));
	
	fftw_init_threads();
	fftw_plan_with_nthreads(useri.nthread);
	#if (FFTW_TYPE == FFTW_TYPE_FLOAT)
	 plan = fftwf_plan_dft(2, rankdim, dev_fft_cpu, dev_fft_cpu, FFTW_FORWARD, FFTW_PATIENT);
	#elif (FFTW_TYPE == FFTW_TYPE_DOUBLE)
	 //plan = fftw_plan_dft(2, rankdim, dev_fft_cpu, dev_fft_cpu, FFTW_FORWARD, FFTW_PATIENT);
	 plan = fftw_plan_dft(2, rankdim, dev_fft_cpu, dev_fft_cpu, FFTW_FORWARD, FFTW_ESTIMATE);
	#else
		#error Unknown FFTW type selected
	#endif
	
	dev_radial_lut_cpu = new unsigned int[s_radial_lut.dim];

	dev_image_sot_cpu = new FFTW_COMPLEX[s_fft_images.dim];
	switch (useri.execution_mode)
	{
	case DIFFMICRO_MODE_FIFO:
		dev_images_cpu = new FFTW_COMPLEX[s_fft_images.dim * capacity];
		dev_power_spectra_cpu = new FFTW_REAL[s_power_spectra.dim * capacity];
		calc_sizes(1, dim_power_spectrum, capacity, sizeof(CUFFT_COMPLEX), s_fft_images);
		tot_memory_fft_images = s_fft_images.memory_tot;
		calc_sizes(1, dim_power_spectrum, capacity, sizeof(STORE_REAL), s_power_spectra);
		tot_calculation_memory = s_power_spectra.memory_tot;
		break;
	case DIFFMICRO_MODE_TIMECORRELATION:
		dev_images_cpu = new FFTW_COMPLEX[s_time_series.dim * capacity];
		dev_fft_time_cpu = (FFTW_COMPLEX*)(fftw_malloc(s_fft_time.memory_tot * useri.nthread));
		dev_corr_cpu = new FFTW_REAL[s_time_series.dim * useri.nthread];
		dev_power_spectra_cpu = new FFTW_REAL[s_power_spectra.dim];
		#if (FFTW_TYPE == FFTW_TYPE_FLOAT)
				plantime = new fftwf_plan[useri.nthread];
		#elif (FFTW_TYPE == FFTW_TYPE_DOUBLE)
				plantime = new fftw_plan[useri.nthread];
		#else
		#error Unknown FFTW type selected
		#endif
		rankdim[0] = s_fft_time.dim;
		fftw_plan_with_nthreads(1);
		for (i = 0; i < useri.nthread; ++i)
		{
		#if (FFTW_TYPE == FFTW_TYPE_FLOAT)
				plantime[i] = fftwf_plan_dft(1, rankdim, dev_fft_cpu, dev_fft_cpu, FFTW_FORWARD, FFTW_PATIENT);
		#elif (FFTW_TYPE == FFTW_TYPE_DOUBLE)
				//plan = fftw_plan_dft(1, rankdim, dev_fft_cpu, dev_fft_cpu, FFTW_FORWARD, FFTW_PATIENT);
				plantime[i] = fftw_plan_dft(1, rankdim,
					&(dev_fft_time_cpu[i * s_fft_time.dim]), &(dev_fft_time_cpu[i * s_fft_time.dim]),
					FFTW_FORWARD, FFTW_ESTIMATE);
		#else
		#error Unknown FFTW type selected
		#endif
		}
		calc_sizes(1, nimages, capacity, sizeof(FFTW_COMPLEX), s_time_series);
		tot_memory_fft_images = s_time_series.memory_tot;

		calc_sizes(1, dimtimeseries_zeropadding, useri.nthread, sizeof(CUFFT_COMPLEX), s_fft_time);
		tot_calculation_memory = s_fft_time.memory_tot;

		break;
	default:
		break;
	}


	// real size

	

	for (i = 0; i < s_radial_lut.dim; ++i)
		dev_radial_lut_cpu[i] = ram_radial_lut[i];

	n_capacity = capacity;

	if(capacity == 0)	return 1;
	return 0;
}

void cpu_free_pointers()
{
	if (NULL != dev_im_cpu)
	{
		delete[] dev_im_cpu;
		dev_im_cpu = NULL;
	}
	if(NULL != dev_fft_cpu)
	{
		fftw_free(dev_fft_cpu);
		dev_fft_cpu = NULL;
	}
	if (NULL != dev_fft_time_cpu)
	{
		fftw_free(dev_fft_time_cpu);
		dev_fft_time_cpu = NULL;
	}
	if (NULL != dev_corr_cpu)
	{
		delete[] dev_corr_cpu;
		dev_corr_cpu = NULL;
	}
	if(NULL != dev_images_cpu)
	{
		delete[] dev_images_cpu;
		dev_images_cpu = NULL;
	}
	if (NULL != dev_image_sot_cpu)
	{
		delete[] dev_image_sot_cpu;
		dev_image_sot_cpu = NULL;
	}
	if (NULL != dev_power_spectra_cpu)
	{
		delete[] dev_power_spectra_cpu;
		dev_power_spectra_cpu = NULL;
	}
	if (NULL != dev_radial_lut_cpu)
	{
		delete[] dev_radial_lut_cpu;
		dev_radial_lut_cpu = NULL;
	}

}

void cpu_deallocation()
{
	cpu_free_pointers();
	#if (FFTW_TYPE == FFTW_TYPE_FLOAT)
	 	fftwf_destroy_plan(plan);
	#elif (FFTW_TYPE == FFTW_TYPE_DOUBLE)
	 	fftw_destroy_plan(plan);
	#else
		#error Unknown CUDA type selected
	#endif

	if (NULL != plantime)
	{
		for (INDEX i = 0; i < useri.nthread; ++i)
		{
			#if (FFTW_TYPE == FFTW_TYPE_FLOAT)
						fftwf_destroy_plan(plantime[i]);
			#elif (FFTW_TYPE == FFTW_TYPE_DOUBLE)
						fftw_destroy_plan(plantime[i]);
			#else
			#error Unknown CUDA type selected
			#endif
		}
		delete[] plantime;
		plantime = NULL;
	}

	fftw_cleanup_threads();
}


int image_to_dev_cpu(SIGNED_INDEX ind_fifo, MY_REAL &mean, unsigned short *im, bool flg_debug)
{
	FFTW_COMPLEX* dev_store_ptr;
	int ret = 0;
	FFTW_REAL mean_tmp;

	// selecting the correct memory area
	if (0 > ind_fifo)
		dev_store_ptr = dev_image_sot_cpu;
	else
		dev_store_ptr = &(dev_images_cpu[ind_fifo * s_fft_images.dim]);

	time_fft_norm.start();

	// from image to complex matrix
	short_to_real_with_gain_cpu(useri.nthread, s_load_image.dim, im, (FFTW_REAL)(one_over_fft_norm_cpu / 65536.), dev_fft_cpu);
	
	// FFT execution
	#if (FFTW_TYPE == FFTW_TYPE_FLOAT)
	 fftwf_execute(plan);
	#elif (FFTW_TYPE == FFTW_TYPE_DOUBLE)
	 fftw_execute(plan);
	#else
		#error Unknown FFT type selected
	#endif
	
	
	// normalization
	mean_tmp = dev_fft_cpu[0][0];

	mean = mean_tmp;
	if(mean < 0.000000000000001)
		{
			mean = 1.;
			mean_tmp = 1.;
			ret = 1;
		}
	mean_tmp =	(FFTW_REAL)(1./mean_tmp);

	//for(int i=0; i<100000; ++i)
	//gain_complex_cpu(useri.nthread, s_fft_images.dim, dev_fft_cpu, mean_tmp, dev_store_ptr);
	gaincomplexlut_cpu(s_fft_images.dim, dev_radial_lut_cpu, dev_fft_cpu, mean_tmp, dev_store_ptr);

	time_fft_norm.stop();

	++n_computed_fft;

	return ret;
}

void copy_power_spectra_from_dev_cpu(STORE_REAL* power_spectrum_r)
{
	INDEX i, dim;
	dim = s_power_spectra.dim * s_power_spectra.numerosity;
	for (i = 0; i < dim; ++i)
	{
		power_spectrum_r[i] = dev_power_spectra_cpu[i];
	}
}

void diff_power_spectrum_to_avg_cpu_cpu(FFTW_REAL coef1, FFTW_REAL coef2, INDEX j, INDEX ind_dist)
{
	diff_power_spectrum_to_avg_cpu(useri.nthread, s_power_spectra.dim, &(dev_images_cpu[j * s_fft_images.dim]), dev_image_sot_cpu,
		coef1, coef2, &(dev_power_spectra_cpu[ind_dist * s_power_spectra.dim]));
	//(s_power_spectra.dim, &(dev_images_gpu[j * s_fft_images.dim]), dev_image_sot_gpu, coef1, coef2, &(dev_power_spectra_gpu[ind_dist * s_power_spectra.dim]));
};

