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

This file contains the functions that execute the main algorithm on cpu and gpu.
Execution on the different hardware is implemented with minimal modifications on the main algorithm
by using function pointers. Function pointers selection is performed in 
*/

#ifndef _CORRELATION_H_
#define _CORRELATION_H_

#include "global_define.h"
#include "cuda_init.h"

#include <cstdlib>
#include <cufft.h>
#include <cmath>

#include "radial_storage_lut.h"
#include "diffmicro_log.h"
#include "timeavg_by_correlation.h"

const int version = 0;

//--------------------------------------------------------------------------
/*!
This structure and the function void calc_sizes(INDEX dimy, INDEX dimx, INDEX n_tot, INDEX size_of_element, sizes &s)
allow to store in a single place the values concerning the memory managment of some allocated memory.
To understand the working principle we will make the example an image. To store a series of images the following data are saved
- dimension in pixels dimx, dimy and dim = dimx*dimy
- numerosity -> the stored number of images 
- memory_one -> memory size in byte of one image
- memory_tot -> memory size in byte for the total number of images
- cexe -> CUDA #blocks and #threads necessary to browse all the pixel of a single image as a linear vector
*/
struct sizes
	{
		//! number of colums of the matrix-image
		INDEX dimx;
		//! number of rows of the matrix-image
		INDEX dimy;
		//! = dimx * dimy
		INDEX dim;
		//! number of this matrix-image type storable on the video card (memory already allocated)
		INDEX numerosity;
		//! memory size in byte of a single matrix-image
		INDEX memory_one;
		//! = memory_one * numerosity
		INDEX memory_tot;
		//! #blocks and #threads necessary to browse all the pixel of a single image as a linear vector
		cuda_exec cexe;
		//! # of pixels processable per thread on the CPU given the ammount of CPU specified in N_MAX_THREADS
	};

//! size, dimensions, and memory occupation for the unsigned short image used to load an image on the video card
extern sizes s_load_image;
//! size, dimensions, and memory occupation of the temporary memory area where the 2-D FFTs are calculated
extern sizes s_fft;
//! size, dimensions, and memory occupation of the storable Fourier transforms
extern sizes s_fft_images;
//! size, dimensions, and memory occupation of the storable power spectra
extern sizes s_power_spectra;
//! size, dimensions, and memory occupation of the radial look up table
extern sizes s_radial_lut;
//! size, dimensions, and memory occupation of the wavevectors' time series
extern sizes s_time_series;
//! size, dimensions, and memory occupation of the memory area where the temporal FFTs are calculated
extern sizes s_fft_time;

/*!
given a vector of dimension dim this function returns the CUDA #blocks and #threads necessary
to browse all the elements of the vector.
*/
void number_bk_th(INDEX dim, unsigned int n_max_th, unsigned int &nbk, unsigned int &nth);

/*!
see struct sizes
*/
void calc_sizes(INDEX dimy, INDEX dimx, INDEX n_tot, INDEX size_of_element, sizes &s);

//------------------------------------------------------------------------------------
//! device memory area where each image is loaded
extern unsigned short *dev_im_gpu;
//! device memory area where each FFT is performed for GPU execution
extern CUFFT_COMPLEX *dev_fft_gpu;
//! device memory area where each FFT is performed for CPU execution
extern FFTW_COMPLEX* dev_fft_cpu;
//! memory area where the FFTW in time are performed
extern FFTW_COMPLEX* dev_fft_time_cpu;
//! device memory area where FFTs are stored for GPU execution
extern CUFFT_COMPLEX *dev_images_gpu;
//! device memory area where FFTs are stored for CPU execution
extern FFTW_COMPLEX* dev_images_cpu;
//! device memory area where the image to be subtracted is stored for GPU execution
extern CUFFT_COMPLEX *dev_image_sot_gpu;
//! device memory area where the image to be subtracted is stored for GPU execution
extern FFTW_COMPLEX* dev_image_sot_cpu;
//! device memory area where power spectra are stored for GPU execution
extern STORE_REAL *dev_power_spectra_gpu;
//! device memory area where power spectra are stored for CPU execution
extern FFTW_REAL* dev_power_spectra_cpu;


//-----------------------------------------------------------------------------------

/*!
This function allocates the memory card and initialize all
sizes variable necessary to run the calculus for gpu execution. \n
Allocation of the memory on the video card is made with a trial and error process.
It is found that if any memory allocation error occurs it is better to reallocate the entire memory.
Note that, since the power spectrum is a symmetric matrix, we can store only the upper half
of the FFTs and perform the difference operation only over this upper half.
*/
int gpu_allocation(int flg_mode, INDEX &nimages, INDEX &dimy, INDEX &dimx, INDEX &dim_power_spectrum, unsigned int *ram_radial_lut);
/*!
This function allocates the memory card and initialize all
sizes variable necessary to run the calculus for cpu execution. \n
Allocation of the memory on the video card is made with a trial and error process.
It is found that if any memory allocation error occurs it is better to reallocate the entire memory.
Note that, since the power spectrum is a symmetric matrix, we can store only the upper half
of the FFTs and perform the difference operation only over this upper half.
*/
int cpu_allocation(int flg_mode, INDEX& nimages, INDEX& dimy, INDEX& dimx, INDEX& dim_power_spectrum, unsigned int* ram_radial_lut);
/*!
This function allocates the memory card and initialize all
sizes variable necessary to run the calculus for cpu and gpu execution.
*/
extern int (*diffmicro_allocation)(int flg_mode, INDEX& nimages, INDEX& dimy, INDEX& dimx, INDEX& dim_power_spectrum, unsigned int* ram_radial_lut);

/*!This function releases memory allocated by gpu_allocation*/
void gpu_free_pointers();
/*!This function releases memory allocated by cpu_allocation*/
void cpu_free_pointers();
/*!This function releases memory allocated by cpu_allocation or gpu_allocation.
The function pointer allows selecting the execution on gpu or cpu, while retaining the same code structure.*/
extern void (*diffmicro_free_pointers)();
/*!This function releases memory allocated by gpu_allocation*/
void gpu_deallocation();
/*!This function releases memory allocated by cpu_allocation*/
void cpu_deallocation();
/*!This function releases memory allocated by cpu_allocation or gpu_allocation.
The function pointer allows selecting the execution on gpu or cpu, while retaining the same code structure.*/
extern void (*diffmicro_deallocation)();
//-----------------------------------------------------------------------------------
// INPUT OUTPUT

/*!
This functions load a unsigned int 16bit image on the video card. Performs its Fourier Transform
on the temporary memory dev_fft and the store only the normalized upper half of the FFT in
the prepared memory area pointed by dev_store_ptr.

A value proportional to the average of the image is returned in the mean variable.
*/
int image_to_dev_gpu(SIGNED_INDEX ind_fifo, STORE_REAL &mean, unsigned short *im, bool flg_debug = false);
/*!
This functions load a unsigned int 16bit image, performs its Fourier Transform
on the temporary memory dev_fft_cpu and the store only the normalized upper half of the FFT in
the prepared memory area pointed by dev_store_ptr.

A value proportional to the average of the image is returned in the mean variable.
*/
int image_to_dev_cpu(SIGNED_INDEX ind_fifo, MY_REAL& mean, unsigned short* im, bool flg_debug = false);
/*!The function pointer allows selecting the execution on gpu or cpu, while retaining the same code structure.*/
extern int (*image_to_dev)(SIGNED_INDEX ind_fifo, STORE_REAL& mean, unsigned short* im, bool flg_debug);

/*!This function copies back the power spectra at the end of the calculation from the gpu memory*/
void copy_power_spectra_from_dev_gpu(STORE_REAL* power_spectrum_r);
/*!This function mimic the behavior of copy_power_spectra_from_dev_gpu*/
void copy_power_spectra_from_dev_cpu(STORE_REAL* power_spectrum_r);
/*!The function pointer allows selecting the execution on gpu or cpu, while retaining the same code structure.*/
extern void (*copy_power_spectra_from_dev)(STORE_REAL* power_spectrum_r);

//-----------------------------------------------------------------------------------
/*!
This function calulates all the power spectra of the differences between the FFTs of the images stored in
dev_images and the FFT stored in dev_fft.
	
To result of these operations are taken into account refreshing the average power spectra at different time delays.
To refresh the correct power spectrum with the correct refreshing coefficients the function is called with
counter_avg[] and dist_map[] arrays.
- counter_avg[] is the counter of how many power spectra have been already averaged in the [] memory area
- dist_map[] tells which time delay correspond to the difference (dev_images[] - dev_fft)
	to average the result of the operation in the correct memory area
*/
//int diff_autocorr_gpu(INDEX dim_file_list, INDEX dim_fifo, unsigned int *counter_avg, INDEX ind_sot, INDEX *file_index, INDEX *dist_map, std::vector<bool> &flg_valid_image_fifo, INDEX n_max_avg);
//int diff_autocorr_cpu(INDEX dim_file_list, INDEX dim_fifo, unsigned int* counter_avg, INDEX ind_sot, INDEX* file_index, INDEX* dist_map, std::vector<bool>& flg_valid_image_fifo, INDEX n_max_avg);
//extern int (*diff_autocorr)(INDEX dim_file_list, INDEX dim_fifo, unsigned int* counter_avg, INDEX ind_sot, INDEX* file_index, INDEX* dist_map, std::vector<bool>& flg_valid_image_fifo, INDEX n_max_avg);

template<typename TYPEFLOAT>
void complex_to_module_phase(INDEX dim, CUFFT_COMPLEX *in, TYPEFLOAT *outm, TYPEFLOAT *outp)
{
	INDEX i;
	for (i = 0; i < s_load_image.dim; ++i)
	{
		outm[i] = sqrt(in[i].x* in[i].x + in[i].y* in[i].y);
		outp[i] = atan2(in[i].y, in[i].x);
	}
	
}

/*!This function assign the correct pointer to function to execute the algorithm on cpu of gpu:
 - if hardware == 0 -> HARDWARE_CPU (see macro definition)
 - if hardware == 1 -> HARDWARE_GPU (see macro definition)
 */
void hardware_function_selection(INDEX harware);

//---------------------------------
/*!
convert an array of short into an array of complex values and multiplys by a real coefficient.
The imaginary part is zeroed.
- dim -> number of elements in the array
- vets -> input
- gain -> multiplicative term
- vetc -> output
*/
void shorttorealwithgain_cpu(INDEX dim, unsigned short vets[], FFTW_REAL gain, FFTW_COMPLEX vetc[]);
/*!
convert an array of short into an array of complex values and multiplys by a real coefficient.
The imaginary part is zeroed. This function uses multithreading option.
- nth -> number of threads for execution
- dim -> number of elements in the array
- vets -> input
- gain -> multiplicative term
- vetc -> output
*/
void short_to_real_with_gain_cpu(INDEX nth, INDEX dim, unsigned short vets[], FFTW_REAL gain, FFTW_COMPLEX vetc[]);

/*!This function copy a complex array into another with a real gain factor. Single thread execution.*/
void complextocomplexwithgain_cpu(INDEX dim, FFTW_COMPLEX vets[], FFTW_REAL gain, FFTW_COMPLEX vetc[]);
/*!This function copy a complex array into another with a real gain factor. Multithreading execution.
It uses complextocomplexwithgain_cpu.*/
void complex_to_complex_with_gain_cpu(INDEX nth, INDEX dim, FFTW_COMPLEX vets[], FFTW_REAL gain, FFTW_COMPLEX vetc[]);

/*!This function copy the real part of a complex array into an array with gain factor. Single thread execution.*/
void complextorealwithgain_cpu(INDEX dim, FFTW_COMPLEX vets[], FFTW_REAL gain, FFTW_REAL vetc[]);
/*!This function copy the real part of a complex array into an array with gain factor. Multithreading execution.
It uses complextorealwithgain_cpu.*/
void complex_to_real_with_gain_cpu(INDEX nth, INDEX dim, FFTW_COMPLEX vets[], FFTW_REAL gain, FFTW_REAL vetc[]);


/*! WARNING: Do not use this function in multithreading!!!*/
void gaincomplexlut_cpu(INDEX dim, unsigned int* lut, FFTW_COMPLEX in[], FFTW_REAL gain, FFTW_COMPLEX out[]);
//void gaincomplexlut_cpu(INDEX dim, unsigned int* lut, FFTW_COMPLEX in[], FFTW_REAL gain, FFTW_COMPLEX out[]);
//void gaincomplexlut_cpu(INDEX dim, unsigned int* lut, FFTW_COMPLEX in[], FFTW_REAL gain, FFTW_COMPLEX out[]);

/*!This function is a wrapper for the kernel diff_power_spectrum_to_avg_gpu.
It is used to implement the function pointer (*diff_power_spectrum_to_avg)
that allows executing on cpu and gpu without major code modification.*/
void diff_power_spectrum_to_avg_gpu_gpu(CUFFT_REAL coef1, CUFFT_REAL coef2, INDEX j, INDEX ind_dist);
/*!This function is a wrapper for the function diff_power_spectrum_to_avg_cpu.
This function is implemented to keep symmetry in the code between cpu and gpu execution.*/
void diff_power_spectrum_to_avg_cpu_cpu(FFTW_REAL coef1, FFTW_REAL coef2, INDEX j, INDEX ind_dist);
/*!This is the function pointer to execute diff_power_spectrum_to_avg_gpu_gpu or diff_power_spectrum_to_avg_cpu_cpu*/
extern void (*diff_power_spectrum_to_avg)(FFTW_REAL coef1, FFTW_REAL coef2, INDEX j, INDEX ind_dist);

/*!Matrix copy. Copy a row into a column*/
void cpx_row2col_gain_cpu(INDEX dim, INDEX dimx_in, INDEX i_row_in, FFTW_COMPLEX in[], FFTW_REAL gain, INDEX dimx_out, INDEX i_col_out, FFTW_COMPLEX out[]);
/*!Matrix copy. Copy a column into a row*/
void cpx_col2row_gain_cpu(INDEX dimcopy, INDEX dimx_in, INDEX i_col_in, FFTW_COMPLEX in[], FFTW_REAL gain, INDEX dimx_out, INDEX i_row_out, FFTW_COMPLEX out[]);

/*!This function executes the time series analysis on the wavevectors with selectable number of threads.*/
void timeseries_analysis_cpu(INDEX nth);
/*!This function executes the time series analysis on the wavevectors. CPU version.*/
void time_series_analysis_cpu();
/*!This function executes the time series analysis on the wavevectors. GPU version.*/
void time_series_analysis_gpu();
/*!The function pointer allows selecting the execution on gpu or cpu, while retaining the same code structure.*/
extern void (*time_series_analysis)();

/*! This function performs the analysis of the time series of each pixel of the FFT of the images*/
void timeseriesanalysis_cpu(INDEX dimtimeseries, FFTW_COMPLEX* tseries, INDEX dimfft, FFTW_COMPLEX* fft_memory, fftw_plan* tplan, FFTW_REAL* corr_memory);

/*! This function is used rearrange fft of the images after applying the radial look up table in the memory.
Each fft pixel is sent to the correct point of the timeseries.
The time series are composed by contigous wavevectors of the 2D FFTs coming from different images sorted by time increasing order.
There is a time series for each wavevector. CPU hardware version.*/
void lutfft_to_timeseries_cpu(INDEX dimcopy, FFTW_REAL gain, INDEX t, INDEX starting_freq);
/*! This function is used rearrange fft of the images after applying the radial look up table in the memory.
Each fft pixel is sent to the correct point of the timeseries.
The time series are composed by contigous wavevectors of the 2D FFTs coming from different images sorted by time increasing order.
There is a time series for each wavevector. GPU hardware version.*/
void lutfft_to_timeseries_gpu(INDEX dimcopy, CUFFT_REAL gain, INDEX t, INDEX starting_freq);
/*!The function pointer allows selecting the execution on gpu or cpu, while retaining the same code structure.*/
extern void (*lutfft_to_timeseries)(INDEX dimcopy, FFTW_REAL gain, INDEX t, INDEX starting_freq);

/*! This function is the inverse of lutfft_to_timeseries. It reconstructs the power spectra from the time series.*/
void timeseries_to_lutpw_gpu(INDEX dimcopy, CUFFT_REAL gain, INDEX t, INDEX starting_freq, STORE_REAL* ram_power_spectra);
/*! This function is the inverse of lutfft_to_timeseries. It reconstructs the ffts from the time series.*/
void timeseries_to_lutfft_gpu(INDEX dimcopy, CUFFT_REAL gain, INDEX t, INDEX starting_freq);
/*! This function is the inverse of lutfft_to_timeseries. It reconstructs the ffts from the time series.*/
void timeseries_to_lutfft_cpu(INDEX dimcopy, FFTW_REAL gain, INDEX t, INDEX starting_freq);
/*!The function pointer allows selecting the execution on gpu or cpu, while retaining the same code structure.*/
extern void (*timeseries_to_lutfft)(INDEX dimcopy, FFTW_REAL gain, INDEX t, INDEX starting_freq);

/*! Reversing operation done by lutfft_to_timeseries to reconstruct from the time series the images.
The results, being a power spectrum, is converted from a complex to a real memory area*/
void timeseries_to_lutpw_cpu(INDEX dimcopy, FFTW_REAL gain, INDEX t, INDEX starting_freq, STORE_REAL* ram_power_spectra);
/*!The function pointer allows selecting the execution on gpu or cpu, while retaining the same code structure.*/
extern void (*timeseries_to_lutpw)(INDEX dimcopy, FFTW_REAL gain, INDEX t, INDEX starting_freq, STORE_REAL* ram_power_spectra);

void Image_to_complex_matrix(unsigned short* dev_im_gpu_, CUFFT_COMPLEX* dev_fft_gpu_,int i);

void Image_to_complex_matrix2(unsigned short* dev_im_gpu_, int i, INDEX nimages);


void Calc_structure_function(INDEX nimages,int i,int device_count);

void Calc_StructureFunction_With_TimeCorrelation(INDEX nimages, INDEX dimx, INDEX dimy, FFTW_REAL* dev_images_cpu1);

//void timeseriesanalysis_gpu(INDEX dimtimeseries, CUFFT_COMPLEX* tseries, INDEX dimfft, CUFFT_COMPLEX* fft_memory, cufftHandle* tplan, CUFFT_REAL* corr_memory);

/*!This function calculates the operation over the time series of a single wavevector on cpu hardware.*/
void timeseriesanalysis_cpu(INDEX dimtimeseries, INDEX dim_t, FFTW_COMPLEX* tseries, INDEX dimfft, FFTW_COMPLEX* fft_memory, fftw_plan* tplan, FFTW_REAL* corr_memory);
/*!This function calculates the operation of the time series on a group of wavevectors on gpu hardware.*/
void timeseriesanalysis_gpu(INDEX dimtimeseries, INDEX dim_t, CUFFT_COMPLEX* tseries, INDEX dimfft, CUFFT_COMPLEX* fft_memory, cufftHandle* tplan, CUFFT_REAL* corr_memory, cuda_exec mycuda_dim_t, cuda_exec mycuda_dim, cuda_exec mycuda_dim_dim_t);

/*!This kernel is used to compute the average part of the TIME_CORRELATION algorithm.
It performs a two-sided in-place average of the time series values.*/
__global__ void averagesabs2_array_gpu(INDEX dim, INDEX dim_t, CUFFT_COMPLEX* _in, CUFFT_REAL* out);

#endif
