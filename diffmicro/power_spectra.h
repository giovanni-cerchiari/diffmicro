/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 2011
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
This functions and classes are written for diffmicro.exe application.

This file contains the high-level execution of the algorithms. In this file it is not known if the program is
executing on GPU or CPU hardware. The difference is implemented via function pointers, which are used by the
functions here implemented.
*/

#ifndef _POWER_SPECTRA_H_
#define _POWER_SPECTRA_H_

#include "global_define.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>

#include "prep_string.h"

#include "diffmicro_io.h"
#include "correlation.h"
#include "angular_avg.h"

#include "diffmicro_log.h"

/*!
This class is the memory manager of the power spectra calculus.
Loading of the images is organized as a FIFO. The FIFO is only "virtual":
no memory area is cleaned or recreated, but allocated in the initialization 
of the program. This class uses this memory using a browsing index ind_fifo
that selects at each step a differet pointer inside the memory area.
*/
class fifo_min
{
	public:
		fifo_min();
		~fifo_min();

		/*!maximum number of upper half FFTs of images storable on the graphic card*/
		INDEX size;
		/*!current number of elements stored in the FIFO*/
		INDEX current_size;
		/*!this array associates the index of the file stored to ind_fifo*/
		INDEX *file_index;
		/*!This flg is used to enable or disable one image into the fifo*/
		bool *flg_valid_image;
		/*!file to be loaded in the FIFO*/
		INDEX ind_file;
		/*!periodical index that points to a memory area of the FIFO*/
		INDEX ind_fifo;
		/*!bias to evaluate the correct dist_map. It corresponds to the starting index of the
		group-diagonal during the process*/
		INDEX dist_bias;
		/*!This array associates the time delay between the images in the FIFO and the image to be
		substracted to the memory area of the corresponding power spectrum*/
		INDEX *dist_map;
	

		/*!clear variables and allocated memory*/
		void clear();
		/*!initilization:
		- _size_fifo = maximum number of half FFTs of images storable on the graphic card
		- image_size = number of elements of a single image (necessary to allocate m_im)*/
		void init(INDEX _size_fifo, INDEX image_size);
		/*!reset function is called before calculating a group of power spectra. _dist_bias 
		corresponds to the starting index of the	group-diagonal during the process*/
		void reset(INDEX _dist_bias);
		/*!refresh FIFO status loading the next pair of images. One in the FIFO and the other in dev_fft
		memory area*/
		INDEX load_next(INDEX ind_sot, std::vector<bool> &flg_valid_image, STORE_REAL *image_mean);

	protected:
	private:

		/*!temporary RAM memory area where to store the image that is then loaded on the graphic card*/
		unsigned short *m_im;

};


/*!
This function manage all the calculations of the program in FIFO mode.
In the first part memory areas are prepared for the calculus.
In the second part the calculus is performed. \n
The calculus of the differences is made on the video card in the Fourier space.
Since the output should be all the power spectra averaged in respect to any possible time delay,
only the averaged power spectra needs to saved and not all the power spectra coming from all the
possible couple of differences. For this reason only the memory of the averaged power spectra is
allocated. The optimal memory occupancy of the video card is found to be when we can perform as
many differences as many power spectra we can store. This means that the total number of storable
FFTs should be equal to the total number of power spectra that are present at the same time on
the video card. \n
For this reason the memory IO with the video card is organized as follows.
First we select a group of power spectra to be calculated.
These power spectra are intended displaced in time-delay one aside to the other.
Then we organize a FIFO over the memory area of the FFTs.
At each cycle of the algorithm we load two images, one into the FIFO and the other in the
temporary memory area where all the FFTs are calculated. In the FIFO are stored in sequence
a group of images separated in time by the frame rate of the camera.
The picture outside the FIFO is selected in order to satisfy the appropriate temporal delay
condition for all the images in the FIFO. In fact since the power spectra are of the same number
as the images stored and both of them contains elements contiguous in time separation;
it is possible to select the appropriate frame that make all the possible differences storable
in into the memory area of the power spectra with a refreshing calculus of the average value
*/
bool calc_power_spectra(INDEX dimy, INDEX dimx);

/*! This function allows you to plot the structure function */
void plot_dynamics(INDEX dimx,INDEX i);

/*!Calculating power spectra via time differences and FIFO memory*/
void calc_power_spectra_fifo(INDEX nimages, INDEX& useri_dist_max, STORE_REAL* image_mean, INDEX& dimr, unsigned int* power_spectra_avg_counter, STORE_REAL* ram_power_spectra, MY_REAL* azh_avgs);

void calc_power_spectra_fifo_cc(int size_freq, INDEX dimx,INDEX dimy,INDEX nimages, INDEX& useri_dist_max, STORE_REAL* image_mean, INDEX& dimr, unsigned int* power_spectra_avg_counter, STORE_REAL* ram_power_spectra, MY_REAL* azh_avgs);

/*! Calculating the power spectra via Fourier tranform of power series*/
void calc_power_spectra_autocorr(INDEX dimy, INDEX dimx, INDEX nimages, STORE_REAL* image_mean, INDEX& dimr, unsigned int* power_spectra_avg_counter, STORE_REAL* ram_power_spectra, MY_REAL* azh_avgs);

/*! Calculating the power spectra via Fourier tranform of power series using CUDA 2D*/
void calc_power_spectra_autocorr_2D(INDEX dimy, INDEX dimx, INDEX nimages, STORE_REAL* image_mean, INDEX& dimr, unsigned int* power_spectra_avg_counter, STORE_REAL* ram_power_spectra, MY_REAL* azh_avgs);

void calc_power_spectra_autocorr_2D_FFTshifted(INDEX dimy, INDEX dimx, INDEX nimages, STORE_REAL* image_mean, INDEX& dimr, unsigned int* power_spectra_avg_counter, STORE_REAL* ram_power_spectra, MY_REAL* azh_avgs);

/*!This function executes the calculation on a macro-diagonal of the FIFO algorithm.*/
void calc_diagonal(INDEX starting_index, unsigned int power_spectra_avg_counter[], fifo_min &fifo, INDEX nimages, STORE_REAL image_mean[], bool flg_debug = false);

/*!
copy n_pw power spectra from video card to ram_power_spectra. Note that power_spectra on the video card
are already stored one in queue to the other
*/
void power_spectra_from_dev(INDEX n_pw, unsigned int* ram_radial_lut, STORE_REAL ram_power_spectra[], INDEX start_dist_display = 0, bool flg_debug = false);
/*!This function updates the power spectra with differences of ffts. It is used in the FIFO algorithm.*/
int diff_autocorr(INDEX dim_file_list, INDEX dim_fifo, unsigned int* counter_avg, INDEX ind_sot, INDEX* file_index, INDEX* dist_map, std::vector<bool>& flg_valid_image_fifo, INDEX n_max_avg);

/*!This function loads all the pictures to fill the memory area with the time series.
The time series are the series of fft pixel of each image sorted by increasing time.*/
void load_memory_for_time_correlation(INDEX dimx, INDEX dimy, INDEX nimages, INDEX start_spacial_freq_in_lut, INDEX dimfreq, unsigned short* m_im, FFTW_REAL* image_mean);

void load_memory_for_time_correlation_2D(INDEX dimx, INDEX dimy, INDEX nimages, INDEX start_spacial_freq_in_lut, INDEX dimfreq, unsigned short* m_im, FFTW_REAL* image_mean);

void load_memory_for_time_correlation_2D_FFTshifted(INDEX dimx, INDEX dimy, INDEX nimages, INDEX start_spacial_freq_in_lut, INDEX dimfreq, unsigned short* m_im, FFTW_REAL* image_mean);

/*!This function reads the final result of the calculation to reconverted the elaborated time series into images*/
void read_memory_after_time_correlation(INDEX nimages, INDEX dimr, MY_REAL* azh_avgs, FFTW_REAL* ram_power_spectra);

/*! This function saves partial results of the time series analysis on disk to be merged in the final stage of the program.
The temporary files that are written are the output files, which are subsequently overwritten after the merge.*/
void save_partial_timeseries(INDEX nimages, INDEX igroup, INDEX dimgroup, STORE_REAL* ram_power_spectra);

void pw_azth_avg2(unsigned int* lut, INDEX npw, INDEX dimr, MY_REAL azh_avgs[], STORE_REAL ram_power_spectra[], FFTW_COMPLEX* dev_images_cpu);

void plot_dynamics_cc(INDEX nimages, int m);


#endif