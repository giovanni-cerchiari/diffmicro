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

#include "correlation.h"

int (*diffmicro_allocation)(int flg_mode, INDEX& nimages, INDEX& dimy, INDEX& dimx, INDEX& dim_power_spectrum, unsigned int* ram_radial_lut);
void (*diffmicro_free_pointers)();
void (*diffmicro_deallocation)();
int (*image_to_dev)(SIGNED_INDEX ind_fifo, STORE_REAL& mean, unsigned short* im, bool flg_debug);
void (*copy_power_spectra_from_dev)(STORE_REAL* power_spectrum_r);
//int (*diff_autocorr)(INDEX dim_file_list, INDEX dim_fifo, unsigned int* counter_avg, INDEX ind_sot, INDEX* file_index, INDEX* dist_map, std::vector<bool>& flg_valid_image_fifo, INDEX n_max_avg);
void (*diff_power_spectrum_to_avg)(FFTW_REAL coef1, FFTW_REAL coef2, INDEX j, INDEX ind_dist);
void (*time_series_analysis)();

void (*lutfft_to_timeseries)(INDEX dimcopy, FFTW_REAL gain, INDEX t, INDEX starting_freq);
void (*timeseries_to_lutfft)(INDEX dimcopy, FFTW_REAL gain, INDEX t, INDEX starting_freq);
void (*timeseries_to_lutpw)(INDEX dimcopy, FFTW_REAL gain, INDEX t, INDEX starting_freq, STORE_REAL *ram_power_spectra);


void hardware_function_selection(INDEX hardware)
{
	switch (hardware)
	{
	case HARDWARE_CPU:
		std::cerr << "execution on CPU" << std::endl;
		diffmicro_allocation = cpu_allocation;
		diffmicro_free_pointers = cpu_free_pointers;
		diffmicro_deallocation = cpu_deallocation;
		image_to_dev = image_to_dev_cpu;
		copy_power_spectra_from_dev = copy_power_spectra_from_dev_cpu;
		diff_power_spectrum_to_avg = diff_power_spectrum_to_avg_cpu_cpu;

		time_series_analysis = time_series_analysis_cpu;
		timeseries_to_lutpw = timeseries_to_lutpw_cpu;
		timeseries_to_lutfft = timeseries_to_lutfft_cpu;
		lutfft_to_timeseries = lutfft_to_timeseries_cpu;
		break;
	case HARDWARE_GPU:
		std::cerr << "execution on GPU" << std::endl;
		diffmicro_allocation = gpu_allocation;
		diffmicro_free_pointers = gpu_free_pointers;
		diffmicro_deallocation = gpu_deallocation;
		image_to_dev = image_to_dev_gpu;
		switch (useri.execution_mode)
		{
		case DIFFMICRO_MODE_FIFO:
			copy_power_spectra_from_dev = copy_power_spectra_from_dev_gpu;
			break;
		case DIFFMICRO_MODE_TIMECORRELATION:
			copy_power_spectra_from_dev = copy_power_spectra_from_dev_cpu;
			break;
		default:

			break;
		}
		
		diff_power_spectrum_to_avg = diff_power_spectrum_to_avg_gpu_gpu;
		lutfft_to_timeseries = lutfft_to_timeseries_gpu;
		//time_series_analysis = time_series_analysis_gpu;
		timeseries_to_lutpw = timeseries_to_lutpw_cpu;
		timeseries_to_lutfft = timeseries_to_lutfft_gpu;

		break;
	default:
		std::cerr << "error: hardare not defined" << std::endl;
		break;
	}
}

