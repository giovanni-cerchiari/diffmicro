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

#include "stdafx.h"
#include "diffmicro_log.h"

stopwatch time_reading_from_disk;
stopwatch time_writing_on_disk;
stopwatch time_from_host_to_device;
stopwatch time_from_device_to_host;
stopwatch time_fft_norm;
stopwatch time_differences;
stopwatch time_azh_avg;
stopwatch general_stw;
stopwatch time_time_correlation;
stopwatch time_reshuffling_memory;

unsigned long n_computed_fft(0);
unsigned long n_computed_diff(0);
unsigned long n_groups_reload(0);

unsigned long radial_lut_length(0);
unsigned long n_capacity(0);

unsigned long tot_memory_fft_images(0);
unsigned long tot_calculation_memory(0);

unsigned char *execution_map(NULL);

void print_log(std::ostream &out, INDEX &dimy, INDEX &dimx)
{
	out << "device : " << deviceProp.name << std::endl << std::endl;
	out <<"EXECUTION_SIZES"<<std::endl;
	out <<"#_images:\t"<<useri.file_list.size()<<std::endl;
	out <<"image_size:\t"<<dimx<<"\t"<<dimy<<std::endl;
	out <<"#_storable_FFT:\t"<<s_power_spectra.numerosity<<std::endl;
	out <<"#_averages_requested:\t"<<useri.get_int(N_PW_AVERAGES)<<std::endl;
	out <<"#_computed_FFTs:\t"<<n_computed_fft<<std::endl;
	out <<"#_computed_differences:\t"<<n_computed_diff<<std::endl;
	out <<std::endl<<"ELAPSED_TIMES"<<std::endl;
	out <<"1)reading_from_disk_[s]:\t"<<time_reading_from_disk.t()<<std::endl;
	out <<"2)writing_to_disk_[s]:\t"<<time_writing_on_disk.t()<<std::endl;
	out <<"3)comunication_from_host_to_device_[s]:\t"<<time_from_host_to_device.t()<<std::endl;
	out <<"4)comunication_from_device_to_host_[s]:\t"<<time_from_device_to_host.t()<<std::endl;
	out <<"5)FFT_and_normalization_of_images_[s]:\t"<<time_fft_norm.t()<<std::endl;
	out <<"6)differences_between_FFTs_of_images_[s]:\t"<<time_differences.t()<<std::endl;
	out <<"7)azimuthal_averages_[s]:\t"<<time_azh_avg.t()<<std::endl;
	if (!useri.flg_execution_mode)
	{
		out << "8)time_reshuffling_memory_[s]:\t" << time_reshuffling_memory.t() << std::endl;
		out << "9)time_time_correlation_[s]:\t" << time_time_correlation.t() << std::endl;
	}
	out <<"total_execution_time_[s]:\t"<<general_stw.t()<<std::endl;
	
	if (!useri.flg_execution_mode)
	{
		out << "tot_-_sum_n):\t" << general_stw.t() -
			(time_reading_from_disk.t() + time_writing_on_disk.t() +
				time_from_host_to_device.t() + time_from_device_to_host.t() +
				time_fft_norm.t() + time_differences.t() + time_azh_avg.t()+
				time_reshuffling_memory.t()+ time_time_correlation.t())
			<< std::endl;
	}
	else
	{
		out << "tot_-_sum_n):\t" << general_stw.t() -
			(time_reading_from_disk.t() + time_writing_on_disk.t() +
				time_from_host_to_device.t() + time_from_device_to_host.t() +
				time_fft_norm.t() + time_differences.t() + time_azh_avg.t())
			<< std::endl;
	}
	out << "number_of_groups:\t" << n_groups_reload << std::endl;
	out << "radial_lut_length:\t" << radial_lut_length << std::endl;
	out << "n_capacity:\t" << n_capacity << std::endl;
	out << "tot_memory_fft_images:\t" << tot_memory_fft_images << std::endl;
	out << "tot_calculation_memory:\t" << tot_calculation_memory << std::endl;
}

void init_log(INDEX dim)
{
	if (NULL != execution_map) close_log();
	execution_map = new unsigned char[dim*dim];
	memset(execution_map, 0, dim*dim*sizeof(char));
}

void close_log()
{
	if (NULL != execution_map) delete[] execution_map;
	execution_map = NULL;
}

