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

#include "stdafx.h"
#include "global_define.h"

#include <iostream>
#include <cstdlib>
#include <string>

#include "cuda_init.h"

#include "diffmicro_io.h"
#include "power_spectra.h"
#include "diffmicro_log.h"
#include "correlation.h"



//bool show_graphs = false;

void cpu_gpu_diff(INDEX dim, fftw_complex cpu_d[], cufftDoubleComplex gpu_d[], bool show) {

	double diff{};
	for (INDEX i = 0; i < dim; i++) {
		if (show) 
			std::cout << cpu_d[i][0] << " " << gpu_d[i].x << std::endl;
		diff = +abs(cpu_d[i][0] - gpu_d[i].x);


	}
	std::cout << "This is your difference : " << diff << std::endl;
}

void cpu_gpu_diff(INDEX dim, STORE_REAL cpu_d[], STORE_REAL gpu_d[],bool show) {

	double diff{};
	for (INDEX i = 0; i < dim; i++) {
		if (show)
			std::cout << cpu_d[i] << " " << gpu_d[i] << std::endl;
		diff = +abs(cpu_d[i] - gpu_d[i]);


	}
	std::cout << "This is your difference : " << diff << std::endl;
}


int dfd_main(int argc, char* argv[])
// argc = number of strings for starting the program = 2
// argv = array containing 'argc' strings for starting the program = diffmicro.exe path\users.txt
// users.txt contains all the info for starting the program

{
	INDEX dimx, dimy;
	unsigned short* im = NULL;
	//unsigned short* m_im = NULL;
	std::string path_ui;
	bool flg_file_init;
	INDEX nimages = (INDEX)(5);
	STORE_REAL* image_mean = new STORE_REAL[nimages];
	memset(image_mean, 0, nimages * sizeof(STORE_REAL));
	CUFFT_COMPLEX* dev_images_gpu_mem(NULL);
	STORE_REAL* dev_power_spectra_gpu_mem(NULL);
	fifo_min fifo;


	if (1 < argc)
	{
		path_ui = argv[1];
		flg_file_init = true;
	}
	else
	{
		path_ui = "user.txt";
		flg_file_init = false;
	}

	if (0 < path_ui.size())	if (true == useri.load(path_ui)) useri.variables_to_gui();

	// start the graphical user interface
	start_gui(path_ui, !flg_file_init);

	init_log(useri.file_list.size());
	general_stw.start();

	//load_binary_image(useri.file_list[0], dimy, dimx, false, im);
	load_image(useri.file_list[0], dimy, dimx, false, im, false);
	//switch (useri.display_status)
	//{
	//case DIFFMICRO_GRAPH_OFF :
	//	
	//	break;
	//case DIFFMICRO_GRAPH_ON :
	//	break;
	//default:
	//	std::cerr << "graph mode not defined" << std::endl; 
	//	break;
	//}
	//if (useri.display_status==DIFFMICRO_GRAPH_ON)
	//	initilize_display(dimx, dimy, useri.file_list.size());
	//im = new unsigned short[dimx * dimy];
	////m_im = new unsigned short[dimx * dimy];
	//if (useri.display_status==DIFFMICRO_GRAPH_ON)
	//	load_image(useri.file_list[0], dimy, dimx, true, im, true);


	if (false == cuda_init(false))
	{
		useri.hardware = HARDWARE_CPU;
	}

	hardware_function_selection(useri.hardware);

	radial_lut_dimr_from_max_freq(dimy / 2, dimx, (float)(useri.frequency_max), dim_radial_lut);
	ram_radial_lut = new unsigned int[dim_radial_lut];
	radial_storage_lut(dimy / 2, dimx, dim_radial_lut, ram_radial_lut);
	


	//=================================== GPU TEST ====== image_to_dev =======================================


	/*if (0 != diffmicro_allocation(nimages, dimy, dimx, dim_radial_lut, ram_radial_lut))
	{
		std::cerr << "error in allocating device memory" << std::endl;
		return false;
	}*/
	
	/*fifo.init(s_power_spectra.numerosity, s_load_image.dim);


	load_image(useri.file_list[0], dimy, dimx, true, m_im, flg_display_read);*/

	/*for (INDEX i{ 0 }; i < s_fft_images.dim; i++)
		std::cout << m_im[i]<< " ";
	std::cout << std::endl;*/

	/*INDEX ind_fifo = 0;
	if (1 == image_to_dev_gpu(ind_fifo, image_mean[ind_fifo], m_im, false))
	{
		std::cerr << "warning an image appears to be null" << std::endl
			<< "\t=> average has been put equal to one to avoid bad divisions" << std::endl;
	}
	std::cout << "Image mean by GPU is  :  " << image_mean[ind_fifo] << std::endl;*/


	//dev_images_gpu_mem = new CUFFT_COMPLEX[s_fft_images.dim * nimages]();
	//if (cudaMemcpy(dev_images_gpu_mem, dev_images_gpu, 2 * s_fft.memory_tot, cudaMemcpyDeviceToHost) != 0)
	//{
	//	std::cout << "Cuda error" << std::endl;
	//	return 1;
	//}

	//cudaDeviceSynchronize();
	STORE_REAL* ram_power_spectra;

	ram_power_spectra = new STORE_REAL[s_power_spectra.numerosity * dimx * dimy / 2];

	//----------------------------------------------------------------
	// Azimuthal averages initialization

	INDEX dimr;
	MY_REAL* azh_avgs = NULL;

	if ((dimx == dimy) && (true == useri.flg_write_azimuthal_avg))
	{
		power_spectra_to_azhavg_init(useri.nthread, dimx / 2, dimr);
		azh_avgs = new MY_REAL[dimr * nimages];

		//first distance is never calculated and, so, assigned to zero
		memset(azh_avgs, 0, sizeof(STORE_REAL) * dimr);
	}
	else
	{
		useri.flg_write_azimuthal_avg = false;
	}


	// group averages initialization
	unsigned int* power_spectra_avg_counter = new unsigned int[useri.file_list.size()];
	memset(power_spectra_avg_counter, 0, useri.file_list.size() * sizeof(unsigned int));

	INDEX useri_dist_max;
	

	if (nimages <= useri.dist_max)
	{
		useri_dist_max = nimages - 1;
	}
	else
	{
		useri_dist_max = useri.dist_max;
	}

	lldiv_t group;
	INDEX n_group, group_rem;

	group = std::div((long long)(useri_dist_max), (long long)(s_power_spectra.numerosity));
	n_group = (INDEX)(group.quot);
	group_rem = (INDEX)(group.rem);

	fifo.init(s_power_spectra.numerosity, s_load_image.dim);
	fifo.init(group_rem, s_load_image.dim);

	INDEX starting_index = 1;

	int flg_continue_averages = 1;
	// initializing array



	fifo.reset(starting_index);
	INDEX i = starting_index;
	while ((i < nimages) && (1 == flg_continue_averages))
	{
		
		fifo.load_next(i, useri.flg_valid_image, image_mean);
		if (true == useri.flg_valid_image[i])
		{
			time_differences.start();

			INDEX j, ind_dist;
			CUFFT_REAL coef1, coef2;
			INDEX ind_sot = i;
			INDEX dim_file_list = useri.file_list.size();
			INDEX dim_fifo = fifo.current_size;
			INDEX* dist_map = fifo.dist_map;
			INDEX n_max_avg = useri.n_pw_averages;
			unsigned int* counter_avg = power_spectra_avg_counter;
			std::vector<bool>& flg_valid_image_fifo = useri.flg_valid_image;
			INDEX* file_index = fifo.file_index;
			unsigned char* exe_map_row = &(execution_map[ind_sot * dim_file_list]);

			for (j = 0; j < dim_fifo; ++j)
			{
				//----------------------------------------------------------------------------
				// half power spectrum with average
				ind_dist = dist_map[j];
				
				// if number maximum number of averages have been reached for this power spectrum
				// no difference and refreshing of the power spectrum is made
				if ((n_max_avg != counter_avg[ind_dist]) && (true == flg_valid_image_fifo[j]))
				{
					coef1 = (CUFFT_REAL)(counter_avg[ind_dist]);
					++counter_avg[ind_dist];
					coef2 = (CUFFT_REAL)(1. / (CUFFT_REAL)(counter_avg[ind_dist]));
					coef1 *= coef2;

					diff_power_spectrum_to_avg(coef1, coef2, j, ind_dist);
					//diff_power_spectrum_to_avg_gpu << <s_power_spectra.cexe.nbk, s_power_spectra.cexe.nth >> >
					//	(s_power_spectra.dim, &(dev_images_gpu[j * s_fft_images.dim]), dev_image_sot_gpu, coef1, coef2, &(dev_power_spectra_gpu[ind_dist * s_power_spectra.dim]));


					exe_map_row[file_index[j]] += (unsigned char)(1);

					flg_continue_averages = 1;
				}

			}

			//flg_continue_averages = diff_autocorr(useri.file_list.size(), fifo.current_size, power_spectra_avg_counter, i, fifo.file_index, fifo.dist_map, useri.flg_valid_image, useri.n_pw_averages);
			time_differences.stop();
			n_computed_diff += fifo.current_size;
		}

		++i;
	}

	dev_images_gpu_mem = new CUFFT_COMPLEX[s_fft_images.dim * nimages]();
	if (cudaMemcpy(dev_images_gpu_mem, dev_images_gpu, 2 * s_fft.memory_tot, cudaMemcpyDeviceToHost) != 0)
	{
			std::cout << "Cuda error" << std::endl;
			return 1;
	}

	dev_power_spectra_gpu_mem = new STORE_REAL [s_power_spectra.dim * nimages]();

	if (cudaMemcpy(dev_power_spectra_gpu_mem, dev_power_spectra_gpu, s_power_spectra.memory_tot, cudaMemcpyDeviceToHost) != 0)
	{
		std::cout << "Cuda error" << std::endl;
		return 1;
	}
	gpu_deallocation();


	//=================================== CPU TEST ====== image_to_dev =======================================

	useri.hardware = HARDWARE_CPU;
	hardware_function_selection(useri.hardware);
	


	/*if (0 != diffmicro_allocation(nimages, dimy, dimx, dim_radial_lut, ram_radial_lut))
	{
		std::cerr << "error in allocating device memory" << std::endl;
		return false;
	}*/
	//load_image(useri.file_list[0], dimy, dimx, true, m_im, flg_display_read);

	//if (1 == image_to_dev_cpu(ind_fifo, image_mean[ind_fifo], m_im, false))
	//{
	//	std::cerr << "warning an image appears to be null" << std::endl
	//		<< "\t=> average has been put equal to one to avoid bad divisions" << std::endl;
	//}
	//std::cout << "Image mean by CPU is  :  " << image_mean[ind_fifo] << std::endl;

	starting_index = 1;

	flg_continue_averages = 1;
	// initializing array



	fifo.reset(starting_index);
	i = starting_index;
	while ((i < nimages) && (1 == flg_continue_averages))
	{

		fifo.load_next(i, useri.flg_valid_image, image_mean);
		if (true == useri.flg_valid_image[i])
		{
			time_differences.start();
			INDEX j, ind_dist;
			CUFFT_REAL coef1, coef2;
			INDEX ind_sot = i;
			INDEX dim_file_list = useri.file_list.size();
			INDEX dim_fifo = fifo.current_size;
			INDEX* dist_map = fifo.dist_map;
			INDEX n_max_avg = useri.n_pw_averages;
			unsigned int* counter_avg = power_spectra_avg_counter;
			std::vector<bool>& flg_valid_image_fifo = useri.flg_valid_image;
			INDEX* file_index = fifo.file_index;
			unsigned char* exe_map_row = &(execution_map[ind_sot * dim_file_list]);

			for (j = 0; j < dim_fifo; ++j)
			{
				//----------------------------------------------------------------------------
				// half power spectrum with average
				ind_dist = dist_map[j];

				// if number maximum number of averages have been reached for this power spectrum
				// no difference and refreshing of the power spectrum is made
				if ((n_max_avg != counter_avg[ind_dist]) && (true == flg_valid_image_fifo[j]))
				{
					coef1 = (CUFFT_REAL)(counter_avg[ind_dist]);
					++counter_avg[ind_dist];
					coef2 = (CUFFT_REAL)(1. / (CUFFT_REAL)(counter_avg[ind_dist]));
					coef1 *= coef2;

					diff_power_spectrum_to_avg(coef1, coef2, j, ind_dist);
					
					
					//void diff_power_spectrum_to_avg_cpu(INDEX nth, INDEX dim, FFTW_COMPLEX min[], FFTW_COMPLEX sot[], FFTW_REAL coef1, FFTW_REAL coef2, FFTW_REAL pw[])
					//diff_power_spectrum_to_avg_cpu(useri.nthread, s_power_spectra.dim, &(dev_images_cpu[j * s_fft_images.dim]), dev_fft_cpu,
					//	coef1, coef2, &(dev_power_spectra_cpu[ind_dist * s_power_spectra.dim]));


					exe_map_row[file_index[j]] += (unsigned char)(1);

					flg_continue_averages = 1;
				}

			}

			//flg_continue_averages = diff_autocorr(useri.file_list.size(), fifo.current_size, power_spectra_avg_counter, i, fifo.file_index, fifo.dist_map, useri.flg_valid_image, useri.n_pw_averages);
			time_differences.stop();
			n_computed_diff += fifo.current_size;
		}

		++i;
	}

	//cpu_gpu_diff(s_fft_images.dim, &(dev_images_cpu[ind_fifo * s_fft_images.dim]), &(dev_images_gpu_mem[ind_fifo * s_fft_images.dim]),true);
	//cpu_gpu_diff(s_fft_images.dim*(nimages-1), &(dev_images_cpu[0]), &(dev_images_gpu_mem[0]),true);
	cpu_gpu_diff(s_power_spectra.dim, &(dev_power_spectra_cpu[0]), &(dev_power_spectra_gpu_mem[0]),true);
	cpu_deallocation();

	//power_spectra_to_azhavg_free();
	//delete[] power_spectra_avg_counter;
	//delete[] ram_power_spectra;
	delete[] image_mean;
	delete[] ram_radial_lut;
	if (dev_images_gpu_mem != NULL)
		delete[] dev_images_gpu_mem;
	if (dev_power_spectra_gpu_mem != NULL)
		delete[] dev_power_spectra_gpu_mem;
	delete[] im;
	//delete[]  m_im;
	close_log();
	//	system("pause");
	//	delete[] im_default;
	return 0;
}

