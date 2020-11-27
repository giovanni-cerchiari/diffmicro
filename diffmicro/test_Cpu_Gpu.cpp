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
//#include "plot_par.h"

//bool show_graphs = false;



int popo_main(int argc, char* argv[])
// argc = number of strings for starting the program = 2
// argv = array containing 'argc' strings for starting the program = diffmicro.exe path\users.txt
// users.txt contains all the info for starting the program

{
	INDEX dimx, dimy;
	unsigned short* im = NULL;
	unsigned short* m_im = NULL;
	std::string path_ui;
	bool flg_file_init;
	INDEX nimages = (INDEX)(5);
	STORE_REAL* image_mean = new STORE_REAL[nimages];
	memset(image_mean, 0, nimages * sizeof(STORE_REAL));
	CUFFT_COMPLEX* dev_images_gpu_mem(NULL);



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
	/*if (useri.display_status==DIFFMICRO_GRAPH_ON)
		initilize_display(dimx, dimy, useri.file_list.size());
	im = new unsigned short[dimx * dimy];
	m_im = new unsigned short[dimx * dimy];
	if (useri.display_status==DIFFMICRO_GRAPH_ON)
		load_image(useri.file_list[0], dimy, dimx, true, im, true);*/


	if (false == cuda_init(false))
	{
		useri.hardware = HARDWARE_CPU;
	}
	radial_lut_dimr_from_max_freq(dimy / 2, dimx, (float)(useri.frequency_max), dim_radial_lut);
	ram_radial_lut = new unsigned int[dim_radial_lut];
	radial_storage_lut(dimy / 2, dimx, dim_radial_lut, ram_radial_lut);

	//=================================== GPU TEST ====== image_to_dev =======================================
	/*if (0 != gpu_allocation(nimages, dimy, dimx, dim_radial_lut, ram_radial_lut))
	{
		std::cerr << "error in allocating device memory" << std::endl;
		return false;
	}*/

	load_image(useri.file_list[0], dimy, dimx, true, m_im, flg_display_read);
	
	/*for (INDEX i{ 0 }; i < s_fft_images.dim; i++)
		std::cout << m_im[i]<< " ";
	std::cout << std::endl;*/

	INDEX ind_fifo = 0;
	if (1 == image_to_dev_gpu(ind_fifo, image_mean[ind_fifo], m_im, false))
	{
		std::cerr << "warning an image appears to be null" << std::endl
			<< "\t=> average has been put equal to one to avoid bad divisions" << std::endl;
	}
	std::cout << "Image mean by GPU is  :  " << image_mean[ind_fifo] << std::endl;


	dev_images_gpu_mem = new CUFFT_COMPLEX[s_fft_images.dim * nimages]();
	if (cudaMemcpy(dev_images_gpu_mem, dev_images_gpu, 2*s_fft.memory_tot, cudaMemcpyDeviceToHost) != 0) 
	{
		std::cout << "Cuda error" << std::endl;
		return 1;
	}
		
	cudaDeviceSynchronize();
	gpu_deallocation();
	

	//=================================== CPU TEST ====== image_to_dev =======================================
	

	

	/*if (0 != cpu_allocation(nimages, dimy, dimx, dim_radial_lut, ram_radial_lut))
	{
		std::cerr << "error in allocating device memory" << std::endl;
		return false;
	}*/
	load_image(useri.file_list[0], dimy, dimx, true, m_im, flg_display_read);
	
	if (1 == image_to_dev_cpu(ind_fifo, image_mean[ind_fifo], m_im, false))
	{
		std::cerr << "warning an image appears to be null" << std::endl
			<< "\t=> average has been put equal to one to avoid bad divisions" << std::endl;
	}
	std::cout << "Image mean by CPU is  :  " << image_mean[ind_fifo] << std::endl;


	double diff{};
	for (INDEX i = 0; i < s_fft_images.dim; i++) {
		std::cout << &(dev_images_cpu[ind_fifo * s_fft_images.dim+i][0]) << " " << &(dev_images_gpu_mem[ind_fifo * s_fft_images.dim+i].x) << std::endl;
		diff = +abs(&(dev_images_cpu[ind_fifo * s_fft_images.dim + i][0]) - &(dev_images_gpu_mem[ind_fifo * s_fft_images.dim + i].x));


	}
	std::cout << "This is your difference : " << diff << std::endl;

	cpu_deallocation();

	//power_spectra_to_azhavg_free();
	//delete[] power_spectra_avg_counter;
	//delete[] ram_power_spectra;
	delete[] image_mean;
	delete[] ram_radial_lut;
	if (dev_images_gpu_mem != NULL)
		delete[] dev_images_gpu_mem;
	delete[] im, m_im;
	close_log();
	//	system("pause");
	//	delete[] im_default;
	return 0;
}

