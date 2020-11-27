
//Copyright 2020 Giovanni Cerchiari, Mojtaba Norouzisadeh
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

#include <iostream>
#include "correlation.h"
#include <vector>
#include <diffmicro_io.h>
#include "global_define.h"


int zdf_main(int argc, char* argv[]) {

	int const dimx{ 6 }, dimy{ 6 };
	unsigned int long long dimy_l, dimx_l;
	unsigned short im[dimx * dimy];
	std::string file_name = "image_test.png";
	//MY_REAL* image_mean = new STORE_REAL[1];
	//memset(image_mean, 0, sizeof(STORE_REAL));
	double image_mean{ 0 };
	SIGNED_INDEX ind_fifo{1};
	unsigned int long long nimages{ 1 };

	if (false == cuda_init(false))
	{
		useri.hardware = HARDWARE_CPU;
	}
	useri.hardware = HARDWARE_CPU;
	hardware_function_selection(useri.hardware);


	load_image(file_name, dimy_l, dimx_l, 1, im, 0);

	INDEX dimr(0);
	float max_freq(8.5);
	unsigned int* lut;

	radial_lut_dimr_from_max_freq(dimy, dimx, max_freq, dimr);
	lut = new unsigned int[dimr];

	radial_storage_lut(dimy, dimx, dimr, lut);

	// need to use "diffmicro_allocation" to initialize some variables used in image_to_dev_cpu function
	// for example dev_fft_cpu and creating the plans for fftw

	if (0 != diffmicro_allocation(nimages, dimy_l, dimx_l, dimr, lut))		
	{
		std::cerr << "error in allocating device memory" << std::endl;
		return false;
	}


	std::cout << "image to dev result :  " <<image_to_dev(ind_fifo, image_mean, im, 0) << " image mean "  <<image_mean;
	

	return 0;
}

