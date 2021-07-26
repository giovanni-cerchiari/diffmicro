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
\mainpage
diffmicro.cpp : Defines the entry point for the console application.

The calculus that is going to be performed can be explained in the following way.
A set of images equally spaced in time is given.
The pictures are usually taken with a digital camera with a fixed frame rate.
The program must evaluate all the power spectra belonging to the difference of any possible couple of images.
All these power spectra can be organized in groups in respect to the time delay between the two images that generates the differences.
The program should output the average power spectrum of each group and its azimuthal average.\n
Note that, since the power spectrum is a symmetric matrix we can store only the upper half
of the FFTs and perform the difference operation only over this upper half. So the output of the program is
only the upper half of the averaged power spectra.
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


 int main(int argc, char* argv[])
	// argc = number of strings for starting the program = 2
	// argv = array containing 'argc' strings for starting the program = diffmicro.exe path\users.txt
	// users.txt contains all the info for starting the program

{

	
	INDEX dimx, dimy;
	
	std::string path_ui;
	bool flg_file_init;
	std::vector<std::string > list_dir;
	std::vector<std::string > list_file_tmp;
	std::string path1;

	std::cout <<"==============================================================="<<std::endl;
	std::cout <<"diffmicro  Copyright (C) 2011  Giovanni Cerchiari"<<std::endl;
	std::cout <<"diffmicro  Copyright (C) 2020  Giovanni Cerchiari, Mojtaba Norouzisadeh"<<std::endl;
	std::cout <<"This program comes with ABSOLUTELY NO WARRANTY; for details visit : https://www.gnu.org/licenses/"<<std::endl;
	std::cout <<"This is free software, and you are welcome to redistribute it"<<std::endl;
	std::cout <<"under certain conditions;  visit : https://www.gnu.org/licenses/ for details."<<std::endl;
	std::cout <<"==============================================================="<<std::endl;
	

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

	if (0 < path_ui.size())	
		if (true == useri.load(path_ui)) {

			//useri.variables_to_gui();

		}

	// start the graphical user interface
	//start_gui(path_ui, !flg_file_init);
	/*if (false == list_directory(useri.path_input, list_dir, list_file_tmp))
	{
		return false;
	}

	std::cout << useri.series.size() << std::endl;*/
	for (int i = 0; i < useri.series.size(); i++) {

		unsigned short* im = NULL;
		/*if (useri.nb_of_series == 1) {

			useri.list_images_in_file(useri.path_input, useri.file_list);

		}
		else {*/

		path1 = useri.path_input + useri.series[i] + "\\";
		useri.list_images_in_file(path1, useri.file_list);

		//}



		if (useri.flg_graph_mode)
			init_figure_enviroment();



		init_log(useri.file_list.size());
		general_stw.start();

		//load_binary_image(useri.file_list[0], dimy, dimx, false, im);
		//std::string path = "C:\\samples\\performance_image 512\\pippo_0000.tif";
		//load_image(useri.file_list[0], dimy, dimx, false, im, false);
		FILE* file_bin;
		unsigned short dimx_y;
		if (useri.binary == 0) {
			cv::Mat img_cv;
			img_cv = cv::imread(useri.file_list[0], CV_LOAD_IMAGE_ANYDEPTH);

			if (!img_cv.data)                              // Check for invalid input
			{
				std::cerr << "Could not open or find image : " << useri.file_list[0] << std::endl;
				return false;
			}

			dimx = img_cv.cols;
			dimy = img_cv.rows;
		}
		else
		{
			
			file_bin = fopen(useri.file_list[0].c_str(), "rb");
			if (file_bin == NULL)
				std::cout << "ERROR" << std::endl;
			int nb_val_lues = fread(&dimx_y, sizeof(unsigned short), 1, file_bin);
			dimy = dimx_y;
			dimx = dimx_y;
			//nb_val_lues = fread(im, sizeof(unsigned short), dimx * dimy, file_bin);
			fclose(file_bin);

		}
		

		if (useri.flg_graph_mode)
			initilize_display(dimx, dimy, useri.file_list.size());
		im = new unsigned short[dimx * dimy];
		load_image(useri.file_list[0], dimy, dimx, true, im, useri.flg_graph_mode);

		if (!useri.flg_execution_mode)
		{
			useri.execution_mode = DIFFMICRO_MODE_TIMECORRELATION;
			std::cout << "execution in \"time correlation\" mode on ";
		}
		else {
			useri.execution_mode = DIFFMICRO_MODE_FIFO;
			std::cout << "execution in \"FIFO\" mode on " << std::endl;
		}

		if (!useri.flg_hardware_selection)
		{
			useri.hardware = HARDWARE_GPU;
			std::cout << "GPU selected " << std::endl;
		}
		else {
			useri.hardware = HARDWARE_CPU;
			std::cout << "CPU selected " << std::endl;
		}

		if (false == cuda_init(false))
		{
			std::cerr << "Execution on GPU is not possible" << std::endl << "Execuation changed to CPU" << std::endl;

			useri.hardware = HARDWARE_CPU;
		}


		hardware_function_selection(useri.hardware);

		calc_power_spectra(dimy, dimx);

		general_stw.stop();

		plot_dynamics(dimx,i);

		//--------------------------------------------------------------
		// printing elapsed times

		std::cout << "-----------------------------------------------------------------" << std::endl;
		std::cout << "LOG" << std::endl << std::endl;
		print_log(std::cout, dimy, dimx);

		std::fstream fout;
		fout.open(useri.log_filename.c_str(), std::ios::out);
		print_log(fout, dimy, dimx);
		fout.close();
		delete[] im;
	}
		waitkeyboard(10);
		//--------------------------------------------------------------
		close_display();
		
		close_log();
	
	system("pause");
//	delete[] im_default;
	return 0;
}

