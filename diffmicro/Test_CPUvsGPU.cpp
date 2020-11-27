//Copyright 2020 Mojtaba Norouzisadeh

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



int lll_main(int argc, char* argv[])
// argc = number of strings for starting the program = 2
// argv = array containing 'argc' strings for starting the program = diffmicro.exe path\users.txt
// users.txt contains all the info for starting the program

{
	INDEX dimx, dimy;
	unsigned short* im = NULL;
	std::string path_ui;
	bool flg_file_init;
	

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

	initilize_display(dimx, dimy, useri.file_list.size());
	im = new unsigned short[dimx * dimy];
	load_image(useri.file_list[0], dimy, dimx, true, im, true);

	if (false == cuda_init(false))
	{
		useri.hardware = HARDWARE_CPU;
	}
	
	hardware_function_selection(useri.hardware);

	calc_power_spectra(dimy, dimx);

	useri.hardware = HARDWARE_CPU;
	calc_power_spectra(dimy, dimx);

	//--------------------------------------------------------------
	// printing elapsed times
	general_stw.stop();

	std::cout << "-----------------------------------------------------------------" << std::endl;
	std::cout << "LOG" << std::endl << std::endl;
	print_log(std::cout, dimy, dimx);

	std::fstream fout;
	fout.open(useri.log_filename.c_str(), std::ios::out);
	print_log(fout, dimy, dimx);
	fout.close();

	waitkeyboard(0);
	//--------------------------------------------------------------
	close_display();
	delete[] im;
	close_log();
	//	system("pause");
	//	delete[] im_default;
	return 0;
}

