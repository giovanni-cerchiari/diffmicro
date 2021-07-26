/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 2016
*/

/*
Copyright: Mojtaba Norouzisadeh, Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
e-mail:norouzi.mojtaba.sade@gmail.com

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
#include "diffmicro_io.h"

bool get_file_lines_number(std::string filename, INDEX &n_lines)
{
	std::fstream fin;
	std::string line;

	fin.open(filename.c_str(), std::ios::in);
	if (!fin.good()) return false;

	n_lines = 0;
	while (fin.good())
	{
		std::getline(fin, line);
		++n_lines;
	}

	fin.close();
	return true;
}

void diffmicro_user_interface::variables_to_gui()
{
	set_string(PATH_IMAGES, path_input);
	/*set_string(TIME_FILENAME, file_times);*/
	set_string(VALID_IMAGE_FILENAME, file_valid_images);
	set_int(FIRST_IMAGE, first_image);
	set_int(LAST_IMAGE, last_image);
	set_int(N_PW_AVERAGES, n_pw_averages);
	set_int(DIST_MAX, dist_max);
	/*set_int(TIME_STEP, (INDEX)(time_step));*/
	set_int(FREQUENCY_MAX, frequency_max);

	set_int(NUMBER_THREAD_GPU, nthread_gpu);
	set_int(NUMBER_THREAD_CPU, nthread);
	set_int(RAM_CPU, RAM);

	set_string(PATH_POWER_SPECTRA, path_power_spectra);
	set_bool(FLG_WRITE_POWER_SPECTRA, flg_write_power_spectra);
	set_bool(EXECUTION_MODE, flg_execution_mode);
	set_bool(HARDWARE_SELECTION, flg_hardware_selection);
	set_bool(GRAPH_MODE, flg_graph_mode);

	set_bool(FLG_WRITE_AZIMUTHAL_AVG, flg_write_azimuthal_avg);
	set_bool(FLG_START, flg_start);
	set_bool(FLG_TIME_FILENAME, flg_time_file);
	set_bool(FLG_VALID_IMAGE_FILENAME, flg_valid_image_file);
}


void diffmicro_user_interface::gui_to_variables()
{
	path_input = get_string(PATH_IMAGES);
	first_image = get_int(FIRST_IMAGE);
	last_image = get_int(LAST_IMAGE);
	n_pw_averages = get_int(N_PW_AVERAGES);
	dist_max = get_int(DIST_MAX);
	/*file_times = get_string(TIME_FILENAME);*/
	flg_valid_image_file = get_bool(FLG_VALID_IMAGE_FILENAME);
	file_valid_images = get_string(VALID_IMAGE_FILENAME);
	/*time_step = (MY_REAL)(get_int(TIME_STEP));*/
	frequency_max = get_int(FREQUENCY_MAX);
	path_power_spectra = get_string(PATH_POWER_SPECTRA);
	flg_write_power_spectra = get_bool(FLG_WRITE_POWER_SPECTRA);
	flg_execution_mode = get_bool(EXECUTION_MODE);
	flg_hardware_selection = get_bool(HARDWARE_SELECTION);
	flg_graph_mode= get_bool(GRAPH_MODE);

	nthread_gpu = get_int(NUMBER_THREAD_GPU);
	nthread = get_int(NUMBER_THREAD_CPU);
	RAM = get_int(RAM_CPU);

	flg_write_azimuthal_avg = get_bool(FLG_WRITE_AZIMUTHAL_AVG);
	flg_start = get_bool(FLG_START);
}


bool diffmicro_user_interface::save(std::string filename)
{
	bool flg;
	std::string value;
	std::fstream fout;
	ui_variable *var;
	if (0 == filename.size()) filename = "user.txt";
	
	fout.open(filename.c_str(), std::ios::out);

	if (!fout.good()) return false;

	var = get_variable(PATH_IMAGES); var->control->get_value(&value);
	fout << "path_images\t" << value << std::endl;
	var = get_variable(FIRST_IMAGE); var->control->get_value(&value);
	fout << "images_interval\t" << value;
	var = get_variable(LAST_IMAGE); var->control->get_value(&value);
	fout << "\t" << value << std::endl;
	/*flg = get_bool(FLG_TIME_FILENAME);
	fout << "flg_time_file\t";
	if (true == flg) fout << "1" << std::endl;
	else             fout << "0" << std::endl;
	value = get_string(TIME_FILENAME);
	if(0 != value.size()) fout << "file_times\t" << value << std::endl;
	fout << "time_step\t" << get_int(TIME_STEP) << std::endl;*/
	value = get_string(VALID_IMAGE_FILENAME);
	fout << "flg_file_valid_image\t" << get_bool(FLG_VALID_IMAGE_FILENAME) << std::endl;
	if (0 != value.size()) fout << "file_valid_images\t" <<value << std::endl;
	var = get_variable(N_PW_AVERAGES); var->control->get_value(&value);
	fout << "number_of_averages\t" << value << std::endl;
	var = get_variable(DIST_MAX); var->control->get_value(&value);
	fout << "delay_max\t" << value << std::endl;
	var = get_variable(FREQUENCY_MAX); var->control->get_value(&value);
	fout << "frequency_max\t" << value << std::endl;
	var = get_variable(PATH_POWER_SPECTRA); var->control->get_value(&value);
	fout << "path_output\t" << value << std::endl;
	fout << "flg_write_images_average\t1" << std::endl;
	var = get_variable(FLG_WRITE_POWER_SPECTRA); var->control->get_value(&flg);
	fout << "flg_write_power_spectra\t";
	if (true == flg) fout << "1" << std::endl;
	else             fout << "0" << std::endl;
	var = get_variable(FLG_WRITE_AZIMUTHAL_AVG); var->control->get_value(&flg);
	fout << "flg_write_azimuthal_averages\t";
	if (true == flg) fout << "1" << std::endl;
	else             fout << "0" << std::endl;
	var = get_variable(EXECUTION_MODE); var->control->get_value(&flg);
	fout << "# to specify EXECUTION MODE 1 ---> FIFO , 0----> Time correlation" << std::endl;
	fout << "EXECUTION_MODE\t";
	if (true == flg) fout << "1" << std::endl;
	else             fout << "0" << std::endl;
	var = get_variable(HARDWARE_SELECTION); var->control->get_value(&flg);
	fout << "# to specify EXECUTION MODE 1 ---> CPU , 0----> GPU" << std::endl;
	fout << "HARDWARE_SELECTION\t";
	if (true == flg) fout << "1" << std::endl;
	else             fout << "0" << std::endl;

	

	var = get_variable(NUMBER_THREAD_GPU); var->control->get_value(&value);
	fout << "NUMBER_THREAD_GPU\t" << value << std::endl;
	var = get_variable(NUMBER_THREAD_CPU); var->control->get_value(&value);
	fout << "NUMBER_THREAD_CPU\t" << value << std::endl;
	var = get_variable(RAM_CPU); var->control->get_value(&value);
	fout << "RAM_CPU\t" << value << std::endl;;

	var = get_variable(GRAPH_MODE); var->control->get_value(&flg);
	fout << "# to show graph 1 ---> show the graph, 0----> don't show" << std::endl;
	fout << "GRAPH_MODE\t";
	if (true == flg) fout << "1" << std::endl;
	else             fout << "0" << std::endl;

	var = get_variable(FLG_START); var->control->get_value(&flg);
	fout << "# if (start==1) the graphical user interface will not be opened" << std::endl;
	fout << "start\t";
	if (true == flg) fout << "1" << std::endl;
	else             fout << "0" << std::endl;

	fout.close();
	return true;
}
bool diffmicro_user_interface::list_images_in_file(std::string path, std::vector<std::string >& list_file) {
	INDEX i;
	bool flg_path = false;
	//	bool flg_pathtmp = false;
	bool flg_pathout = false;
	if (false == prep_file_list(path, list_file))
	{
		std::cerr << "invalid path_images : " << path << std::endl;
		return false;
	}
	else {
		flg_path = true;
		flg_pathout = true;
		
	}

	// VARIABLES CHECK AND ADJUSTMENTS
	if (0 != time.size() && (true == flg_time_file))
	{
		if (time.size() != file_list.size())
		{
			std::cerr << "file_time lenght do not mach number of files #file = " << file_list.size() << "\t#time = " << time.size() << std::endl;
			time.clear();
			//			flg_valid_image.clear();
			return false;
		}
	}
	else
	{
		time_step = (MY_REAL)(1.0);
		time.resize(file_list.size());
		for (i = 0; i < file_list.size(); ++i) time[i] = (MY_REAL)(i);
	}

	/*if (0 != flg_valid_image.size())
	{
		if (execution_mode == DIFFMICRO_MODE_TIMECORRELATION) {

			std::cerr << "Execution mode time correlation not compatible with valid image input" << std::endl;
			std::cerr << "Execution mode changed to fifo" << std::endl;
			execution_mode = DIFFMICRO_MODE_FIFO;
		}

		if (time.size() != flg_valid_image.size())
		{
			std::cerr << "file_time lenght do not mach number of files #file = " << file_list.size() << "\t#flg_valid_image = " << flg_valid_image.size() << std::endl;
			time.clear();
			flg_valid_image.clear();
			return false;
		}
	}
	else
	{*/
		flg_valid_image.resize(file_list.size());
		for (i = 0; i < flg_valid_image.size(); ++i) flg_valid_image[i] = true;
	//}

	if (0 == this->file_list.size())
	{
		std::cerr << "input folder appears to be empty : " << this->path_input /*this->path_temporary*/ << std::endl;
		return false;
	}

	if ((false == flg_path) || /*(false == flg_pathtmp) || */(false == flg_pathout))
	{
		std::cerr << "error : not enough information" << std::endl;
		return false;
	}

	// force consistency of images interval
	if ((0 == this->first_image) || (this->first_image > this->file_list.size())) this->first_image = 1;
	if ((0 == this->last_image) || (this->last_image > this->file_list.size())) this->last_image = (INDEX)(this->file_list.size());
	if (this->first_image >= this->last_image)
	{
		this->first_image = 1;
		this->last_image = (INDEX)(this->file_list.size());
	}

	// force consistency of averages parameters
	if (0 == this->n_pw_averages) this->n_pw_averages = (INDEX)(this->file_list.size());

	//New edit
	if (execution_mode == DIFFMICRO_MODE_TIMECORRELATION && this->n_pw_averages == (INDEX)(this->file_list.size())) {

		std::cerr << "Execution mode time correlation not compatible with arbitrary number of averages " << std::endl;
		std::cerr << "number of average set to number of images" << std::endl;
		this->n_pw_averages = (INDEX)(this->file_list.size());
	}

	if (0 == this->dist_max) this->dist_max = (INDEX)(this->file_list.size());
	//---------------------------------------------------------------------------------------

	// constructing paths and file names
	std::string filename;
	std::vector< std::string> path_part;
	INDEX filename_part;
	if ('\\' != path_power_spectra[path_power_spectra.size() - 1]) path_power_spectra = path_power_spectra + "\\";
	//if ('\\' != path_temporary[path_temporary.size() - 1]) path_temporary = path_temporary + "\\";
	if ('\\' != path_input[path_input.size() - 1]) path_input = path_input + "\\";
	split('\\', path, path_part);
	filename_part = (INDEX)(path_part.size() - 1);
	if (0 == path_part[filename_part].size()) --filename_part;
	filename = path_part[filename_part];

	power_spectra_filename = path_power_spectra + filename + "_spectrum_";
	azimuthal_avg_filename = path_power_spectra + filename + "_dinamics.mat";
	images_avg_filename = path_power_spectra + filename + "_average.vet";
	time_sequence_filename = path_power_spectra + filename + "_time_sequence.txt";

	// cropping file names vector as request by the user
	if (this->file_list.size() != this->last_image)
		file_list.resize(this->last_image);

	if (1 != this->first_image)
	{
		std::vector< std::string>::iterator it;
		it = file_list.begin();
		for (INDEX i = 1; i < this->first_image; ++i) ++it;
		file_list.erase(file_list.begin(), it);
	}

	if (1 != this->step_of_time)
	{
		int nb;
		lldiv_t nb_img;
		nb_img = std::div((long long)(this->file_list.size()), (long long)(useri.step_of_time));
		if (nb_img.rem == 0) {
			nb = (INDEX)(nb_img.quot);
		}
		else {
			nb = (INDEX)(nb_img.quot) + 1;
		}
		for (INDEX i = 0; i < nb; ++i) {
			file_list[i] = file_list[i * step_of_time];
		}
		file_list.resize(nb);
	}

	log_filename = path_power_spectra + filename + "_log.txt";

	if ((0 != time.size()) && (0.99999 < time_step) && (true == flg_time_file))
	{
		if (false == adjust_time_spacing(file_list, time, time_step, flg_valid_image))
		{
			std::cerr << "file_time was defined, but times values are not adeguate" << std::endl;
			return false;
		}
	}

	//---------------------------------------------------------------------------------------
	// PRINTING TO STANDARD OUTPUT
	std::cout << "first_image : " << this->file_list[0] << std::endl;
	std::cout << "last_image : " << this->file_list[this->file_list.size() - 1] << std::endl;
	std::cout << "number_of_averages : " << this->n_pw_averages << std::endl;
	std::cout << "delay_max : " << this->dist_max << std::endl;
	std::cout << "path output : " << this->path_power_spectra << std::endl;

}
bool diffmicro_user_interface::load(std::string file_ui)
{
	INDEX i;
	std::fstream fin;
	std::string line;
	std::string path;

	std::vector< std::string> entry, list_dir, list_file_tmp, entry1;

	//-----------------------------------------------------
	// all necessary variables have their flag.
	// if the variable will be correctly acquired the flag
	// will be set to true
	bool flg_path = false;
//	bool flg_pathtmp = false;
	bool flg_pathout = false;
	
	this->time_step = (MY_REAL)(0.0);

	//-----------------------------------------------------
	// READING TAGS

	fin.open(file_ui.c_str(), std::ios::in);
	if (false == fin.good())
	{
		std::cerr << "error attempting to read : " << file_ui << std::endl;
		return false;
	}

	std::getline(fin, line);
	while (true == fin.good())
	{
		split('\t', line, entry);

		if (0 == strcmp(entry[0].c_str(), "path_images"))
		{
			if (entry.size() != 2)
			{
				std::cerr << "invalid path_images" << std::endl;
				return false;
			}

			path = entry[1];
			this->path_input = path;


//			if (false == list_directory(path, list_dir, list_file_tmp))
			/*if (false == prep_file_list(this->path_input, this->file_list))
			{
				std::cerr << "invalid path_images : " << path << std::endl;
				return false;
			}*/
			flg_path = true;
		}

		if (0 == strcmp(entry[0].c_str(), "series"))
		{

			//path = entry[1];
			split(',', entry[1], this->series);

			//this->path_input = path;


			//			if (false == list_directory(path, list_dir, list_file_tmp))
						/*if (false == prep_file_list(this->path_input, this->file_list))
						{
							std::cerr << "invalid path_images : " << path << std::endl;
							return false;
						}*/
			//flg_path = true;
		}

		/*if (0 == strcmp(entry[0].c_str(), "file_times"))
		{
			if (entry.size() != 2)
			{
				std::cerr << "invalid file_times" << std::endl;
				return false;
			}

			this->file_times = entry[1];
			if (false == load_one_column_file(file_times, this->time))
			{
				std::cerr << "invalid file_times : " << this->file_times << std::endl;
				return false;
			}
		}*/

		if (0 == strcmp(entry[0].c_str(), "file_valid_images"))
		{
			if (entry.size() != 2)
			{
				std::cerr << "invalid file_valid_images" << std::endl;
				return false;
			}

			this->file_valid_images = entry[1];
			if (false == load_one_column_file(file_valid_images, this->flg_valid_image))
			{
				std::cerr << "invalid file_valid_images : " << this->file_valid_images << std::endl;
				return false;
			}
		}

		if (0 == strcmp(entry[0].c_str(), "images_interval") && (entry.size() == 3))
		{
			this->first_image = convert_string<INDEX>(entry[1]);
			this->last_image = convert_string<INDEX>(entry[2]);
		}

		if (0 == strcmp(entry[0].c_str(), "Version") && (entry.size() == 2))
			this->version = convert_string<INDEX>(entry[1]);

		if (0 == strcmp(entry[0].c_str(), "step_of_time") && (entry.size() == 2))
			this->step_of_time = convert_string<INDEX>(entry[1]);

		if (0 == strcmp(entry[0].c_str(), "binary") && (entry.size() == 2))
			this->binary = convert_string<INDEX>(entry[1]);

		/*if (0 == strcmp(entry[0].c_str(), "number_of_series") && (entry.size() == 2))
			this->nb_of_series = convert_string<INDEX>(entry[1]);*/

		/*if (0 == strcmp(entry[0].c_str(), "size_image") && (entry.size() == 2))
			this->size_image = convert_string<INDEX>(entry[1]);*/

		if (0 == strcmp(entry[0].c_str(), "number_of_averages") && (entry.size() == 2))
			this->n_pw_averages = convert_string<INDEX>(entry[1]);

		if (0 == strcmp(entry[0].c_str(), "delay_max") && (entry.size() == 2))
			this->dist_max = convert_string<INDEX>(entry[1]);

		if (0 == strcmp(entry[0].c_str(), "frequency_max") && (entry.size() == 2))
			this->frequency_max = convert_string<INDEX>(entry[1]);

		/*if (0 == strcmp(entry[0].c_str(), "time_step") && (entry.size() == 2))
			this->time_step = convert_string<INDEX>(entry[1]);*/

		if (0 == strcmp(entry[0].c_str(), "path_output"))
		{
			if (entry.size() != 2)
			{
				std::cerr << "invalid path_output" << std::endl;
				return false;
			}

			this->path_power_spectra = entry[1];
			if (false == list_directory(this->path_power_spectra, list_dir, list_file_tmp))
			{
				std::cerr << "invalid path_output : " << this->path_power_spectra << std::endl;
				return false;
			}
			flg_pathout = true;
		}

		if (0 == strcmp(entry[0].c_str(), "flg_write_images_average") && (entry.size() == 2))
			this->flg_write_images_avg = convert_string<bool>(entry[1]);

		if (0 == strcmp(entry[0].c_str(), "flg_write_power_spectra") && (entry.size() == 2))
			this->flg_write_power_spectra = convert_string<bool>(entry[1]);

		/*if (0 == strcmp(entry[0].c_str(), "flg_time_file") && (entry.size() == 2))
			this->flg_time_file = convert_string<bool>(entry[1]);*/

		if (0 == strcmp(entry[0].c_str(), "flg_write_azimuthal_averages") && (entry.size() == 2))
			this->flg_write_azimuthal_avg = convert_string<bool>(entry[1]);

		if (0 == strcmp(entry[0].c_str(), "flg_file_valid_image") && (entry.size() == 2))
			this->flg_valid_image_file = convert_string<bool>(entry[1]);

		if (0 == strcmp(entry[0].c_str(), "EXECUTION_MODE") && (entry.size() == 2))
			this->flg_execution_mode = convert_string<bool>(entry[1]);

		if (0 == strcmp(entry[0].c_str(), "HARDWARE_SELECTION") && (entry.size() == 2))
			this->flg_hardware_selection = convert_string<bool>(entry[1]);

		if (0 == strcmp(entry[0].c_str(), "NUMBER_THREAD_GPU") && (entry.size() == 2))
			this->nthread_gpu = convert_string<INDEX>(entry[1]);

		if (0 == strcmp(entry[0].c_str(), "NUMBER_THREAD_CPU") && (entry.size() == 2))
			this->nthread = convert_string<INDEX>(entry[1]);

		if (0 == strcmp(entry[0].c_str(), "RAM_CPU") && (entry.size() == 2))
			this->RAM = convert_string<INDEX>(entry[1]);

		if (0 == strcmp(entry[0].c_str(), "GRAPH_MODE") && (entry.size() == 2))
			this->flg_graph_mode = convert_string<bool>(entry[1]);

		if ((0 == strcmp(entry[0].c_str(), "start")) && (entry.size() == 2))
			this->flg_start = convert_string<bool>(entry[1]);



		std::getline(fin, line);
	}
	//std::cerr <<"flg_write_power_spectra"<<flg_write_power_spectra<<std::endl;
	//---------------------------------------------------------------------------------------


	return flg_start;
}