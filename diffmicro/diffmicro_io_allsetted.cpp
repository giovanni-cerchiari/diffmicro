/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 2016
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
#include "diffmicro_io.h"
#include "dirent.h"

int diffmicro_user_interface::check_filename_set_box(int check_box_id, int path_id, int filename_id)
{
	int ret;
	std::fstream fin;
	bool flg;
	std::string path;
	std::string filename;
	ui_variable *varstring;
	ui_variable *check;

	varstring = get_variable(path_id);
	varstring->control->get_value(&path);
	varstring = get_variable(filename_id);
	varstring->control->get_value(&filename);
	check = get_variable(check_box_id);

//	if(0<path.size())if ('\\' != path[path.size() - 1]) path = path + "\\";
	filename = path + filename;
	fin.open(filename.c_str(), std::ios::in);
	if (fin.good())
	{
		ret = 1;
		flg = true;
		fin.close();
	}
	else
	{
		ret = 0;
		flg = false;
	}

	check->control->set_value(&flg);

	return ret;
}

int diffmicro_user_interface::check_positive_integer_set_box(int check_box_id, int string_id)
{
	int ret;
	int num;
	bool flg;
	std::string str;
	ui_variable *varnum;
	ui_variable *check;

	varnum = get_variable(string_id);
	check = get_variable(check_box_id);

	varnum->control->get_value(&str);

	if (true == is_int(str))
	{
		num = convert_string<int>(str);
		if (0 < num)
		{
			ret = 1;
			flg = true;
		}
		else
		{
			ret = 0;
			flg = false;
		}
	}
	else
	{
		ret = 0;
		flg = false;
	}
	check->control->set_value(&flg);
	return ret;
}

int diffmicro_user_interface::check_path_set_box(int check_box_id, int path_id)
{
	int ret;
	bool flg;
	std::string str;
	ui_variable *path;
	ui_variable *check;
	DIR *directory;
	std::fstream fin;

	path = get_variable(path_id);
	check = get_variable(check_box_id);

	path->control->get_value(&str);
	//std::cout <<checkbox_id<<"->"<< str << std::endl;
	
	directory = dirent_opendir(str.c_str());
	if (NULL == directory)
	{
		fin.open(str.c_str(), std::ios::in);
		if (!fin.good())
		{
			flg = false;
			ret = 0;
		}
		else
		{
			flg = true;
			ret = 1;
		}
		fin.close();
	}
	else
	{
		flg = true;
		ret = 1;
		dirent_closedir(directory);
	}
	check->control->set_value(&flg);

	return flg;
}

int diffmicro_user_interface::check_path_button_set_box(int check_box_id, int button_id, int path_id)
{
	bool flg;
	ui_variable *butt;

	butt = get_variable(button_id);
	butt->control->get_value(&flg);

	if (true == flg) return check_path_set_box(check_box_id, path_id);
	else
	{
		flg = true;
		butt = get_variable(check_box_id);
		butt->control->set_value(&flg);
		return 1;
	}
}


bool diffmicro_user_interface::all_setted()
{
	bool flg;
	int status = 1;
	int status_time;
	ui_variable *var;

	status *= check_path_set_box(CHECKBOX_PATH_IMAGES, PATH_IMAGES);
	status *= check_positive_integer_set_box(CHECKBOX_FIRST_IMAGE, FIRST_IMAGE);
	status *= check_positive_integer_set_box(CHECKBOX_LAST_IMAGE, LAST_IMAGE);
	status *= check_path_set_box(CHECKBOX_PATH_POWER_SPECTRA, PATH_POWER_SPECTRA);
		
	status *= check_positive_integer_set_box(CHECKBOX_DIST_MAX, DIST_MAX);
	status *= check_positive_integer_set_box(CHECKBOX_N_PW_AVERAGES, N_PW_AVERAGES);
	//status *= check_positive_integer_set_box(CHECKBOX_VERSION, VERSION);

	status *= check_positive_integer_set_box(CHECKBOX_FREQUENCY_MAX, FREQUENCY_MAX);

	flg = true;
	check_path_button_set_box(CHECKBOX_POWER_SPECTRA_FILENAME, FLG_WRITE_POWER_SPECTRA, POWER_SPECTRA_FILENAME);
	var = get_variable(CHECKBOX_POWER_SPECTRA_FILENAME); var->control->set_value(&flg);
	check_path_button_set_box(CHECKBOX_AZIMUTHAL_AVG_FILENAME, FLG_WRITE_AZIMUTHAL_AVG, AZIMUTHAL_AVG_FILENAME);
	var = get_variable(CHECKBOX_AZIMUTHAL_AVG_FILENAME); var->control->set_value(&flg);

	var = get_variable(CHECKBOX_EXECUTION_MODE); var->control->set_value(&flg);

	var = get_variable(CHECKBOX_GRAPH_MODE); var->control->set_value(&flg);
	//should check for graphic card availability 
	var = get_variable(CHECKBOX_HARDWARE_SELECTION); var->control->set_value(&flg);

	var = get_variable(CHECKBOX_NUMBER_THREAD_GPU); var->control->set_value(&flg);
	status *= check_positive_integer_set_box(CHECKBOX_NUMBER_THREAD_GPU, NUMBER_THREAD_GPU);
	status *= check_positive_integer_set_box(CHECKBOX_NUMBER_THREAD_CPU, NUMBER_THREAD_CPU);
	//status *= check_positive_integer_set_box(CHECKBOX_RAM_CPU, RAM_CPU);

	//status_time = check_path_button_set_box(CHECKBOX_TIME_FILENAME, FLG_TIME_FILENAME, TIME_FILENAME);

	/*var = get_variable(FLG_TIME_FILENAME); var->control->get_value(&flg);
	if (true == flg) status *= check_positive_integer_set_box(CHECKBOX_TIME_STEP, TIME_STEP);
	else
	{
		flg = true;
		var = get_variable(CHECKBOX_TIME_STEP);
		var->control->set_value(&flg);
	}*/

	/*status *= status_time;*/
	status *= check_path_button_set_box(CHECKBOX_VALID_IMAGE_FILENAME, FLG_VALID_IMAGE_FILENAME, VALID_IMAGE_FILENAME);

	flg = true;
	var = get_variable(FLG_START); var->control->get_value(&flg);
	if (true == flg) status *= 1;
	else             status *= 0;

	if (0 == status)
	{
		flg_start = false;
		set_bool(FLG_START, flg_start);
		return false;
	}
	else
	{
		return true;
	}
	
}

