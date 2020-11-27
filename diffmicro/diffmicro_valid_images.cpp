/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 2016
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

bool diffmicro_user_interface::adjust_time_spacing(std::vector<std::string> &file_list, std::vector<MY_REAL> time_input, MY_REAL time_step, std::vector<bool> &flg_valid_image)
{
	INDEX i,j,dim,dimnew;
	std::vector<INDEX> time_sequence;
	std::vector<std::string> file_list_tmp;
	std::vector<bool> flgvalidimage;

	dim = file_list.size();
	time_sequence.resize(dim);

	if (0.000000001 > time_step)
	{
		std::cerr << "error: time step smaller than 10^(-9)" << std::endl;
		return false;
	}

	for (i = 1; i < dim; ++i)
	{
		if (time_input[i - 1] >= time_input[i])
		{
			std::cerr << "error: input time sequence is not sorted" << std::endl;
			return false;
		}
	}

	for (i = 0; i < dim; ++i) time_input[i] -= time_input[0];

	for (i = 0; i < dim; ++i) time_sequence[i] = (INDEX)(std::floor(time_input[i]/time_step+0.5));

	for (i = 1; i < dim; ++i)
	{
		if (time_sequence[i - 1] == time_sequence[i])
		{
			std::cerr << "error: requested time step is too large" << std::endl;
			return false;
		}
	}

	file_list_tmp = file_list;
	file_list.clear();
	
	dimnew = time_sequence[dim - 1]+1;
	file_list.resize(dimnew);
	flgvalidimage = flg_valid_image;
	flg_valid_image.resize(dimnew);

	for (i = 0; i < dimnew; ++i) flg_valid_image[i] = false;

//	for (i = 0; i < dim; ++i) std::cerr << time_sequence[i] <<"\t"<< flgvalidimage[i] << "\t" << file_list_tmp[i] << std::endl;

	for (i = 0; i < dim; ++i)
	{
		file_list[time_sequence[i]] = file_list_tmp[i];
		flg_valid_image[time_sequence[i]] = flgvalidimage[i];
	}

//	std::cerr << "-------------------------------------" << std::endl;
//	for (i = 0; i < dimnew; ++i) std::cerr << flg_valid_image[i] << "\t" << file_list[i] << std::endl;

	return true;
}

