/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 12/2015

implemented with opencv v 3.0
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
#include "trackbar_opencv.h"


trackbar_opencv::trackbar_opencv(int _id, std::string _name) : control_opencv(_id, _name)
{
	x = new float[4];
}

trackbar_opencv::~trackbar_opencv()
{
	control_opencv::~control_opencv();
}

void trackbar_opencv::draw(cv::Mat &mat)
{
	cv::Point begin, end;
	unsigned int i, dim;
	float step;

	begin.x = (int)(x[0] * (float)(mat.cols));
	begin.y = (int)(x[1] * (float)(mat.rows));
	end.x = (int)(x[2] * (float)(mat.cols));
	end.y = (int)(x[3] * (float)(mat.rows));
	cv::rectangle(mat, begin, end, color[0], CV_FILLED, 8, 0);
	
	step = (x[2] - x[0]) / (float)(value[0] + 1);
	dim = value.size();
	for (i = 1; i < dim; ++i)
	{
		begin.x = (int)(((float)(value[i]    )*step + x[0]) * (float)(mat.cols));
		end.x   = (int)(((float)(value[i] + 1)*step + x[0]) * (float)(mat.cols));
		cv::rectangle(mat, begin, end, color[i], CV_FILLED, 8, 0);
	}
	

}

trackbar_opencv& trackbar_opencv::operator=(const trackbar_opencv& ctrl)
{
	
	this->name = ctrl.name;
	this->id = ctrl.id;
	this->type = ctrl.type;
	this->name = ctrl.name;

	this->value = ctrl.value;
	this->color = ctrl.color;
	
 return *this;
}

