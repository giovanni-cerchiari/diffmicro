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
#include "single_value_trackbar_opencv.h"


single_value_trackbar::single_value_trackbar(int _id, std::string _name) : trackbar_opencv(_id, _name)
{
	int i;

	type = CTRL_SINGLE_TRACKBAR;

	cv::Scalar col(20, 20, 20);

	color.push_back(col);
	for (i = 0; i<3; ++i) col[i] = 200;
	color.push_back(col);

	value.push_back(1);
	value.push_back(0);
}

single_value_trackbar::~single_value_trackbar()
{
	trackbar_opencv::~trackbar_opencv();
}

single_value_trackbar& single_value_trackbar::operator=(const single_value_trackbar& ctrl)
{
	trackbar_opencv::operator=(ctrl);
	return *this;
}

void single_value_trackbar::get_value(void *out)
{
	((int*)(out))[0] = value[0];
	((int*)(out))[1] = value[1];
}

void single_value_trackbar::set_value(void* _newvalue)
{
	int new_value = ((int*)(_newvalue))[1];

	set_maximum_value(((int*)(_newvalue))[0]);

	if (0 <= new_value)
	{
		if (new_value <= value[0]) value[1] = new_value;
		else                        value[1] = value[0];
	}
	else
		value[1] = 0;
}

void single_value_trackbar::set_maximum_value(int _new_max)
{
	if (_new_max == value[0]) return;

	if (1 <= _new_max) value[0] = _new_max;

	if (value[1] > value[0])  value[1] = value[0];

}

int single_value_trackbar::handle(float pos[])
{
	if (x[0]>pos[0] || x[2]<pos[0] || x[1]>pos[1] || x[3]<pos[1]) return -1;
	else                                                          return 1;
}

void single_value_trackbar::update(float pos[])
{
	int newvalue[2];
	float p;
	float span;
	span = x[2] - x[0];
	p = pos[0] - x[0];

	newvalue[0] = value[0];
	newvalue[1] = (int)((p / span)*(float)(value[0]+1));
	set_value(newvalue);
}

void single_value_trackbar::onclick(float pos[])
{
	this->update(pos);
}
void single_value_trackbar::onmove(float pos[])
{
	this->update(pos);
}

void single_value_trackbar::onrelease(float pos[])
{}



