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
#include "minmax_trackbar_opencv.h"


minmax_trackbar::minmax_trackbar(int _id, std::string _name) : trackbar_opencv(_id, _name)
{

	type = CTRL_MINMAX_TRACKBAR;

	cv::Scalar col(20, 20, 20);

	color.push_back(col);
	col[0] = 200; col[1] = 0; col[2] = 0;
	color.push_back(col);
	col[0] = 0; col[1] = 200; col[2] = 0;
	color.push_back(col);
	col[0] = 0; col[1] = 0; col[2] = 200;
	color.push_back(col);

	value.push_back(3);
	value.push_back(0);
	value.push_back(1);
	value.push_back(2);

	selected_handle = -1;

	deselect();
	enable();
}

minmax_trackbar::~minmax_trackbar()
{
	trackbar_opencv::~trackbar_opencv();
}

minmax_trackbar& minmax_trackbar::operator=(const minmax_trackbar& ctrl)
{
	trackbar_opencv::operator=(ctrl);
	this->selected_handle = ctrl.selected_handle;
	return *this;
}

void minmax_trackbar::select()
{
	selected = true;
}
void minmax_trackbar::deselect()
{
	selected = false;
}
void minmax_trackbar::enable()
{
	enabled = true;
}
void minmax_trackbar::disable()
{
	enabled = false;
}

void minmax_trackbar::get_value(void *out)
{
	int i;

	for (i = 0; i < 4; ++i)
	((int*)(out))[i] = value[i];

}

void minmax_trackbar::set_value(void* _newvalue)
{
	int *new_value;

	new_value = ((int*)(_newvalue));

	set_maximum_value(new_value[0]);
		
	if (0 <= new_value[1] && new_value[1] < new_value[2] && new_value[2]<new_value[3] && new_value[3] <= value[0])
	{
		value[1] = new_value[1];
		value[3] = new_value[3];
		value[2] = (value[3] - value[1]) / 2 + value[1];
	}

}

void minmax_trackbar::set_maximum_value(int _new_max)
{
	if (_new_max == value[0]) return;

	if (1 <= _new_max) value[0] = _new_max;

	if (value[1] > value[0])  value[1] = value[0];

}

int minmax_trackbar::handle(float pos[])
{
	int ret = -1;

	if (x[0] <= pos[0] && pos[0]<x[2] && x[1] <= pos[1] && pos[1]<x[3]) ret = 2;

	if (2 == ret)
	{
		int i,j;

		j = (int)(  ((pos[0] - x[0]) / (x[2] - x[0]))*(float)(value[0]+1)  );
		for (i = 1; i < 4; ++i)
		{
			if (value[i] == j)
			{
				selected_handle = i;
				return i;
			}
		}
		
	}

	return ret;
}

void minmax_trackbar::update(float pos[])
{
	int newvalue[4];
	int i,intpos;
	float p;
	float span;
	span = x[2] - x[0];
	p = pos[0] - x[0];
	intpos = (int)((p / span)*(float)(value[0]+1));

	for (i = 0; i < 4; ++i) newvalue[i] = value[i];

	switch (selected_handle)
	{
	case 1:
		newvalue[1] = intpos;
		break;
	case 2:
		intpos = intpos - newvalue[2];
		for (i = 1; i < 4; ++i) newvalue[i] += intpos;
		break;
	case 3:
		newvalue[3] = intpos;
		break;
	default:

		break;
	}

	
	set_value(newvalue);
}

void minmax_trackbar::onclick(float pos[])
{
	this->update(pos);
}
void minmax_trackbar::onmove(float pos[])
{
	this->update(pos);
}

void minmax_trackbar::onrelease(float pos[])
{
	selected_handle = -1;
}


