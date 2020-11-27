
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
#include "button_opencv.h"


button_opencv::button_opencv(int _id, std::string _name) :control_opencv(_id, _name)
{
	int i;

	type = CTRL_BUTTON;
	x = new float[4];

	for (i = 0; i < 4; ++i) x[i] = 0.0;

	state = false;

	for (i = 0; i < 3; ++i) color_true_enabled[i] = 0;
	for (i = 0; i < 3; ++i) color_false_enabled[i] = 0;
	for (i = 0; i < 3; ++i) color_true_disabled[i] = 100;
	for (i = 0; i < 3; ++i) color_false_disabled[i] = 100;
	
	color_true_enabled[1] = 200; color_true_disabled[1] = 200;
	color_false_enabled[2] = 200; color_false_disabled[2] = 200;
//	color_false[0] = 0; color_false[1] = 00; color_false[2] = 200;
	color_label_true = cv::Scalar(0,0,0);
	color_label_false = cv::Scalar(0,0,0);

	label_false = label_true = name;

	deselect();
	enable();
}

button_opencv::~button_opencv()
{
	control_opencv::~control_opencv();
}


void button_opencv::select()
{
	selected = true;
}
void button_opencv::deselect()
{
	selected = false;
}
void button_opencv::enable()
{
	color_false = color_false_enabled;
	color_true = color_true_enabled;
	enabled = true;
}
void button_opencv::disable()
{
	color_false = color_false_disabled;
	color_true = color_true_disabled;
	enabled = false;
}

void button_opencv::get_value(void *out)
{
	*((bool*)(out)) = this->state;
}
void button_opencv::set_value(void *in)
{
	this->state = *((bool*)(in));
}

void button_opencv::draw(cv::Mat &mat)
{
	unsigned int i, j, jj, ii, k;
	unsigned int xx[4];
	unsigned char *col;
	unsigned char *matdataptr;
	std::string *label;
	cv::Scalar *colbut;

	for (i = 0; i < 2; ++i)
	{
		xx[2 * i    ] = (unsigned int)(x[2 * i    ] * mat.cols);
		xx[2 * i + 1] = (unsigned int)(x[2 * i + 1] * mat.rows);
	}

	if (true == state)
	{
		col = color_true;
		colbut = &color_label_true;
		label = &label_true;
	}
	else
	{
		col = color_false;
		colbut = &color_label_false;
		label = &label_false;
	}

	for (j = xx[1]; j < xx[3]; ++j)
	{
		jj = j  * mat.step[0];
		for (i = xx[0]; i < xx[2]; ++i)
		{
			ii = jj + i * mat.step[1];
			
			for (k = 0; k < 3; ++k)
			{
				matdataptr = (unsigned char*)(&(mat.data[ii + k]));
				matdataptr[0] = col[k];
			}
		}
	}

	if (0 < label->size())
	{
		cv::Point point(xx[0] + (xx[2] - xx[0]) / 10, xx[1] + (xx[3] - xx[1]) / 2);
		cv::putText(mat, *label, point, cv::FONT_HERSHEY_SIMPLEX, 0.6, *colbut, 2);
	}
}

int button_opencv::handle(float pos[])
{
	if (x[0]>pos[0] || x[2]<pos[0] || x[1]>pos[1] || x[3]<pos[1]) return -1;
	else                                                          return 0;
}

void button_opencv::onclick(float pos[])
{
	if ((0 == this->handle(pos)) && (true == this->enabled))
	{
		this->state = !this->state;
	}
}

void button_opencv::onmove(float pos[])
{}

void button_opencv::onrelease(float pos[])
{}

button_opencv& button_opencv::operator=(const button_opencv &ctrl)
{
	int i;
	this->id = ctrl.id;
	this->name = ctrl.name;
	
	for (i = 0; i < 4; ++i)	this->x[i] = ctrl.x[i];

	for (i = 0; i < 3; ++i)
	{
		this->color_true[i] = ctrl.color_true[i];
		this->color_false[i] = ctrl.color_false[i];
	}

	this->color_label_true = ctrl.color_label_true;
	this->color_label_false = ctrl.color_label_false;

	this->state = ctrl.state;

	return *this;
}

