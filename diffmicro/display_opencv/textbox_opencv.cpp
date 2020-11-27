/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 02/2016

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
#include "textbox_opencv.h"
#include "figure_opencv.h"

textbox_opencv::textbox_opencv(int _id, std::string _name) :control_opencv(_id, _name)
{
	unsigned int i;
	type = CTRL_TEXTBOX;
	x = new float[4];

	for (i = 0; i < 4; ++i) x[i] = 0.0;

	color_label_normal = cv::Scalar(255, 255, 255);
	color_label_selected = cv::Scalar(0, 200, 0);
	color_background_disabled = cv::Scalar(100,100,100);
	color_background_enabled = cv::Scalar(20, 20, 20);

	//cursor = 0;

	deselect();
	enable();
}
textbox_opencv::~textbox_opencv()
{
	control_opencv::~control_opencv();
}

void textbox_opencv::get_value(void *out)
{
	*((std::string*)(out)) = text;
}

void textbox_opencv::set_value(void *in)
{
	if (true == enabled)
		text = *((std::string*)(in));
}

void textbox_opencv::select()
{
	selected = true;
	color_label = &(color_label_selected);
}

void textbox_opencv::deselect()
{
	selected = false;
	color_label = &(color_label_normal);
}

void textbox_opencv::enable()
{
	enabled = true;
	color_background = &color_background_enabled;
}
void textbox_opencv::disable()
{
	enabled = false;
	color_background = &color_background_disabled;
}

void textbox_opencv::onclick(float pos[])
{
}

void textbox_opencv::onmove(float pos[])
{
}

void textbox_opencv::onrelease(float pos[])
{
}

void textbox_opencv::draw(cv::Mat &mat)
{
	unsigned int i, j, jj, ii, k;
	cv::Point xx[2];
	unsigned char *col;
	cv::Scalar *colbut;

	xx[0].x = (unsigned int)(x[0] * mat.cols);
	xx[0].y = (unsigned int)(x[1] * mat.rows);
	xx[1].x = (unsigned int)(x[2] * mat.cols);
	xx[1].y = (unsigned int)(x[3] * mat.rows);

	cv::rectangle(mat, xx[0], xx[1], *color_background, -1);
	if (0 < text.size())
	{
		cv::Point point(xx[0].x + (xx[1].x - xx[0].x) / 20, xx[0].y + (xx[1].y - xx[0].y) / 2);
		cv::putText(mat, text, point, cv::FONT_HERSHEY_SIMPLEX, 0.6, *color_label, 2);
	}
}

int textbox_opencv::handle(float pos[])
{
	if (x[0]>pos[0] || x[2]<pos[0] || x[1]>pos[1] || x[3]<pos[1]) return -1;
	else                                                          return 0;
}