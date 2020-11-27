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

#ifndef _TEXTBOX_OPENCV_H_
#define _TEXTBOX_OPENCV_H_

#include "control_opencv.h"

/*! This class is used for user interface. Control textbox may be used to read and write on a text string.*/

class textbox_opencv : public control_opencv
{
public:
	textbox_opencv(int _id = -1, std::string _name = "");
	~textbox_opencv();

	void select();
	void deselect();
	void enable();
	void disable();

	void draw(cv::Mat &img);
	int handle(float pos[]);
	textbox_opencv& operator=(const textbox_opencv &ctrl);
	control_opencv& operator=(const control_opencv &ctrl){ *this = ctrl; return *this; }

	void get_value(void *out);
	void set_value(void *in);

	void onclick(float pos[]);
	void onmove(float pos[]);
	void onrelease(float pos[]);

	std::string text;
//	unsigned int cursor;
	bool selected;

	cv::Scalar *color_background;
	cv::Scalar *color_label;
	cv::Scalar color_label_normal;
	cv::Scalar color_label_selected;
	cv::Scalar color_background_enabled;
	cv::Scalar color_background_disabled;
};

#endif

