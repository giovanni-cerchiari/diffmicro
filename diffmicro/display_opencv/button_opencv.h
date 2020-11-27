
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

#ifndef _BUTTON_OPENCV_H_
#define _BUTTON_OPENCV_H_

#include "control_opencv.h"

/*!
A button is a user interface object with two states.
Pressing the button switches the internal state.
See control_opencv comments for further details
*/
class button_opencv : public control_opencv
{
public:
	button_opencv(int _id = -1, std::string _name = "");
	~button_opencv();

	void select();
	void deselect();
	void enable();
	void disable();

	void draw(cv::Mat &img);
	int handle(float pos[]);
	button_opencv& operator=(const button_opencv &ctrl);
	control_opencv& operator=(const control_opencv &ctrl){ *this = ctrl; return *this; }

	void get_value(void *out);
	void set_value(void *in);

	void onclick(float pos[]);
	void onmove(float pos[]);
	void onrelease(float pos[]);

	bool state;

	unsigned char *color_true;
	unsigned char *color_false;
	cv::Scalar color_label_true;
	cv::Scalar color_label_false;

	unsigned char color_true_enabled[3];
	unsigned char color_false_enabled[3];
//	cv::Scalar *color_label_true_enabled;
//	cv::Scalar *color_label_false_enabled;

	unsigned char color_true_disabled[3];
	unsigned char color_false_disabled[3];
//	cv::Scalar *color_label_true_disabled;
//	cv::Scalar *color_label_false_disabled;

	std::string label_true;
	std::string label_false;

};

#endif
