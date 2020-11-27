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

#ifndef _TRACKBAR_OPENCV_H_
#define _TRACKBAR_OPENCV_H_

#include "control_opencv.h"

/*!
A trackbar is a user interface object that can have integer value between 0 and value[0]
The values assumed by the trackbar are store in value vector.
Each value has its handle to allow for separate manipulation.
See control_opencv for further details.
*/
class trackbar_opencv : public control_opencv
{
public:
	trackbar_opencv(int _id = -1, std::string _name = "");
	trackbar_opencv(const trackbar_opencv& ctrl) { *this = ctrl; }
	~trackbar_opencv();

	virtual void draw(cv::Mat &img);
	virtual int handle(float pos[]) = 0;
	virtual control_opencv& operator=(const control_opencv &ctrl){ *this = ctrl; return *this; }
	virtual trackbar_opencv& operator=(const trackbar_opencv& ctrl);

	virtual void get_value(void *out) = 0;
	virtual void set_value(void *in) = 0;

	virtual void onclick(float pos[]) = 0;
	virtual void onmove(float pos[]) = 0;
	virtual void onrelease(float pos[]) = 0;

	/*!
	maximum value is stored as value[0]
	*/
	std::vector<int> value;
	/*!
	background color is stored as color[0]
	*/
	std::vector<cv::Scalar> color;

};






#endif