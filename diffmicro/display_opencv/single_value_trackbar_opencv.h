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

#ifndef _SINGLE_VALUE_TRACKBAR_OPENCV_H_
#define _SINGLE_VALUE_TRACKBAR_OPENCV_H_

#include "trackbar_opencv.h"

class single_value_trackbar : public trackbar_opencv
{
public:
	single_value_trackbar(int _id = -1, std::string _name = "");
	single_value_trackbar(const single_value_trackbar& ctrl) { *this = ctrl; }
	~single_value_trackbar();

	virtual int handle(float pos[]);
	virtual single_value_trackbar& operator=(const single_value_trackbar& ctrl);

	void get_value(void *out);
	void set_value(void *in);

	void onclick(float pos[]);
	void onmove(float pos[]);
	void onrelease(float pos[]);

	void update(float pos[]);
	void set_maximum_value(int _new_max);

};



#endif