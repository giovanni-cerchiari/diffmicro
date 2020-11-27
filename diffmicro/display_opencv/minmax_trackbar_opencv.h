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

#ifndef _MINMAX_TRACKBAR_OPENCV_H_
#define _MINMAX_TRACKBAR_OPENCV_H_

#include "trackbar_opencv.h"

/*!
This is a special trackbar that changes the dynamic range of an image.
The dynamic range have a maximum and a minimum value.
With this trackbar it is possible to modify maximum, minimum and average value within an interval
See trackbar_opencv and control_opencv for further details
*/

class minmax_trackbar : public trackbar_opencv
{
public:
	minmax_trackbar(int _id = -1, std::string _name = "");
	minmax_trackbar(const minmax_trackbar& ctrl) { *this = ctrl; }
	~minmax_trackbar();

	void select();
	void deselect();
	void enable();
	void disable();

	virtual int handle(float pos[]);
	virtual minmax_trackbar& operator=(const minmax_trackbar& ctrl);

	void get_value(void *out);
	void set_value(void *in);

	void onclick(float pos[]);
	void onmove(float pos[]);
	void onrelease(float pos[]);

	void update(float pos[]);
	void set_maximum_value(int _new_max);

protected:
	/*!
	The handles are in this case 3
	1) minimum
	2) middle value
	3) maximum
	*/
	int selected_handle;

};


#endif
