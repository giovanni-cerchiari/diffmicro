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
#ifndef _WINDOW_CONTROL_OPENCV_H_
#define _WINDOW_CONTROL_OPENCV_H_

#include <list>
#include "figure_opencv.h"
#include "controls_opencv.h"

/*!
This window displays the user interface. The user interface refers to the active figure.
To change the active figure while using the program the user can over with the mouse on any figure of choice.
*/
class window_control_opencv
{
public:
	window_control_opencv();
	~window_control_opencv();

	void show(window_display_opencv &fig);

	void resize_window(unsigned int display_dimx, unsigned int display_dimy);
	void move_window(float x, float y);
	void move_window(int x, int y);

	static void callback_mouse(int ev, int x, int y, int flags, void* userdata);

protected:
	std::string name;

	cv::Mat panel;

	// mouse position for control manipulation (x_current, y_current, x_previous, y_previous, x_onclick, y_onclick)
	float xx[8];
	float *xx_current;
	float *xx_previous;
	float *xx_onclick;
	float *xx_onrelease;

};

extern window_control_opencv *window_control;

#endif

