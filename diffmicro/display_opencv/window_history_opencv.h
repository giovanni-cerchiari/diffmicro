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
#ifndef _WINDOW_HISTORY_OPENCV_H_
#define _WINDOW_HISTORY_OPENCV_H_

#include "figure_opencv.h"

/*!
This window is used to manage the overlay that appears on a window_display
*/
class window_history_opencv
{
public:
	window_history_opencv();
	~window_history_opencv();

	void show(window_display_opencv &fig);

	void resize_window(unsigned int display_dimx, unsigned int display_dimy);
	void move_window(float x, float y);
	void move_window(int x, int y);

	static void callback_mouse(int ev, int x, int y, int flags, void* userdata);

protected:
	std::string name;

	cv::Mat panel;

	float vertical_dimension;
	// mouse position for control manipulation (x_current, y_current, x_previous, y_previous, x_onclick, y_onclick, x_onrelease, x_onrelease)
	float xx[8];
	float *xx_current;
	float *xx_previous;
	float *xx_onclick;
	float *xx_onrelease;

	overlay_opencv* selected_overlay;
};

extern window_history_opencv  *window_history;

extern void(*history_right_click_on_contour)(contour_opencv *ct);
void history_right_click_on_contour_fit_a_circle(contour_opencv *ct);


#endif
