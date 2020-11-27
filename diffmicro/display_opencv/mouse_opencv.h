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
#ifndef _MOUSE_OPENCV_H_
#define _MOUSE_OPENCV_H_

//#include "window_display_opencv.h"
#include "overlay_opencv.h"

/*!This function is an example of void(*handle_on_matrix_callback)*/
void handle_on_matrix_callback_default(int *x, double *xx, void *fig);

extern void(*handle_on_matrix_callback)(int *x, double *xx, void *fig);

enum mouse_mode_opencv
{
	MOUSE_MODE_DRAWING_OPENCV = 0,
	MOUSE_MODE_PLOT_OPENCV = 1
};


/*!
This class is used to define the behaviour of the mouse while operating on a figure
To distinguish which is the figure to be modified the cursor position will
force the figure below to become the active_figure (see figure_opencv.h)
*/
class mouse_opencv
{
public:
	mouse_opencv();
	~mouse_opencv();
	static void mouse_opencv_callback(int ev, int x, int y, int flags, void* userdata);

	void draw(cv::Mat &img, double start[], double scale[]);
	/*
	void move();
	
	void ctrl_press();
	void middle_click();
	
	*/
	void left_click();
	void left_move();
	void right_click();
	void wheel(int value);

	void drawing(int ev, int x, int y, int flags);
	void plot(int ev, int x, int y, int flags);

	contour_opencv *manual_points;

protected:
	// mouse position for control manipulation (x_current, y_current, x_previous, y_previous, x_onclick, y_onclick, x_onrelease, y_onrelease)
	int x[8];
	int *x_current;
	int *x_onclick;
	int *x_onrelease;
	int *x_previous;

	// mouse position for control manipulation (x_current, y_current, x_previous, y_previous, x_onclick, y_onclick, x_onrelease, y_onrelease)
	double xx[8];
	double *xx_current;
	double *xx_onclick;
	double *xx_onrelease;
	double *xx_previous;

	void on_click();
	void on_release();

	int mode;


protected:

	int active_overlay;
	/*!
	Overlays are the object drawn on figures as surplus. To edit one it is practical to
	have a pointer to the obejct under modification
	*/
	overlay_opencv *overlay_edit;

};

/*!This is the mouse class of the application.*/
extern mouse_opencv mouse;




#endif