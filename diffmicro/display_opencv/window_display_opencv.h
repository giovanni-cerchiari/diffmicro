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
#ifndef _WINDOW_DISPLAY_OPENCV_H_
#define _WINDOW_DISPLAY_OPENCV_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <sstream>

#include "mouse_opencv.h"
#include "overlay_opencv.h"
#include "controls_opencv.h"


enum window_display_mode_opencv
{
	DISPLAY_1_CH_OPENCV = 1,
	DISPLAY_2_CH_OPENCV = 2,
	DISPLAY_3_CH_OPENCV = 3
};

/*!
This class can be used to plot a function or an image on a window.

Display pointer internal to the window_display_opencv may be equal to the external.
In this way any external modification of the arrays will be reflected on the display.
If you do not like this behavior, save your matrix outside. Besides, the user must free
the memory of the pointer that we are "looking" into the windows.

The absolute reference frame is define by the edge top left corner of the image.
x axis, corresponding to index 0, is from left to right
y axis, corresponding to index 1, is from oriented from top to bottom
the dimension of each pixel as physical quantity is defined in scale[]

While adding overlay to the window_display_opencv allocate them in the heap.
The class take care of deleting all the overalys. If the overaly is allocated
in the stack a memory error will occur. Again, overlay will point to some memory area
that contains the final data of the image of the functions. These memory area are expected
to be managed by the user.

*/
class window_display_opencv
{
public:
	window_display_opencv(std::string _name);
	window_display_opencv(const window_display_opencv &wd) { *this = wd; }
	~window_display_opencv();
	window_display_opencv &operator=(const window_display_opencv &wd);

	void create_display_window(unsigned int display_dimx, unsigned int display_dimy);

	//! this method binds the display to the external matrix
	void bind(unsigned int _dimx, unsigned int _dimy, double *img, unsigned int channel = 0);
	//! this function fixes the intensity scale as whisehed if _auto_scale will be assigned false
	void intensity_scale(bool _auto_scale = true, double _min = 0, double _max = 0);
	void show();
	//! allow to define the lateral size of the external matrix
	void setscale(double scalex, double scaley);
	//! allow to define the lateral size of the window
	void setdisplayscale(double disp_scalex, double disp_scaley);
	//! allow to define the display area
	void set_display_area(double top_left[], double bottom_right[]);
	//! zoom display
	void zoom(double zoomx, double zoomy);
	//!move window to percentace of the screen
	void movewindow(float x, float y);
	//!move window to pixel position on the screen
	void movewindow(int x, int y);

	//! allows flipping the y axis for function or images best view
	void y_flip();

	void save_display(std::string filename);

	double scalex(){ return scale[0]; }
	double scaley(){ return scale[1]; }
	double display_scalex(){ return scale_display[0]; }
	double display_scaley(){ return scale_display[1]; }
	int display_dimx(){ return image_display[0].cols; }
	int display_dimy(){ return image_display[0].rows; }
	double minval();
	double maxval();
	unsigned int n_channels(){ return dimc; }

	//! name identifier of the window
	std::string name;
	//! id of the figure is extremely useful to identify and manipulate the window in figure list
	int id;

	//! upper left position that is displayed
	double start_display[2];

	//!overlays to display on top of the figure
	overlay_opencv* active_overlay;
	//!collection of the overlays for this window_display
	std::list<overlay_opencv*> overlay;
	//! delete the specified overlay
	void erase_overlay(overlay_opencv* ov);

	//! active object of the user interface
	control_opencv* active_control;
	//! elements of the user interface for this window_display
	std::vector<control_opencv*> control;

	//! dimensions of the external matrix/function/image
	unsigned int dimx, dimy;

	//! pointer to the memory location of the external image
	double *image[4];

	int colormap;


protected:

	

	//! number of matrices attributed to the image
	unsigned int dimc;

	//! ordinary grayscale display
	cv::Mat image_display[4];

	//! for colormap display
	cv::Mat image_display_color;

	//! side dimension of one pixel of the general function
	double scale[2];
	//! side dimension of one pixel as on display
	double scale_display[2];
	//! minimum and maximum value at which the intensity will be normalized
	double m_min[4];
	double m_max[4];

	//! origin of chartesian axis
	double origin[2];

	void fill_display_matrices(double range);
	void min_max_images();

	void init();
	void init_controls();
	void clear();

};

enum control_id
{
	ID_BUTTON_AUTOSCALE = 0,
	ID_BUTTON_COLORMAP = 1,
	ID_BUTTON_CONTOUR = 2,
	ID_BUTTON_OVERLAY = 3,
	ID_BUTTON_YFLIP = 4,
	ID_BUTTON_MODE = 5,
	ID_CTRL_MINMAX_TRACKBAR = 6
};



void max_min_display_matrix(unsigned int dim, double *mat, double &max, double &min);



#endif
