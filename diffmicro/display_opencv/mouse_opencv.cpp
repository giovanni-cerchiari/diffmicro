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

#include "stdafx.h"
#include "mouse_opencv.h"
#include "figure_opencv.h"
#include "window_control_opencv.h"

void handle_on_matrix_callback_default(int *x, double *xx, void* data){}
void(*handle_on_matrix_callback)(int *x, double *xx, void* data) = handle_on_matrix_callback_default;

mouse_opencv mouse;

mouse_opencv::mouse_opencv()
{
	xx_current = xx;
	xx_previous = &(xx[2]);
	xx_onclick = &(xx[4]);
	xx_onrelease = &(xx[6]);

	x_current = x;
	x_previous = &(x[2]);
	x_onclick = &(x[4]);
	x_onrelease = &(x[6]);

	active_overlay = -1;

	manual_points = NULL;
	overlay_edit = NULL;

	mouse.mode = MOUSE_MODE_DRAWING_OPENCV;
}

mouse_opencv::~mouse_opencv()
{
	// if it is not null, it is not yet been assigned to any figure
	if (NULL != manual_points) delete manual_points;

}

void mouse_opencv::draw(cv::Mat &img, double start[], double scale[])
{
	cv::Scalar color(255, 255, 255);
	cv::Point valuetextp;
	std::stringstream ss;
	std::string text;
	unsigned int i, k,c;
	unsigned int dimx, dimy;
	double xxx[2];
//	double scale[2];

	dimx = active_figure->dimx;
	dimy = active_figure->dimy;

	xxx[0] = xx_current[0] * active_figure->scalex();
	xxx[1] = xx_current[1] * active_figure->scaley();

	switch (active_overlay)
	{
	case 0:
		overlay_edit->draw(img, start, scale);

		valuetextp.x = 0;
		valuetextp.y = 30;
		break;
	default:
		valuetextp.x = x_current[0];
		valuetextp.y = x_current[1];

		break;
	}

	i = (unsigned int)(std::round(xxx[0]));
	k = (unsigned int)(std::round(xxx[1]));

	if ((i < dimx) && (k < dimy))
	{
		ss << active_figure->image[0][k*dimx + i] << " ";
		for (c = 1; c < active_figure->n_channels(); ++c)	ss << ", " << active_figure->image[c][k*dimx + i] << " ";

		text = ss.str();
		cv::putText(img, text, valuetextp + cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
		ss.str("");

	}
	
	if (NULL != handle_on_matrix_callback) handle_on_matrix_callback(x, xx, active_figure);
	
	//ss << "(" << xxx[0] << ", " << xxx[1] << ")";
	ss << "(" << xx_current[0] << ", " << xx_current[1] << ")";

	text = ss.str();
	cv::putText(img, text, valuetextp + cv::Point(20, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

//	std::cout << manual_points << std::endl;
	if (NULL != manual_points) manual_points->draw(img, start, scale);

}


void mouse_opencv::mouse_opencv_callback(int ev, int x, int y, int flags, void* userdata)
{
	unsigned int i;
	bool flg;

	active_figure = ((window_display_opencv*)(userdata));

	active_figure->control[ID_BUTTON_MODE]->get_value(&flg);
	if (false == flg) mouse.mode = MOUSE_MODE_DRAWING_OPENCV;
	else              mouse.mode = MOUSE_MODE_PLOT_OPENCV;

	mouse.x_current[0] = x;
	mouse.x_current[1] = y;

	mouse.xx_current[0] = active_figure->display_scalex() * (double)(x)+active_figure->start_display[0];
	mouse.xx_current[1] = active_figure->display_scaley() * (double)(y)+active_figure->start_display[1];


	// ON CLICK
	if ((cv::EVENT_LBUTTONDOWN == ev) || (cv::EVENT_RBUTTONDOWN == ev) || (cv::EVENT_MBUTTONDOWN == ev) ||
		(cv::EVENT_LBUTTONDBLCLK == ev) || (cv::EVENT_RBUTTONDBLCLK == ev) || (cv::EVENT_MBUTTONDBLCLK == ev))
	{
		mouse.on_click();
	}

	// ON RELEASE
	if ((cv::EVENT_LBUTTONUP == ev) || (cv::EVENT_RBUTTONUP == ev) || (cv::EVENT_MBUTTONUP == ev))		mouse.on_release();

	// WHELL -> ZOOM
	if ((cv::EVENT_MOUSEWHEEL == ev) || (cv::EVENT_MOUSEHWHEEL == ev))		mouse.wheel(flags);

	switch (mouse.mode)
	{
		case MOUSE_MODE_DRAWING_OPENCV:
			mouse.drawing(ev, x, y, flags);
			break;
		case MOUSE_MODE_PLOT_OPENCV:
			mouse.plot(ev, x, y, flags);
			break;
	};
      	

	if (cv::EVENT_MOUSEMOVE == ev)
	{
		switch (flags)
		{
		case cv::EVENT_FLAG_LBUTTON:
			mouse.left_move();
			break;
			case cv::EVENT_FLAG_RBUTTON:  // RIGHT CLICK -> PAN
			mouse.right_click();
			break;
		case cv::EVENT_FLAG_MBUTTON:
			break;
		case (cv::EVENT_FLAG_CTRLKEY | cv::EVENT_FLAG_LBUTTON) :
			mouse.left_move();
			break;
		case cv::EVENT_FLAG_SHIFTKEY:
			break;
		case cv::EVENT_FLAG_ALTKEY:

			break;
		default:
			break;
		};
	}




	//std::cout << "e = "<<ev<<"\tx = " << x << "\ty = " << y <<"\tf = "<<flags<<"\tid = "<<active_figure->id<< std::endl;
	active_figure->show();
	if (false == flg_figure_enviroment)
		window_control->show(*active_figure);

	for (i = 0; i < 2; ++i)
	{
		mouse.x_previous[i] = mouse.x_current[i];
		mouse.xx_previous[i] = mouse.xx_previous[i];
	}
	
}

void mouse_opencv::drawing(int ev, int x, int y, int flags)
{
	unsigned int i, k;

	switch (ev)
	{
	case cv::EVENT_LBUTTONDOWN:
		switch (flags)
		{
		case 1:
			mouse.overlay_edit = new circle_opencv;
			mouse.active_overlay = 0;
			mouse.left_click();

			break;
		case 9:
			mouse.overlay_edit = new rectangle_opencv;
			mouse.active_overlay = 0;
			mouse.left_click();
			break;
		case 33:
			i = (unsigned int)(mouse.xx_current[0]);
			k = (unsigned int)(mouse.xx_current[1]);

			if ((i < active_figure->dimx) && (k < active_figure->dimy))
			{
				contour_opencv_edit_parameters par;
				double scale[2];
				scale[0] = active_figure->scalex(); scale[1] = active_figure->scaley();
				par.dimx = active_figure->dimx;
				par.dimy = active_figure->dimy;
				par.mat = active_figure->image[0];
				par.scale = scale;
				par.val = active_figure->image[0][k*active_figure->dimx + i];
				mouse.overlay_edit = new contour_opencv;
				mouse.overlay_edit->edit(&par);
				active_figure->overlay.push_back(mouse.overlay_edit);
				mouse.overlay_edit = NULL;
			}
			break;
		case 41:

			if (NULL == mouse.manual_points)
			{
				mouse.manual_points = new contour_opencv;
				mouse.manual_points->val = 0;
				mouse.manual_points->color_normal = cv::Scalar(100, 255, 100);
				mouse.manual_points->set_marker_type(MARKER_CROSS_OPENCV);
				mouse.manual_points->marker_size = 4;
			}

			mouse.manual_points->push_back(mouse.xx_current);

			break;
		}
	default:
		break;
	}

}

void mouse_opencv::plot(int ev, int x, int y, int flags)
{

	unsigned int i, k;

	switch (ev)
	{
	case cv::EVENT_LBUTTONDOWN:
		switch (flags)
		{
		case 1:

			break;
		case 9:
			mouse.overlay_edit = new rectangle_opencv;
			mouse.active_overlay = 0;
			mouse.left_click();
			break;
		case 33:

			break;
		case 41:



			break;
		}
	default:
		break;
	}

}


void mouse_opencv::left_click()
{
	int i;
	double xx[4];

	//this->overlay_edit = new circle_opencv;
	//this->overlay_edit = new rectangle_opencv;

	
	for (i = 0; i < 2; ++i)
	{
		xx[i] = xx_onclick[i];
		xx[2 + i] = xx_onclick[i];
	}
	overlay_edit->edit(xx);

	
}

void mouse_opencv::left_move()
{
	int i;
	double xx[4];

	for (i = 0; i < 2; ++i)
	{
		xx[    i] = xx_onclick[i];
		xx[2 + i] = xx_current[i];
	}

	if (NULL != overlay_edit)	overlay_edit->edit(xx);

}


void mouse_opencv::right_click()
{
	int i;


	//for (i = 0; i < 2; ++i) figure[active_figure].start_display[i] += figure[active_figure].display_scalex() * (double)(x_previous[i]-x_current[i]);
	for (i = 0; i < 2; ++i) active_figure->start_display[i] += xx_previous[i] - xx_current[i];

}

void mouse_opencv::wheel(int value)
{

	double gain;
	if (0 > value) gain = 0.9;
	else           gain = 1.1;

	//std::cout << "sx = " << active_figure->start_display[0];
	//std::cout << "\tdx = " << active_figure->display_scalex()<<std::endl;

	// I did not fully understood the 0.01, but works
	active_figure->start_display[0] += (gain-1.-0.01)*active_figure->display_scalex() * ((double)(x_current[0]));
	active_figure->start_display[1] += (gain-1.-0.01)*active_figure->display_scaley() * ((double)(x_current[1]));

	active_figure->zoom(gain, gain);

}


void mouse_opencv::on_click()
{
	int i;

	for (i = 0; i < 2; ++i)
	{
		this->x_onclick[i] = this->x_current[i];
		this->x_previous[i] = this->x_current[i];
		this->xx_onclick[i] = this->xx_current[i];
		this->xx_previous[i] = this->xx_current[i];
	}
	
}

void mouse_opencv::on_release()
{
	int i;

	for (i = 0; i < 2; ++i)
	{
		this->x_onrelease[i] = this->x_current[i];
		this->xx_onrelease[i] = this->xx_current[i];
	}

	if (0 <= active_overlay)
	{
		switch (mode)
		{
			case MOUSE_MODE_DRAWING_OPENCV:
				active_figure->overlay.push_back(overlay_edit);
				active_overlay = -1;
				mouse.overlay_edit = NULL;
			break;
			case MOUSE_MODE_PLOT_OPENCV:
				double topleft[2];
				double bottomright[2];

				for(i=0; i<2; ++i) if(this->xx_onclick[i]==this->xx_onrelease[i]) break;
				for(i=0; i<2; ++i)
				{
					if(this->xx_onclick[i]<this->xx_onrelease[i])
					{
						topleft[i] = this->xx_onclick[i];
						bottomright[i] = this->xx_onrelease[i];
					}
					else
					{
						topleft[i] = this->xx_onrelease[i];
						bottomright[i] = this->xx_onclick[i];
					}
				}
				
				active_figure->set_display_area(topleft, bottomright);
				active_overlay = -1;
				delete this->overlay_edit;
				this->overlay_edit = NULL;
				break;
		}
	}
}
