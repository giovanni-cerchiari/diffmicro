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
#include "window_control_opencv.h"
#include "window_history_opencv.h"

void(*history_right_click_on_contour)(contour_opencv *ct);

window_history_opencv  *window_history(NULL);

window_history_opencv::window_history_opencv()
{
	int i;
	int dimx_panel;
	int dimy_panel;
	unsigned char *paneldatai;

	//window_history->selected_overlay = NULL;
	this->selected_overlay = NULL;

	name = "history_window";

	xx_current = xx;
	xx_previous = &(xx[2]);
	xx_onclick = &(xx[4]);
	xx_onrelease = &(xx[6]);
//	if (useri.display_status==DIFFMICRO_GRAPH_ON)
		cv::namedWindow(name, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

	dimx_panel = 200;
	dimy_panel = 1000;
	vertical_dimension = 1. / 15.;

	panel.create(dimy_panel, dimx_panel, CV_8U);
	for (i = 0; i < panel.rows*panel.cols; ++i)
	{
		paneldatai = (unsigned char*)(&(panel.data[i]));
		paneldatai[0] = 0;
	}
	cv::cvtColor(panel, panel, CV_GRAY2RGB);

	cv::setMouseCallback(name, window_history_opencv::callback_mouse, (void *)(active_figure));

	history_right_click_on_contour = history_right_click_on_contour_fit_a_circle;
}

window_history_opencv::~window_history_opencv()
{
	panel.deallocate();
	cv::destroyWindow(name);
}


void window_history_opencv::resize_window(unsigned int display_dimx, unsigned int display_dimy)
{
	unsigned int i;
	unsigned char* paneldatai;
	panel.deallocate();
	panel.create(display_dimy, display_dimx, CV_8U);
	for (i = 0; i < panel.rows*panel.cols; ++i)
	{
		paneldatai = (unsigned char*)(&panel.data[i]);
		paneldatai[0] = 0;
	}
	cv::cvtColor(panel, panel, CV_GRAY2RGB);
}

void window_history_opencv::move_window(int x, int y)
{
	int dimx, dimy;
	GetDesktopResolution(dimx, dimy);
	if ((x >= 0) && (y >= 0) && (x < dimx) && (y < dimy))
	{
		cv::moveWindow(name, x, y);
	}
}

void window_history_opencv::move_window(float x, float y)
{
	int dimx, dimy;
	GetDesktopResolution(dimx, dimy);
	if ((x >= 0) && (y >= 0) && (x < 1) && (y < 1))
	{
		this->move_window((int)(x*(float)(dimx)), (int)(y*(float)(dimy)));
	}
}

void window_history_opencv::callback_mouse(int ev, int x, int y, int flags, void* afig)
{
	int i;
	std::list<overlay_opencv*>::iterator it, end;

	window_history->xx[0] = (float)(x) / (float)(window_history->panel.cols);
	window_history->xx[1] = (float)(y) / (float)(window_history->panel.rows);

	window_history->selected_overlay = select_overlay((int)(window_history->xx[1] / window_history->vertical_dimension), active_figure->overlay);

	switch (ev)
	{
	case cv::EVENT_LBUTTONDOWN:
		if (NULL != window_history->selected_overlay)
			active_figure->erase_overlay(window_history->selected_overlay);
		break;
	case cv::EVENT_FLAG_RBUTTON:
		if ( (NULL != window_history->selected_overlay) && (0==strcmp("contour",window_history->selected_overlay->name.c_str())))
		{
			contour_opencv *ct = (contour_opencv*)(window_history->selected_overlay);
			history_right_click_on_contour(ct);
		}
		break;
	default:
		break;
	}


	// set previous position parameters
	for (i = 0; i < 2; ++i) window_history->xx[2 + i] = window_history->xx[i];
	window_history->show(*active_figure);
	window_control->show(*active_figure);
	active_figure->show();

	window_history->selected_overlay = NULL;
}


void window_history_opencv::show(window_display_opencv &fig)
{
	int i, dim;
	float step;
	unsigned char *paneldatai;
	return_get_value_overlay_opencv rval;
	std::list<overlay_opencv*>::iterator it, end;
	cv::Point pbegin, pend, ptext;
	std::stringstream ss;
	std::string text;
	cv::Scalar black(0, 0, 0);

	dim = 3*panel.cols*panel.rows;
	for (i = 0; i < dim; ++i)
	{
		paneldatai = (unsigned char*)(&(panel.data[i]));
		paneldatai[0] = 0;
	}

	step = (float)(panel.rows) * vertical_dimension;

	pbegin.x = 0;
	pend.x = panel.cols;

	end = fig.overlay.end();
	for (it = fig.overlay.begin(),i=0; it != end; ++it,++i)
	{
		pbegin.y = (int)( (float)(i)     *step);
		pend.y   = (int)(((float)(i)+0.9)*step);
		cv::rectangle(panel, pbegin, pend, *((*it)->color), CV_FILLED);

		ptext.x = (pbegin.x + pend.x) / 9;
		ptext.y = (pbegin.y + pend.y) / 2;

		(*it)->getvalue(&rval);
		ss.str("");
		ss << (*it)->name << "  ";
		switch (rval.type)
		{
		case OOV_TYPE_DOUBLE:
			ss << ((double*)(rval.val))[0];
			break;
		case OOV_TYPE_STRING:
			ss << ((std::string*)(rval.val))[0];
			break;
		};
		
		text = ss.str();
		cv::putText(panel, text, ptext, cv::FONT_HERSHEY_SIMPLEX, 0.5, black, 2);
	}
	
	imshow(this->name, panel);
}


void history_right_click_on_contour_fit_a_circle(contour_opencv *ct)
{
	circle_opencv *cr = new circle_opencv;
	double x0[4], r;
	circonference_fit<double, double>(ct->dimpoint, ct->x, x0, r);
	x0[2] = r + x0[0]; x0[3] = x0[1];
	cr->edit(x0);
	cr->color_normal.val[1] = 100;
	active_figure->overlay.push_back(cr);
}
