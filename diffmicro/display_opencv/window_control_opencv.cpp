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

window_control_opencv  *window_control(NULL);

window_control_opencv::window_control_opencv()
{
	int i;
	int dimx_panel;
	int dimy_panel;
	unsigned char *paneldatai;

	name = "control_window";

	xx_current = xx;
	xx_previous = &(xx[2]);
	xx_onclick = &(xx[4]);
	xx_onrelease = &(xx[6]);
//	if (useri.display_status==DIFFMICRO_GRAPH_ON)
		cv::namedWindow(name, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

	dimx_panel = 800;
	dimy_panel = 300;

	panel.create(dimy_panel, dimx_panel, CV_8U);
	for (i = 0; i < panel.rows*panel.cols; ++i)
	{
		paneldatai = (unsigned char*)(&panel.data[i]);
		paneldatai[0] = 0;
	}
	cv::cvtColor(panel, panel, CV_GRAY2RGB);

	cv::setMouseCallback(name,		window_control_opencv::callback_mouse, (void *)(active_figure));

}

window_control_opencv::~window_control_opencv()
{

	panel.deallocate();
	cv::destroyWindow(name);

}

void window_control_opencv::resize_window(unsigned int display_dimx, unsigned int display_dimy)
{
	unsigned int i;
	unsigned char * paneldatai;
	panel.deallocate();
	panel.create(display_dimy, display_dimx, CV_8U);
	for (i = 0; i < panel.rows*panel.cols; ++i)
	{
		paneldatai = (unsigned char*)(&(panel.data[i]));
		//(unsigned char)(panel.data[i]) = 0;
		paneldatai[0] = 0;
	}
	cv::cvtColor(panel, panel, CV_GRAY2RGB);
}

void window_control_opencv::move_window(int x, int y)
{
	int dimx, dimy;
	GetDesktopResolution(dimx, dimy);
	if ((x >= 0) && (y >= 0) && (x < dimx) && (y < dimy))
	{
		cv::moveWindow(name, x, y);
	}
}

void window_control_opencv::move_window(float x, float y)
{
	int dimx, dimy;
	GetDesktopResolution(dimx, dimy);
	if ((x >= 0) && (y >= 0) && (x < 1) && (y < 1))
	{
		this->move_window((int)(x*(float)(dimx)), (int)(y*(float)(dimy)));
	}
}


void window_control_opencv::callback_mouse(int ev, int x, int y, int flags, void* afig)
{
	int i;
	std::vector<control_opencv*>::iterator it, end;
	window_display_opencv *fig;

	fig = active_figure;//(window_display_opencv*)(afig);

	window_control->xx[0] = (float)(x) / (float)(window_control->panel.cols);
	window_control->xx[1] = (float)(y) / (float)(window_control->panel.rows);

	if (cv::EVENT_LBUTTONDOWN == ev)
	{
		fig->active_control = NULL;
		// set onclick position parameters
		for (i = 0; i < 2; ++i) window_control->xx[4 + i] = window_control->xx[i];

		end = fig->control.end();
		for (it = fig->control.begin(); it!=end; ++it)
		{
			if (0 <= (*it)->handle(window_control->xx))
			{
				fig->active_control = *it;
				select_control(fig->active_control, fig->control);
				break;
			}
		}
		if (NULL != fig->active_control) fig->active_control->onclick(window_control->xx);

	}

	if (cv::EVENT_MOUSEMOVE == ev && cv::EVENT_FLAG_LBUTTON == flags)
	{
		if (NULL != fig->active_control) fig->active_control->onmove(window_control->xx);
	}

	if (cv::EVENT_LBUTTONUP == ev)
	{
		if (NULL != fig->active_control)
		{
			fig->active_control->onrelease(window_control->xx);
			if (CTRL_TEXTBOX != fig->active_control->type)
			{
				fig->active_control->deselect();
				fig->active_control = NULL;
			}
		}
	}

	// set previous position parameters
	for (i = 0; i < 2; ++i) window_control->xx[2 + i] = window_control->xx[i];
	window_control->show(*active_figure);
	active_figure->show();
}


void window_control_opencv::show(window_display_opencv &fig)
{

	std::vector<control_opencv*>::iterator it, end;

	end = fig.control.end();
	for (it = fig.control.begin(); it != end; ++it)
	{
		(*it)->draw(this->panel);
	}

	imshow(name, panel);
}





