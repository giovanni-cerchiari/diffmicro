/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 02/2016

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
#include "figure_opencv.h"
#include "window_control_opencv.h"
#include "window_history_opencv.h"

bool flg_figure_enviroment(false);
window_display_opencv* active_figure;
std::list<window_display_opencv*> figure;

window_display_opencv* find_figure(int id_figure)
{
	std::list<window_display_opencv*>::iterator it, end;
	bool found = false;
	window_display_opencv* fig;
	fig = NULL;

	end = figure.end();
	for (it = figure.begin(); it != end; ++it)
	{
		if (id_figure == (*it)->id)
		{
			id_figure = (*it)->id;
			fig = *it;
			found = true;
		}
	}

	return fig;
}


window_display_opencv* open_figure(int id_figure)
{
	
	int id;
	std::string name;
	std::stringstream str;
	window_display_opencv* fig;


	fig = find_figure(id_figure);

	if (NULL == fig)
	{
		id = 0;

		while (NULL != find_figure(id)) ++id;

		str << "figure" << id;
		name = str.str();
		fig = new window_display_opencv(name);
		fig->id = id;
		fig->bind(0, 0, NULL);
		fig->create_display_window(0, 0);
		figure.push_back(fig);
	}

	return fig;
}

void figure_grid(unsigned int n_figure, std::list<int> &id_figure)
{
	window_display_opencv *fig;
	unsigned int i, j, i_figure;
	unsigned int dimx_desktop, dimy_desktop;
	unsigned int dimx_history, dimy_history;
	unsigned int dimx_window, dimy_window;
	unsigned int xdiv, ydiv;
	int id;

	//++n_figure;

	GetDesktopResolution(dimx_desktop, dimy_desktop);

	dimy_desktop = (unsigned int)((float)(0.9)*(float)(dimy_desktop));
	dimy_history = dimy_desktop;

	dimx_history = (unsigned int)((float)(0.15)*(float)(dimx_desktop));
	dimx_desktop -= dimx_history;

	xdiv = n_figure; ydiv = 1;
	dimx_window = dimx_desktop / xdiv;
	dimy_window = dimy_desktop / ydiv;
	while (dimx_window < dimy_window)
	{
		ydiv += 1;
		xdiv = (unsigned int)(std::ceil((float)(n_figure) / (float)(ydiv)));
		dimx_window = dimx_desktop / xdiv;
		dimy_window = dimy_desktop / ydiv;
	}
	id_figure.clear();
	i_figure = 0;
	for (j = 0; (j < ydiv) && (i_figure<n_figure); ++j)
	{
		for (i = 0; (i < xdiv) && (i_figure<n_figure); ++i)
		{
//			if ((0 == j) && (0 == i))++i;
			fig = open_figure(-1);
			id_figure.push_back(fig->id);
			fig->create_display_window(dimx_window, dimy_window);
			fig->movewindow((int)(i*dimx_window) + dimx_history, (int)(j*dimy_window));
	//		fig->bind(0, 0, NULL);
			active_figure = fig;
			fig->show();
			++i_figure;
		}
	}
	
	window_history->move_window(0, 0);
	window_control->move_window(dimx_history, 0);

}

window_display_opencv* new_figure(unsigned int dimx, unsigned int dimy, double *img)
{
	unsigned int dimx_desktop, dimy_desktop;
	unsigned int sidewx, sidewy;
	double top_left[2], bottom_right[2];

	active_figure = open_figure(-1);

	GetDesktopResolution(dimx_desktop, dimy_desktop);
	sidewx = dimx_desktop / 3;
	sidewy = (sidewx * dimy) / dimx;

	active_figure->create_display_window(sidewx, sidewy);
	active_figure->bind(dimx, dimy, img);
	active_figure->setscale(1, 1);
	top_left[0] = 0.0; top_left[1] = 0.0;
	bottom_right[0] = (double)(dimx)*active_figure->scalex();
	bottom_right[1] = (double)(dimy)*active_figure->scaley();
	active_figure->set_display_area(top_left, bottom_right);
	active_figure->setdisplayscale(active_figure->display_scalex(), active_figure->display_scalex());
	active_figure->show();

	return active_figure;
}

window_display_opencv* new_figure(unsigned int dimx, unsigned int dimy, unsigned int dim_channel, double **images)
{
	if (0 == dim_channel || 3 < dim_channel) return active_figure;
	
	active_figure = new_figure(dimx, dimy, images[0]);
	for (int i = 1; i < dim_channel; ++i) active_figure->bind(dimx, dimy, images[i], i);
	active_figure->show();
	return active_figure;
}

void add_axes_and_legend(window_display_opencv* fig)
{
	if (NULL == fig) return;
	std::list<overlay_opencv*>::iterator it, end;
	overlay_opencv *ov;
	axes_opencv *axes;
	legend_opencv *legend;
	bool flg_axes = false;
	bool flg_legend = false;

	end = fig->overlay.end();
	for (it = fig->overlay.begin(); it != end; ++it)
	{
		if (OVERLAY_AXES_OPENCV == (*it)->type) flg_axes = true;
		if (OVERLAY_LEGEND_OPENCV == (*it)->type) flg_legend = true;
	}

	if (false == flg_axes)
	{
		axes = new axes_opencv;
		ov = (overlay_opencv*)(axes);
		fig->overlay.push_back(ov);
	}

	if (false == flg_legend)
	{
		legend = new legend_opencv();
		legend->list_overlay = &(fig->overlay);
		ov = (overlay_opencv*)(legend);
		fig->overlay.push_back(ov);
	}
		
}


window_display_opencv* plot(window_display_opencv* fig, scatter_opencv *scatter)
{
	if (NULL != fig) return plot(fig->id, scatter);
	else             return plot(-1, scatter);
}
window_display_opencv* plot(window_display_opencv* fig, barplot_opencv *barplot)
{
	if (NULL != fig) return plot(fig->id, barplot);
	else             return plot(-1, barplot);
}

window_display_opencv* plot(int id_figure, scatter_opencv *scatter)
{
	
//	window_display_opencv* fig;
	overlay_opencv *ov;

	unsigned int i;
	unsigned int sidewx = 900;
	unsigned int sidewy = 700;
	unsigned int posmin, posmax;

	active_figure = open_figure(id_figure);
	if (0 == active_figure->display_dimx())		active_figure->create_display_window(sidewx, sidewy);
	add_axes_and_legend(active_figure);

	adapt_plot_area_to_function(active_figure, scatter->dimpoint, scatter->x, scatter->y);

	bool flipy = true;
	active_figure->control[ID_BUTTON_YFLIP]->set_value(&flipy);

	ov = (overlay_opencv*)(scatter);
	active_figure->overlay.push_back(ov);
	active_figure->show();

	return active_figure;
}

window_display_opencv* plot(int id_figure, barplot_opencv *barplot)
{

	//window_display_opencv* fig;
	overlay_opencv *ov;

	unsigned int i;
	unsigned int sidewx = 900;
	unsigned int sidewy = 700;
	unsigned int posmin, posmax;

	active_figure = open_figure(id_figure);
	if (0 == active_figure->display_dimx()) active_figure->create_display_window(sidewx, sidewy);
	add_axes_and_legend(active_figure);
	
	adapt_plot_area_to_function(active_figure, barplot->dimpoint, barplot->x, barplot->y);

	bool flipy = true;
	active_figure->control[ID_BUTTON_YFLIP]->set_value(&flipy);

	ov = (overlay_opencv*)(barplot);
	active_figure->overlay.push_back(ov);
	active_figure->show();

	return active_figure;
}

void adapt_plot_area_to_function(window_display_opencv *fig, unsigned int dimpoint, double *x, double *y)
{
	unsigned int i;
	unsigned int posmin, posmax;
	double bottom_right[2];
	double top_left[2];
	double dx[2];
	bool flg_mode;
	maxminary(dimpoint, x, bottom_right[0], posmax, top_left[0], posmin);
	
	fig->control[ID_BUTTON_MODE]->get_value(&flg_mode);
	if (true == flg_mode)
		maxminary(dimpoint, y, bottom_right[1], posmax, top_left[1], posmin);
	else
		maxminary(dimpoint, y, top_left[1], posmax, bottom_right[1], posmin);

//	top_left[1] = 0.0;
	for (i = 0; i < 2; ++i)
	{
		dx[i] = bottom_right[i] - top_left[i];
	}
	top_left[0] -= 0.2*dx[0]; top_left[1] -= 0.5*dx[1];
	bottom_right[0] += 0.4*dx[0]; bottom_right[1] += 0.1*dx[1];

	fig->set_display_area(top_left, bottom_right);
}

axes_opencv* get_axes(int id_figure)
{
	std::list<window_display_opencv*>::iterator it, end;

	end = figure.end();
	for (it = figure.begin(); it != end; ++it)
	{
		if (id_figure == (*it)->id)
		{
			return (get_axes(*it));
		}
	}
	return NULL;
}

void volume_movie(unsigned int dimx, unsigned int dimy, unsigned int dimz, unsigned int n_channels, double **vol, int frame_time_step_ms)
{
	window_display_opencv *fig;
	text_relative_opencv *text;
	unsigned int k, j;
	bool flg_colormap = true;
	std::stringstream ttext;

	fig = new_figure(dimx, dimy, n_channels, vol);
	fig->control[ID_BUTTON_COLORMAP]->set_value(&flg_colormap);
	text = new text_relative_opencv;
	text->text = "t = 0";
	text->x[0] = 0.05;
	text->x[1] = 0.05;
	text->font_scale = 1;
	text->deselect();
	fig->overlay.push_back((overlay_opencv*)(text));
	fig->show();
	waitkeyboard(frame_time_step_ms);
	for (k = 1; k < dimz; ++k)
	{
		for (j = 0; j < n_channels; ++j)
		{
			fig->image[j] = &(vol[j][dimx*dimy*k]);
		}
		ttext.str("");
		ttext <<"t = " << k;
		text->text = ttext.str();
		fig->show();
		waitkeyboard(frame_time_step_ms);
	}
	delete_figure(fig->id);
}




axes_opencv* get_axes(window_display_opencv* fig)
{
	std::list<overlay_opencv*>::iterator it, end;
	end = fig->overlay.end();
	for (it = fig->overlay.begin(); it != end; ++it)
	{
		if (OVERLAY_AXES_OPENCV == (*it)->type)
		{
			return (axes_opencv*)(*it);
		}
	}
	return NULL;
}

void delete_figure(int id)
{
//	int i;
	std::list<window_display_opencv*>::iterator it;

	//	std::cerr << "lol "<< std::endl;
	//i = 0;

	for (it = figure.begin(); it != figure.end(); ++it)//, ++i)
	{
		//		std::cerr << "i = " << i << "\tid = " << (*it)->id << std::endl;
		if (id == (*it)->id)
		{
			//			std::cerr << "cane = "<<id << std::endl;
			if (active_figure != *it)
			{
				delete *it;
				it = figure.erase(it);
			}
			else
			{
				it = figure.erase(it);
				delete active_figure;
				if (0 < figure.size()) active_figure = *(figure.begin());
				else                   active_figure = NULL;
			}
			return;
		}
	}

}

void delete_figures()
{
	std::list<window_display_opencv*>::iterator it;

	while(0!=figure.size())
	{
		it = figure.begin();
		delete_figure((*it)->id);
	}

}

void init_figure_enviroment()
{
	window_control = new window_control_opencv;
	window_history = new window_history_opencv;
	flg_figure_enviroment = true;
}

void close_figure_enviroment()
{
	flg_figure_enviroment = false;
	if(NULL != window_control) delete window_control;
	if(NULL != window_history) delete window_history;
}