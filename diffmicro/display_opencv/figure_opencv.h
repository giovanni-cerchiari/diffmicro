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

#ifndef _FIGURE_OPENCV_H_
#define _FIGURE_OPENCV_H_

#include "keyboard_opencv.h"
#include "window_display_opencv.h"

extern bool flg_figure_enviroment;

/*!
This is the currently selected figure.
There shall be always a valid active figure. Otherwise, the behaviour of the program is undefined.

Note: I found more practical having only one window for user interface and history to display those property
for one figure only at time.
*/
extern window_display_opencv* active_figure;
/*!
There can be more that one figure on the screen. Here is the place where they are stored togheter.
By making sure that this variable it is used, the program will also take case of deleting all
the figures at the end of the exection.
*/
extern std::list<window_display_opencv*> figure;

/*
struct multiple_windows
{
	int x;
	int y;
	window_display_opencv* fig;
};*/

/*!
High level function to prepare a new window with to matrix *img
*/
window_display_opencv* find_figure(int id_figure);

/*!
If the figure exist in figure list than return the pointer.
If not, the function creates a new window, adds it to figure list and returns the pointer.
*/
window_display_opencv* open_figure(int id_figure);

/*!
Creation of a grid of figure for ease of display
*/
void figure_grid(unsigned int n_figure, std::list<int> &id_figure);

/*!
Check if axes and legend are already present in the specified window_display_opencv.
If not it adds them to the overlays of the window
*/
void add_axes_and_legend(window_display_opencv* fig);

/*!
New window for image display is created
*/
window_display_opencv* new_figure(unsigned int dimx, unsigned int dimy, double *img);

/*!
New window for image display is created. Multiple channels are allowed. No more than 3 channels are supported.
Each double matrix is inteded as a different channel
*/
window_display_opencv* new_figure(unsigned int dimx, unsigned int dimy, unsigned int dim_channel, double **images);

/*!
An image for display of a plot can be opened with open_figure.
If NULL == fig a new window_display_opencv is created and added to figure list
*/
window_display_opencv* plot(window_display_opencv* fig, scatter_opencv *scatter);
/*!
An image for display of a plot can be opened with open_figure.
If NULL == fig a new window_display_opencv is created and added to figure list
*/
window_display_opencv* plot(window_display_opencv* fig, barplot_opencv *barplot);
/*!
An image for display of a plot can be opened with open_figure.
If the window is not present in figure list a new window_display_opencv is created and added to figure list
*/
window_display_opencv* plot(int id_figure, scatter_opencv *scatter);
/*!
An image for display of a plot can be opened with open_figure.
If the window is not present in figure list a new window_display_opencv is created and added to figure list
*/
window_display_opencv* plot(int id_figure, barplot_opencv *barplot);

/*! This function can be used to make the plot appear within the range of the datapoints specified as input.*/
void adapt_plot_area_to_function(window_display_opencv *fig, unsigned int dimpoint, double *x, double *y);


void volume_movie(unsigned int dimx, unsigned int dimy, unsigned int dimz, unsigned int n_channels, double **vol, int frame_time_step_ms);


/*!
Usefull fuction for changing the properties of the axes of a window_display_opencv.
*/
axes_opencv *get_axes(int id_figure);
/*!
Usefull fuction for changing the properties of the axes of a window_display_opencv.
*/
axes_opencv *get_axes(window_display_opencv* fig);

/*!
This funtion frees the memory used by one figure
*/
void delete_figure(int id);
/*!
This funtion frees memory of all figures
*/
void delete_figures();

/*!
This function intialize the plot enviroment
*/
void init_figure_enviroment();
/*!
This function close the plot enviroment
*/
void close_figure_enviroment();
#endif