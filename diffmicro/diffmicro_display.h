/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com

date: 2016
*/
/*
Copyright: Mojtaba Norouzisadeh, Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
e-mail: norouzi.mojtaba.sade@gmail.com

update: 05/2020 - 09/2020
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

#ifndef _DIFFMICRO_DISPLAY_H_
#define _DIFFMICRO_DISPLAY_H_

#include "global_define.h"
#include "figure_opencv.h"

extern bool flg_display_read;
extern INDEX display_read_mat_dimx;
extern INDEX display_read_mat_dimy;
extern double *display_read_mat;
extern window_display_opencv *read_image_window;

extern double *display_pw_mat;


extern window_display_opencv *average_counter_window;
extern scatter_opencv *average_counter_scatter;
extern double *x_avg_counter;
extern double *y_avg_counter;
extern window_display_opencv *average_window;
extern scatter_opencv *average_scatter;
extern double *x_avg;
extern double *y_avg;

/*!This function must be called to initialize the display enviroment of diffmicro.*/
void initilize_display(INDEX dimx, INDEX dimy, INDEX dimfile);

void display_read(unsigned short *img);
void display_execution_map(INDEX dimfilelist, unsigned char *exe_map);
void update_execution_map(INDEX dimfilelist, unsigned char* exe_map);

template<typename TYPE>
void display_average(INDEX dimfilelist, TYPE *average,	window_display_opencv **win, scatter_opencv **sc, double **x, double **y, std::string ylabel)
{
	INDEX i;
	axes_opencv *axes;

	*sc = new scatter_opencv;
	*x= new double[dimfilelist];
	*y = new double[dimfilelist];
	(*sc)->dimpoint = dimfilelist;
	(*sc)->x = *x;
	(*sc)->y = *y;
	(*sc)->label = ylabel;
	for (i = 0; i < dimfilelist; ++i)
	{
		(*x)[i] = (double)(i);
		(*y)[i] = (double)(average[i]);
		//		std::cerr << "i=" << x_avg_counter[i] << "\ty = " << y_avg_counter[i] << std::endl;
	}
	*win = plot(-1, *sc);
	axes = get_axes(*win);
	axes->xlabel = "time distance";
	axes->ylabel = ylabel;

	(*win)->show();
}

void display_power_spectrum_r(INDEX dist, STORE_REAL *pw);
void display_power_spectrum_normal(INDEX dist, STORE_REAL *pw);
void display_radial_lut(INDEX dimy, INDEX dimx, INDEX dim_radial_lut, unsigned int* lut);

void display_open_azhavg(INDEX dimr);
void display_update_azhavg(INDEX dimr, double* azhavg);

/*!This function must be called at the end of the program to free the memory used for display.*/
void close_display();




#endif