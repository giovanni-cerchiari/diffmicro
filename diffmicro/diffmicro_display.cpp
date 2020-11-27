/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 08/2016
*/

/*
Copyright: Mojtaba Norouzisadeh, Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
e-mail: norouzi.mojtaba.sade@gmail.com

update: 05/2020 - 09/2020
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
#include "diffmicro_io.h"
#include "radial_storage_lut.h"
#include "diffmicro_display.h"
#include "histogram.h"
#include "my_math.h"

bool flg_display_read(false);
INDEX display_read_mat_dimx;
INDEX display_read_mat_dimy;
double *display_read_mat;
window_display_opencv *read_image_window(NULL);
histogram<double,unsigned int> read_image_histogram;
double *x_histo(NULL);
double *f_histo(NULL);
barplot_opencv *bar_histogram;
window_display_opencv *read_image_histogram_window(NULL);
window_display_opencv *execution_map_window(NULL);
double *display_execution_map_matrix(NULL);
window_display_opencv* radial_lut_window(NULL);
double* display_radial_lut_matrix(NULL);
window_display_opencv* azhavg_window(NULL);
double* display_azhavg_x(NULL);
double* display_azhavg_y(NULL);
double *display_pw_mat(NULL);
window_display_opencv *pw_window;
text_relative_opencv *pw_dist_text;

window_display_opencv *average_counter_window(NULL);
scatter_opencv *average_counter_scatter(NULL);
double *x_avg_counter(NULL);
double *y_avg_counter(NULL);
window_display_opencv *average_window(NULL);
scatter_opencv *average_scatter(NULL);
double *x_avg(NULL);
double *y_avg(NULL);

void initilize_display(INDEX dimx, INDEX dimy, INDEX dimfile)
{
	INDEX i,dim;
	int desktop_dimx, desktop_dimy;
	int id;
	unsigned int sidew;

	std::string name;
	double top_left[2], bottom_right[2];

	display_read_mat_dimx = dimx;
	display_read_mat_dimy = dimy;

	display_read_mat = new double[dimx*dimy];
	display_pw_mat = new double[dimx*dimy/2];

	dim = dimx*dimy;
	for (i = 0; i < dim; ++i) display_read_mat[i] = (double)(0.0);

	GetDesktopResolution(desktop_dimx, desktop_dimy);

	sidew = (unsigned int)(0.45*(float)(desktop_dimy));
	
	name = "read image";

	read_image_window = new window_display_opencv(name);
	active_figure = read_image_window;

	active_figure->create_display_window(sidew, sidew);
	active_figure->id = 0;
	active_figure->bind(display_read_mat_dimx, display_read_mat_dimy, display_read_mat);
	active_figure->setscale(1, 1);
	top_left[0] = 0.0; top_left[1] = 0.0;
	bottom_right[0] = (double)(display_read_mat_dimx);
	bottom_right[1] = (double)(display_read_mat_dimy);
	active_figure->set_display_area(top_left, bottom_right);
	active_figure->movewindow((float)(0.6), (float)(0.0));

	figure.push_back(active_figure);

	name = "read image histogram";
	read_image_histogram_window = new window_display_opencv(name);
	bar_histogram = new barplot_opencv;
	active_figure = read_image_histogram_window;
	active_figure->create_display_window(sidew, sidew);
	active_figure->id = 1;
	active_figure->bind(0, 0, NULL);
	active_figure->setscale(1, 1);
	figure.push_back(active_figure);

	x_histo = new double[65536];
	f_histo = new double[65536];
	bar_histogram->dimpoint = 65536;
	bar_histogram->x = x_histo;
	bar_histogram->y = f_histo;
	for (i = 0; i < bar_histogram->dimpoint; ++i)
	{
		x_histo[i] = (double)(i);
		f_histo[i] = (double)(i);
	}

	plot(read_image_histogram_window->id, bar_histogram);

	name = "power spectrum";

	pw_window = new window_display_opencv("power spectrum");
	active_figure = pw_window;

	active_figure->create_display_window(sidew, sidew);
	active_figure->id = 2;
	active_figure->bind(display_read_mat_dimx, display_read_mat_dimy/2, display_pw_mat);
	active_figure->setscale(1, 1);
	top_left[0] = 0.0; top_left[1] = 0.0;
	bottom_right[0] = (double)(display_read_mat_dimx);
	bottom_right[1] = (double)(display_read_mat_dimy/2);
	active_figure->set_display_area(top_left, bottom_right);
	active_figure->movewindow((float)(0.4), (float)(0.0));
	pw_dist_text = new text_relative_opencv;
	pw_dist_text->deselect();
	pw_dist_text->x[0] = 0.2;
	pw_dist_text->x[1] = 0.1;
	active_figure->overlay.push_back(( (overlay_opencv*)(pw_dist_text) ));

	figure.push_back(active_figure);

	//AHAHAHAHAHAHAHAHAHAHAH
	flg_display_read = false;
}

void display_read(unsigned __int16 *img)
{
	INDEX i, dim;
	unsigned int posmin, posmax;
	unsigned __int16 min, max;
	double zero = 0.0;
	double one = 1.0;
	unsigned int *histptr;
	size_t bin_number;

	dim = display_read_mat_dimx * display_read_mat_dimy;
	for (i = 0; i < dim; ++i) display_read_mat[i] = (double)(img[i]);

	maxminary<unsigned int, unsigned __int16>(dim, img, max, posmax, min, posmin);

	delete[] x_histo;
	delete[] f_histo;
	bin_number = max;
	read_image_histogram.init(1, &bin_number, &zero, &one);
	read_image_histogram.update(dim, display_read_mat);

	x_histo = new double[bin_number];
	f_histo = new double[bin_number];
	histptr = read_image_histogram.freq.ptr();

	for (i = 0; i < bin_number; ++i)
	{
		x_histo[i] = (double)(i);
		f_histo[i] = (double)(histptr[i]);
	}

	bar_histogram->dimpoint = bin_number;
	bar_histogram->x = x_histo;
	bar_histogram->y = f_histo;

	adapt_plot_area_to_function(read_image_histogram_window, bar_histogram->dimpoint, bar_histogram->x, bar_histogram->y);

	active_figure = read_image_window;
	read_image_window->show();
	read_image_histogram_window->show();

	cv::waitKey(10);

	/*
	INDEX dim_radial_lut;
	radial_lut_dimr_from_max_freq(display_read_mat_dimy, display_read_mat_dimx, (float)(useri.frequency_max), dim_radial_lut);
	unsigned int *ram_radial_lut = new unsigned int[dim_radial_lut];
	double *tmp = new double[dim_radial_lut];
	radial_storage_lut(display_read_mat_dimy, display_read_mat_dimx, dim_radial_lut, ram_radial_lut);

	radial_lut_normal_to_rad(dim, dim_radial_lut, ram_radial_lut, display_read_mat, tmp);
	radial_lut_rad_to_normal(dim, dim_radial_lut, ram_radial_lut, tmp, display_read_mat);
	
	read_image_window->show();

	delete[] tmp;
	delete[] ram_radial_lut;
	*/
	
}

void display_power_spectrum_r(INDEX dist, STORE_REAL *pw)
{
	INDEX i, dim;
	std::stringstream sstr;
	sstr << "dist = " << dist;
	pw_dist_text->text = sstr.str();
	radial_lut_rad_to_normal(display_read_mat_dimx * display_read_mat_dimy / 2, dim_radial_lut , ram_radial_lut, pw , display_pw_mat);
	dim = display_read_mat_dimx * display_read_mat_dimy / 2;
	for (i = 0; i < dim; ++i)
		if (display_pw_mat[i] > 0)
			display_pw_mat[i] = std::log(display_pw_mat[i]);
		else
			display_pw_mat[i] = 0.0;
	pw_window->show();
}

void display_power_spectrum_normal(INDEX dist, STORE_REAL *pw)
{
	INDEX i, dim;
	std::stringstream sstr;
	sstr << "dist = " << dist;
	pw_dist_text->text = sstr.str();
	dim = display_read_mat_dimx * display_read_mat_dimy / 2;
	for (i = 0; i < dim; ++i) display_pw_mat[i] = (double)(pw[i]);
	pw_window->show();
	waitkeyboard(0);
}

void close_display()
{
	close_figure_enviroment();
	if (NULL != display_read_mat)	delete[] display_read_mat;
	if (NULL != x_histo) delete[] x_histo;
	if (NULL != f_histo) delete[] f_histo;
	if (NULL != display_pw_mat) delete[] display_pw_mat;
	if (NULL != x_avg_counter) delete[] x_avg_counter;
	if (NULL != y_avg_counter) delete[] y_avg_counter;
	if (NULL != display_execution_map_matrix) delete[] display_execution_map_matrix;
	if (NULL != display_radial_lut_matrix) delete[] display_radial_lut_matrix;
	if (NULL != display_azhavg_x) delete[] display_azhavg_x;
	if (NULL != display_azhavg_y) delete[] display_azhavg_y;

}

void display_execution_map(INDEX dimfilelist, unsigned char *exe_map)
{
	INDEX i, dim;
	
	dim = dimfilelist*dimfilelist;

	if (NULL == display_execution_map_matrix) display_execution_map_matrix	= new double[dim];

	for (i = 0; i < dim; ++i) display_execution_map_matrix[i] = (double)(exe_map[i]);

	execution_map_window = new_figure(dimfilelist, dimfilelist, display_execution_map_matrix);

}
void update_execution_map(INDEX dimfilelist, unsigned char* exe_map)
{
	INDEX i, dim;

	dim = dimfilelist * dimfilelist;
	if (NULL == display_execution_map_matrix) return;
	for (i = 0; i < dim; ++i) display_execution_map_matrix[i] = (double)(exe_map[i]);
	execution_map_window->show();
}

void display_radial_lut(INDEX dimy, INDEX dimx, INDEX dim_radial_lut, unsigned int* lut)
{
	INDEX i,j, dim;
	dim = dimx * dimy;
	if (NULL == display_execution_map_matrix) display_radial_lut_matrix = new double[dim];
	for (i = 0; i < dim; ++i)
	{
		display_radial_lut_matrix[i] = 0;
	}

	for (i = 0; i < dim_radial_lut; ++i)
	{
		display_radial_lut_matrix[lut[i]] = (double)(i);
	}

	radial_lut_window = new_figure(dimx, dimy, display_radial_lut_matrix);

}

void display_open_azhavg(INDEX dimr)
{
	INDEX i;
	scatter_opencv* scatter = new scatter_opencv;
	
	if (NULL == display_azhavg_y && display_azhavg_x == NULL)
	{
		display_azhavg_x = new double[dimr];
		display_azhavg_y = new double[dimr];
	}
	else
		return;

	for (i = 0; i < dimr; ++i)
	{
		display_azhavg_x[i] = (double)(i);
		if (0 < display_azhavg_x[i])
			display_azhavg_y[i] = std::log(display_azhavg_x[i]);
		else
			display_azhavg_y[i] = 0.0;
	}
	
	scatter->dimpoint = dimr;
	scatter->x = display_azhavg_x;
	scatter->y = display_azhavg_y;
	azhavg_window = plot(-1, scatter);
	waitkeyboard(100);
}

void display_update_azhavg(INDEX dimr, double *azhavg)
{
	INDEX i;
	for (i = 0; i < dimr; ++i)
		if(0< azhavg[i])
			display_azhavg_y[i] = std::log(azhavg[i]);
		else
			display_azhavg_y[i] = 0.0;
	azhavg_window->show();
}