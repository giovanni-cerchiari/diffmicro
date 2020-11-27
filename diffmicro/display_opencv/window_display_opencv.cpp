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

#include <thread> 

#include "my_math.h"

#include "window_display_opencv.h"
#include "window_history_opencv.h"


window_display_opencv::window_display_opencv(std::string _name)
{
	if (0 == _name.size())
	{
		std::cerr << "error in: \"window_display_opencv::window_display_opencv\" opencv window must have a name" << std::endl;
		name = "window_display_opencv";
	}
	else
	{
		name = _name;
	}

	id = 0;
	this->init();
	cv::namedWindow(name, CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);// Create a window for display.
	cv::setMouseCallback(name, mouse_opencv::mouse_opencv_callback, (void *)(this));
}

window_display_opencv::~window_display_opencv()
{
	std::vector<control_opencv*>::iterator itct;
	this->clear();

	while (0 != control.size())
	{
		itct = --(control.end());
		delete *itct;
		control.pop_back();
	}
	
	cv::destroyWindow(name);
}
window_display_opencv& window_display_opencv::operator=(const window_display_opencv &wd)
{
	int i;

	for (i = 0; i < 4; ++i)
	{
		this->image_display[i] = wd.image_display[i];
		this->image[i] = wd.image[i];

		this->m_min[i] = wd.m_min[i];
		this->m_max[i] = wd.m_max[i];
	}
	this->name = wd.name;

	for (i = 0; i < 2; ++i)
	{
		this->scale[i] = wd.scale[i];
		this->scale_display[i] = wd.scale_display[i];
		this->start_display[i] = wd.start_display[i];
	}
	

	this->dimx = wd.dimx;
	this->dimy = wd.dimy;
	this->id = wd.id;

	this->active_overlay = wd.active_overlay;
	this->overlay = wd.overlay;

	this->active_control = wd.active_control;
	this->control = wd.control;

	return *this;
}
void window_display_opencv::create_display_window(unsigned int display_dimx, unsigned int display_dimy)
{
	unsigned int i;
	for (i = 0; i < 4; ++i)
	{
	image_display[i].deallocate();
	image_display[i].create(display_dimy, display_dimx, CV_8U);
	}
}

void window_display_opencv::bind(unsigned int _dimx, unsigned int _dimy, double *img, unsigned int channel)
{
	if (NULL == img) return;
	if ((channel > dimc) && (3<channel))
	{
		std::cerr << "error in bind image to window" << std::endl;
		return;
	}

	if (0 == channel)
	{
		dimx = _dimx;
		dimy = _dimy;
	}

//	std::cout << "step 0->" << image_display.step[0] << "\t1->" << image_display.step[1] << std::endl;

	image[channel] = img;
	if(channel == dimc) dimc = channel+1;
}


void window_display_opencv::intensity_scale(bool _auto_scale, double _min, double _max)
{
	control[ID_BUTTON_AUTOSCALE]->set_value(&_auto_scale);

	m_min[0] = _min;
	m_max[0] = _max;
}

void window_display_opencv::movewindow(int x, int y)
{
	int dimx, dimy;
	GetDesktopResolution(dimx, dimy);
	if ((x >= 0) && (y >= 0) && (x < dimx) && (y < dimy))
	{
		cv::moveWindow(name, x,y);
	}
}

void window_display_opencv::movewindow(float x, float y)
{
	int dimx, dimy;
	GetDesktopResolution(dimx, dimy);
	if ((x >= 0) && (y >= 0) && (x < 1) && (y < 1))
	{
		this->movewindow((int)(x*(float)(dimx)), (int)(y*(float)(dimy)));
	}
}

void window_display_opencv::show()
{
	unsigned int i, j, dim;
	float norm;
	unsigned __int8 *val;

	bool flg_colormap;
	bool flg_overlay;
	bool flg_y_orientation;
	unsigned int c;
	double range;

	unsigned int n_pixel;
	double avg;
	double stddev;

	std::list<overlay_opencv*>::iterator over_it, over_end;

	control[ID_BUTTON_YFLIP]->get_value(&flg_y_orientation);
	
	if (true == flg_y_orientation)
	{
		if (0 < scale_display[1]) this->y_flip();
	}
	else
	{
		if (0 > scale_display[1]) this->y_flip();
	}

	min_max_images();
	range = 255;
	fill_display_matrices(range);


	dim = image_display[0].cols*image_display[0].rows;
	switch (dimc)
	{
	case 1:
		control[ID_BUTTON_COLORMAP]->get_value(&flg_colormap);
		if (true == flg_colormap) cv::applyColorMap(image_display[0], image_display_color, colormap);
		else                       cv::cvtColor(image_display[0], image_display_color, CV_GRAY2RGB);

		break;
	case 2:
		control[ID_BUTTON_COLORMAP]->get_value(&flg_colormap);
		if (true == flg_colormap)
		{
			cv::applyColorMap(image_display[1], image_display_color, colormap);
			for (i = 0; i < dim; ++i)
			{
				norm = (float)(image_display[0].data[i]) / (float)(255.);
				val = &(image_display_color.data[i * 3]);
				for (j = 0; j<3; ++j) val[j] = (unsigned __int8)((float)(val[j])*norm);
			}
		}
		else
		{
		//	cv::cvtColor(image_display[0], image_display_color, CV_GRAY2RGB);
			cv::applyColorMap(image_display[0], image_display_color, CV_GRAY2RGB);
			
			for (i = 0; i < dim; ++i)
			{
				image_display_color.data[i * 3 + 0] = image_display[0].data[i];
				image_display_color.data[i * 3 + 1] = image_display[1].data[i];
				image_display_color.data[i * 3 + 2] = 0;
			}
			
		}
		break;
	case 3:
		cv::cvtColor(image_display[0], image_display_color, CV_GRAY2RGB);

		for (i = 0; i < dim; ++i)
		{
			image_display_color.data[i * 3 + 1] = image_display[1].data[i];
			image_display_color.data[i * 3 + 2] = image_display[2].data[i];
		}
		break;
	default:
		cv::cvtColor(image_display[0], image_display_color, CV_GRAY2RGB);
		break;
	}



	control[ID_BUTTON_OVERLAY]->get_value(&flg_overlay);
	if (true == flg_overlay)
	{
		over_end = overlay.end();
		for (over_it = overlay.begin(), i=0; over_it != over_end; ++over_it)
		{
			(*over_it)->draw(image_display_color, start_display, scale_display);
			if ((OVERLAY_CIRLE_OPENCV == (*over_it)->type) && (0<dimc))
			{
				circle_opencv *ccc = (circle_opencv *)(*over_it);
				ccc->drawtext(20,image_display[0].rows-20*(++i),image_display_color,dimx,dimy,scale[0], scale[1],image[0],n_pixel,avg,stddev);
			}
		}
	}

	if(this == active_figure) mouse.draw(image_display_color, start_display, scale_display);
	window_history->show(*this);
	
	imshow(name, image_display_color);

}

void window_display_opencv::setscale(double scalex, double scaley)
{
	double epsilon = 1./10000000000;

	if (epsilon > fabs(scalex) || epsilon > fabs(scaley))
	{
		scalex = 1.0;
		scaley = 1.0;
	}

	scale[0] = scalex;
	scale[1] = scaley;

	scale_display[0] = scalex;
	scale_display[1] = scaley;
}

void window_display_opencv::setdisplayscale(double disp_scalex, double disp_scaley)
{
	scale_display[0] = disp_scalex;
	scale_display[1] = disp_scaley;
}


void window_display_opencv::set_display_area(double top_left[], double bottom_right[])
{
	unsigned int i;
	bool flip = false;

	unsigned int dim_x, dim_y;

/*	if (0 != this->dimx) dim_x = this->dimx;
	else                 dim_x = this->image_display[0].cols;
	if (0 != this->dimy) dim_y = this->dimy;
	else                 dim_y = this->image_display[0].rows;
	*/
	dim_x = this->image_display[0].cols;
 dim_y = this->image_display[0].rows;

	for (i = 0; i < 2; ++i)
	{
		if (0.000000001>fabs(top_left[i] - bottom_right[i])) return;
	}

	for (i = 0; i < 2; ++i) start_display[i] = top_left[i] / scale[i];

	scale_display[0] = (bottom_right[0] - top_left[0]) / ((double)(dim_x)*scale[0]);

	if (0 > scale_display[1]) flip = true;
	scale_display[1] = (bottom_right[1] - top_left[1]) / ((double)(dim_y)*fabs(scale[1]));

	if (true == flip)
	{
		scale[1] *= -1.;
		this->y_flip();
	}
		
}

void window_display_opencv::zoom(double zoomx, double zoomy)
{
	double epsilon = 1. / 10000000000;

	if (epsilon > fabs(zoomx) || epsilon > fabs(zoomy))
	{
		zoomx = 1.0;
		zoomy = 1.0;
	}

	scale_display[0] /= zoomx;
	scale_display[1] /= zoomy;

}

void window_display_opencv::y_flip()
{
	start_display[1] += (double)(this->image_display[0].rows)*scale_display[1];

	scale[1] = fabs(scale[1]);
	scale_display[1] *= -1;
}

double window_display_opencv::minval()
{
	min_max_images();
	return m_min[0];
}
double window_display_opencv::maxval()
{
	min_max_images();
	return m_max[0];
}



	void window_display_opencv::init()
	{
		int i;

		for (i = 0; i < 4; ++i)
		{
			image_display[i].create(1, 1, CV_8UC3);

			m_min[i] = 0.0;
			m_max[i] = 0.0;
		}
		dimx = 0;
		dimy = 0;
		for (i = 0; i < 2; ++i)
		{
			scale[i] = 1.0;
			scale_display[i] = 1.0;
			start_display[i] = 0.0;
		}
		dimc = 0;
		for (i = 0; i<4; ++i) image[i] = NULL;
		active_overlay = NULL;

		this->init_controls();

		colormap = cv::COLORMAP_JET;

	}

	void window_display_opencv::init_controls()
	{
		int i;
		float sizex_button;
		float sizey_button;
		button_opencv *button;
		single_value_trackbar *single_track;
		minmax_trackbar *minmax_track;
		int value_track[4];
		std::vector<control_opencv*>::iterator it;
		bool flg;

		sizex_button = 1. / 3.;
		sizey_button = sizex_button / 2.;

		button = new button_opencv(ID_BUTTON_AUTOSCALE, "auto scale");
		flg = true; button->set_value(&flg);
		control.push_back(button);

		button = new button_opencv(ID_BUTTON_COLORMAP, "colormap");
		button->label_false = "grayscale";
		button->label_true = "colormap";
		flg = false; button->set_value(&flg);
		control.push_back(button);

		button = new button_opencv(ID_BUTTON_CONTOUR, "contourplot");
		flg = false; button->set_value(&flg);
		control.push_back(button);

		button = new button_opencv(ID_BUTTON_OVERLAY, "drawings");
		button->label_false = "drawing hidden";
		button->label_true = "drawing shown";
		flg = true; button->set_value(&flg);
		control.push_back(button);

		button = new button_opencv(ID_BUTTON_YFLIP, "flip y");
		if (0 > scale[1])
		{
			flg = true; button->set_value(&flg);
		}
		else
		{
			flg = false; button->set_value(&flg);
		}
		control.push_back(button);

		button = new button_opencv(ID_BUTTON_MODE, "mode");
		button->label_false = "drawing mode";
		button->label_true = "plot mode";
		flg = false; button->set_value(&flg);
		control.push_back(button);

		i = 0;
		for (it = control.begin(); it != control.end(); ++it)
		{
			(*it)->x[0] = 0.0;
			(*it)->x[2] = 0.9*sizex_button+(*it)->x[0];
			(*it)->x[1] = (float)(i)*sizey_button;
			(*it)->x[3] = 0.9*sizey_button + (*it)->x[1];
			++i;
		}

		minmax_track = new minmax_trackbar;
		minmax_track->name = "minmax";
		minmax_track->id = ID_CTRL_MINMAX_TRACKBAR;
		value_track[0] = 32; value_track[1] = 0; value_track[2] = 16; value_track[3] = 32;
		minmax_track->set_value(value_track);
		minmax_track->x[0] = sizex_button;
		minmax_track->x[1] = 0.0;
		minmax_track->x[2] = 1.0;
		minmax_track->x[3] = 0.9*sizey_button;
		control.push_back(minmax_track);
		/*
		single_track = new single_value_trackbar;
		single_track->name = "minmax";
		single_track->id = ID_CTRL_MINMAX_TRACKBAR;
		value_track[0] = 32; value_track[1] = 16;
		single_track->set_value(value_track);
		single_track->x[0] = sizex_button;
		single_track->x[1] = 0.0;
		single_track->x[2] = 1.0;
		single_track->x[3] = 0.9*sizey_button;
		control.push_back(single_track);*/

		active_control = NULL;
	}


	void window_display_opencv::erase_overlay(overlay_opencv* ov)
	{
		std::list<overlay_opencv*>::iterator it,end;

		end = overlay.end();
		for (it = overlay.begin(); it != end; ++it)
		{
			if (ov == *it)
			{
				if (active_overlay == ov)	active_overlay = NULL;
				overlay.erase(it);
				delete ov;
				break;
			}
		}

	}

	void window_display_opencv::clear()
	{
		unsigned int i;

		std::list<overlay_opencv*>::iterator itov;

		for(i=0; i<dimc; ++i) image_display[i].deallocate();

		while (0 != overlay.size())
		{
			itov = overlay.begin();
			delete *itov;
			itov = overlay.erase(itov);
		}

	}





	void window_display_opencv::fill_display_matrices(double range)
	{
		unsigned int c;
		unsigned int cols, rows;
		unsigned int i, k;
		unsigned int ii, kk;
		unsigned int iii, iiid;
		unsigned __int8 *dataptr;
		double tmp;

		rows = image_display[0].rows;
		cols = image_display[0].cols;

		for (c = 0; c < 4; ++c) memset(image_display[c].data, 0, image_display[c].total()*image_display[c].elemSize());

		if (0 == dimc) return;

		for (k = 0; k < (unsigned int)(rows); ++k)
		{
			kk = (unsigned int)(std::round(scale[1] * ((double)(k)*scale_display[1] + start_display[1])));
			if (dimy>kk)
			{

				for (i = 0; i < (unsigned int)(cols); ++i)
				{
					ii = (unsigned int)(std::round(scale[0] * ((double)(i)*scale_display[0] + start_display[0])));
					if (dimx>ii)
					{
						iii = kk*dimx + ii;
						iiid = k * image_display[0].step[0] + i;
						for (c = 0; c < dimc; ++c)
						{
							tmp = range*(image[c][iii] - m_min[c]) / (m_max[c] - m_min[c]);
							//tmp = image[kk*dimx + ii];
							if (0 <= tmp)
							{
								dataptr = (unsigned __int8*)(&image_display[c].data[iiid]);
								//tmp *= gain;
//								if (range >= tmp)	(unsigned __int8)(image_display[c].data)[iiid] = (unsigned char)(tmp);
//								else         					(unsigned __int8)(image_display[c].data)[iiid] = 255;
								if (range >= tmp)	dataptr[0] = (unsigned char)(tmp);
								else         		dataptr[0] = 255;
							}
						}
					}
				}
			}
		}

	}

	void window_display_opencv::min_max_images()
	{
		unsigned int c;
		unsigned int i,dim;
		unsigned int pos_min[4];
		unsigned int pos_max[4];
		double m;
		double mmax, mmin;
		bool flg_autoscale;
		int value[4];
		double step, begin;

		if (0 == dimc)
		{
			for (c = 0; c < dimc; ++c)
			{
				m_min[c] = 0.0;
				m_max[c] = 255;
			}
			return;
		}


		control[ID_BUTTON_AUTOSCALE]->get_value(&flg_autoscale);
		if (true == flg_autoscale)
		{


			dim = dimx*dimy;

			for (c = 0; c < dimc; ++c)
			{
				maxminary(dim, image[c], m_max[c], pos_max[c], m_min[c], pos_min[c]);
			}

			/*
			switch (dimc)
			{
			case 1:
				maxminary(dim, image[0], max[0], pos_max[0], min[0], pos_min[0]);

				break;
			case 2:
			{
										std::thread th0(maxminary, dim, image[0], max[0], pos_max[0], min[0], pos_min[0]);
										std::thread th1(maxminary, dim, image[1], max[1], pos_max[1], min[1], pos_min[1]);
										th0.join();
										th1.join();
			}
				break;
				{
				}
			case 3:
			{
			}
				break;
			case 4:
			{
			}
				break;

			}*/

			maxminary(dimc, m_min,    m, pos_max[0], mmin, pos_min[0]);
			maxminary(dimc, m_max, mmax, pos_max[0],    m, pos_min[0]);

			bool flg_colormap;
			control[ID_BUTTON_COLORMAP]->get_value(&flg_colormap);
			if (false == flg_colormap)
			{
				for (c = 0; c < dimc; ++c)
				{
					m_max[c] = mmax;
					m_min[c] = mmin;
				}
			}


			control[ID_CTRL_MINMAX_TRACKBAR]->get_value(value);

			for (c = 0; c < dimc; ++c)
			{
				begin = m_min[c];
				step = (m_max[c] - m_min[c]) / (double)(value[0]);

				m_min[c] = (double)(value[1]) *step + begin;
				m_max[c] = (double)(value[3]) *step + begin;
			}
		}
		else
		{
			for (c = 0; c < dimc; ++c)
			{
				//m_min[c] = 0.0;
				//m_max[c] = 65536.0;
				control[ID_CTRL_MINMAX_TRACKBAR]->get_value(value);

				begin = 0.0;
				step = (4096.) / (double)(value[0]);

				m_min[c] = (double)(value[1]) *step + begin;
				m_max[c] = (double)(value[3]) *step + begin;
			}
		}

	}

	void window_display_opencv::save_display(std::string filename)
	{
		cv::imwrite(filename, this->image_display_color);
	}