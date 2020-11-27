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
#include "overlay_opencv.h"


overlay_opencv* select_overlay(int i_to_select, std::list<overlay_opencv*> lst)
{
	int i;
	std::list<overlay_opencv*>::iterator it, end;
	overlay_opencv* ov = NULL;

	end = lst.end();
	for (it = lst.begin(), i=0; it != end; ++it, ++i)
	{
		if (i_to_select != i) (*it)->deselect();
		else
		{
			ov = *it;
			ov->select();
		}
	}

	return ov;
}

void overlay_opencv::select()
{
	selected = true;
	color = &color_selection;
}
void overlay_opencv::deselect()
{
	selected = false;
	color = &color_normal;
}

circle_opencv::circle_opencv()
{
	name = "circle";
	type = OVERLAY_CIRLE_OPENCV;
	color_normal = cv::Scalar(0, 0, 255);
	color_selection = cv::Scalar(255, 0, 255);
	draw_type = GEOMETRIC_SHAPE_DRAW_TYPE_VERBOSE;
	this->deselect();
	this->init();
}

void circle_opencv::init()
{
	int i;
	for (i = 0; i < 4; ++i) x[i] = 0.0;
	r = 0.0;
}

const circle_opencv& circle_opencv::operator=(const circle_opencv& cr)
{
	int i;
	for (i = 0; i < 4; ++i) this->x[i] = cr.x[i];
	this->r = cr.r;
	this->id = cr.id;
	this->color = cr.color;
	this->color_normal = cr.color_normal;
	this->color_selection = cr.color_selection;
	this->selected = cr.selected;

	this->name = cr.name;
	this->type = cr.type;
	this->draw_type = cr.draw_type;
	return *this;
}

void circle_opencv::draw(cv::Mat &img, double start[], double scale[])
{
	cv::Point p[2];
	cv::Size axes;
	int i,radius;
	double dx[2];
	std::string text;
	std::stringstream ss;
	
	for (i = 0; i < 2; ++i)
	{
		p[i].x = (int)((x[0 + 2*i] - start[0]) / scale[0]);
		p[i].y = (int)((x[1 + 2*i] - start[1]) / scale[1]);
	}
	
//	for (i = 0; i < 2; ++i) dx[i] = (x[i] - x[i + 2])/scale[i];
//	radius = (int)(sqrt(dx[0] * dx[0] + dx[1] * dx[1]));

	axes.width = fabs(r/scale[0]);
	axes.height = fabs(r/scale[1]);
	cv::ellipse(img, p[0], axes, 0.0, 0, 360, *color, 2);// , int lineType = 8, int shift = 0);

//	cv::circle(img, p[0], radius, *color,2);

	if (draw_type == GEOMETRIC_SHAPE_DRAW_TYPE_VERBOSE)
	{
		cv::line(img, p[0], p[1], *color, 2);

		ss << r;
		text = ss.str();
		cv::putText(img, text, p[1] + cv::Point(20, 0), cv::FONT_HERSHEY_SIMPLEX, 0.6, *color, 2);
	}
}

void circle_opencv::drawtext(int posx, int posy, cv::Mat &img, unsigned int dimx, unsigned int dimy, double scalex, double scaley, double *mat, unsigned int &n_pixel, double &avg, double &stddev)
{
	std::stringstream text;
	cv::Point p;
	this->average(dimx, dimy, scalex, scaley, mat, n_pixel, avg, stddev);
	text.str("");
	text <<"r = " << r << "  avg = " << avg << "  integ " <<  avg*(double)(n_pixel);
	p.x = posx;
	p.y = posy;
	cv::putText(img, text.str(), p, cv::FONT_HERSHEY_SIMPLEX, 0.6, *color, 2);
}


void circle_opencv::edit(void* in)
{
	int i;
	double *xx = (double*)(in);
	double dx[2];

	for (i = 0; i < 4; ++i) x[i] = xx[i];
	for (i = 0; i < 2; ++i) dx[i] = x[i] - x[i+2];
	r = sqrt(dx[0] * dx[0] + dx[1] * dx[1]);
}

void circle_opencv::getvalue(return_get_value_overlay_opencv *out)
{
	out->type = OOV_TYPE_DOUBLE;
	out->val = &this->r;
}

void circle_opencv::average(unsigned int dimx, unsigned int dimy, double scalex, double scaley, double *mat, unsigned int &n_pixel, double &avg, double &stddev)
{
	unsigned int dim, i, j;
	unsigned int starti, endi;
	unsigned int startj, endj;
	double coeff1, coeff2, val;
	double x2, y2, r2;

	r2 = r*r;

	n_pixel = (unsigned int)(0);
	avg = 0.0;
	stddev = 0.0;

	if (false == find_interval(dimx, x[0], scalex, starti, endi)) return;
	if (false == find_interval(dimy, x[1], scaley, startj, endj)) return;

//	std::cerr << "start = " << starti << "\t end = " << endi << std::endl;

	n_pixel = 0;
	for (j = startj; j <= endj; ++j)
	{
		y2 = ((double)(j) / scaley)-x[1];
		y2 *= y2;
		for (i = starti; i <= endi; ++i)
		{
//			std::cerr << i << " = " << ((double)(i) / scalex) << std::endl;
			x2 = ((double)(i) / scalex)-x[0];
			if (r2>=x2*x2+y2)
			{
				val = mat[j*dimx + i];

				coeff1 = (double(n_pixel));
				++n_pixel;
				coeff2 = (double)(1.0) / (double)(n_pixel);
				coeff1 *= coeff2;

				avg    = coeff1*avg    + coeff2*val;
				stddev = coeff1*stddev + coeff2*val*val;
			}
		}
	}

	stddev = sqrt(stddev - avg*avg);

}

bool circle_opencv::find_interval(unsigned int dim, double pos, double scale, unsigned int &start, unsigned int &end)
{
	int tmp;

	tmp = std::floor((pos - r)*scale);
	if (0 <= tmp) start = (unsigned int)(tmp);
	else        	 start = 0;

	if (start > (dim-1)) return false;

	tmp = (int)(std::ceil((pos + r)*scale));
	if (0 > tmp) return false;
	if (dim - 1 > tmp) end = (unsigned int)(tmp);
	else              	 end = dim-1;

	return true;
}

//----------------------------------------------------------------------------------
rectangle_opencv::rectangle_opencv()
{
	name = "rectangle";
	type = OVERLAY_RECTANGLE_OPENCV;
	color_normal = cv::Scalar(0, 255, 255);
	color_selection = cv::Scalar(255, 0, 255);
	draw_type = GEOMETRIC_SHAPE_DRAW_TYPE_VERBOSE;

	this->init();
}

void rectangle_opencv::init()
{
	int i;
	for (i = 0; i < 4; ++i) x[i] = 0.0;
	selected = false;
	color = &color_normal;
}

const rectangle_opencv& rectangle_opencv::operator=(const rectangle_opencv& cr)
{
	int i;
	for (i = 0; i < 4; ++i) this->x[i] = cr.x[i];

	this->id = cr.id;
	this->color = cr.color;
	this->color_normal = cr.color_normal;
	this->color_selection = cr.color_selection;
	this->selected = cr.selected;

	this->name = cr.name;
	this->type = cr.type;
	this->draw_type = cr.draw_type;

	return *this;
}

void rectangle_opencv::draw(cv::Mat &img, double start[], double scale[])
{
	cv::Point p[2];
	cv::Point txtp;
	int i;
	double dx[2];
	float radius;
	std::string text;
	std::stringstream ss;

	for (i = 0; i < 2; ++i)
	{
		dx[i] = x[i + 2] - x[i];
		p[i].x = (int)((x[0 + 2 * i] - start[0]) / scale[0]);
		p[i].y = (int)((x[1 + 2 * i] - start[1]) / scale[1]);
	}

	radius = sqrt(dx[0] * dx[0] + dx[1] * dx[1]);

	cv::rectangle(img, p[0], p[1], *color, 2);

	if (draw_type == GEOMETRIC_SHAPE_DRAW_TYPE_VERBOSE)
	{
		cv::line(img, p[0], p[1], *color, 2);

		ss << radius;
		text = ss.str();
		txtp.x = p[1].x + 20;  txtp.y = p[1].y;
		cv::putText(img, text, txtp, cv::FONT_HERSHEY_SIMPLEX, 0.6, *color, 2);

		ss.str("");
		ss << x[2] - x[0];
		text = ss.str();
		txtp.x = p[1].x + 20;  txtp.y = p[0].y;
		cv::putText(img, text, txtp, cv::FONT_HERSHEY_SIMPLEX, 0.6, *color, 2);

		ss.str("");
		ss << x[3] - x[1];
		text = ss.str();
		txtp.x = p[0].x;  txtp.y = p[1].y + 20;
		cv::putText(img, text, txtp, cv::FONT_HERSHEY_SIMPLEX, 0.6, *color, 2);
	}
}

void rectangle_opencv::edit(void* in)
{
	int i;
	double *xx = (double*)(in);

	for (i = 0; i < 4; ++i) x[i] = xx[i];

}

void rectangle_opencv::getvalue(return_get_value_overlay_opencv *out)
{
	out->type = OOV_TYPE_DOUBLE;
	returngetvalue = this->x[2] - this->x[0];
	out->val = &returngetvalue;
}

//-----------------------------------------------------------------------------------------------
contour_opencv::contour_opencv()
{
	name = "contour";
	type = OVERLAY_CONTOUR_OPENCV;
	color_normal = cv::Scalar(255, 255, 255);
	color_selection = cv::Scalar(255, 0, 255);
	dimpoint = 0;
	marker_size = 3;
	x = NULL;
	this->init();
}
contour_opencv::~contour_opencv()
{
	if (NULL != x) delete[] x;
}

const contour_opencv& contour_opencv::operator=(const contour_opencv& cr)
{
	unsigned int i;
	this->dimpoint = cr.dimpoint;
	for (i = 0; i < 2*this->dimpoint; ++i) this->x[i] = cr.x[i];

	this->id = cr.id;
	this->color = cr.color;
	this->color_normal = cr.color_normal;
	this->color_selection = cr.color_selection;
	this->selected = cr.selected;

	this->name = cr.name;
	this->type = cr.type;

	return *this;
}

void contour_opencv::init()
{
	if (NULL != x) delete[] x;
	x = NULL;
	dimpoint = 0;
	capacity = 0;
	selected = false;
	color = &color_normal;

	this->set_marker_type(MARKER_NONE_OPENCV);

}
void contour_opencv::set_marker_type(int _type)
{
	switch (_type)
	{
	case MARKER_NONE_OPENCV:
		marker_draw = NULL;
		break;
	case	MARKER_CIRCLE_OPENCV:
		marker_draw = draw_circle;
		break;
	case	MARKER_RECTANGLE_OPENCV:
		marker_draw = draw_rectangle;
		break;
	case	MARKER_CROSS_OPENCV:
		marker_draw = draw_cross;
		break;
	case	MARKER_TRIANGLE_OPENCV:
		marker_draw = draw_triangle;
		break;
	case	MARKER_DIAMOND_OPENCV:
		marker_draw = draw_diamond;
		break;
	default:
		return;
		break;
	}
	marker_type = _type;
}
void contour_opencv::draw(cv::Mat &img, double start[], double scale[])
{
	unsigned int i, j, k, icol;
	unsigned char* imgdataptr;
	cv::Point xx;

		if (MARKER_NONE_OPENCV == marker_type)
		{
			for (k = 0; k < dimpoint; ++k)
			{
				i = (unsigned int)((x[2 * k] - start[0]) / scale[0]);
				j = (unsigned int)((x[2 * k + 1] - start[1]) / scale[1]);

				if (i < (unsigned int)(img.cols) && j < (unsigned int)(img.rows))
				for (icol = 0; icol < 3; ++icol)
				{
					imgdataptr = (unsigned char*)(&(img.data[j * img.step[0] + i * img.step[1] + icol]));
					imgdataptr[0] = (unsigned char)(color->val[icol]);
				}
			}
		}
		else
		{
			for (k = 0; k < dimpoint; ++k)
			{
				xx.x = (unsigned int)((x[2 * k] - start[0]) / scale[0]);
				xx.y = (unsigned int)((x[2 * k + 1] - start[1]) / scale[1]);
				marker_draw(img, xx, marker_size, color);
			}
		}
			
}

void contour_opencv::push_back(double *xx)
{
	unsigned int i, dim;
	double *todel;
	double *tmp;

	if (capacity <= dimpoint)
	{
		if (NULL != x)
		{
			capacity = dimpoint + 8;
			tmp = new double[2*capacity];
			dim = 2 * dimpoint;
			for (i = 0; i < dim; ++i)	tmp[i] = x[i];

			// removing old memory area
			todel = x;
			x = tmp;
			delete todel;
		}
		else
		{
			dimpoint = 0;
			capacity = dimpoint + 8;
			x = new double[2*capacity];
		}
	}

	x[2 * dimpoint] = xx[0];
	x[2 * dimpoint + 1] = xx[1];
	++dimpoint;
}

void contour_opencv::pop_back()
{
	if ((NULL != x) && (0!=this->dimpoint))--this->dimpoint;
}

void contour_opencv::edit(void *in)
{
	unsigned int i,j;
	contour_opencv_edit_parameters *p = (contour_opencv_edit_parameters *)(in);
	unsigned int *point;

	this->val = p->val;

	equipoints<unsigned int, double>(p->dimx, p->dimy, p->mat, p->val, dimpoint, NULL);
	if (NULL != x) delete[] x;
	x = new double[2*dimpoint];
	capacity = dimpoint;
	point = new unsigned int[2*dimpoint];
	equipoints<unsigned int, double>(p->dimx, p->dimy, p->mat, p->val, dimpoint, point);

	for (i = 0; i < dimpoint; ++i)
	{
		for (j = 0; j < 2; ++j)
		x[2*i + j] = ((double)(point[2 * i + j])+0.5) * p->scale[j];
	}

	delete[] point;
}

void contour_opencv::getvalue(return_get_value_overlay_opencv *out)
{
	out->type = OOV_TYPE_DOUBLE;
	out->val = &this->val;
}

scatter_opencv::scatter_opencv()
{
	name = "scatter";
	type = OVERLAY_SCATTER_OPENCV;
	color_normal = cv::Scalar(255, 255, 255);
	color_selection = cv::Scalar(255, 0, 255);
	dimpoint = 0;
	x = NULL;
	y = NULL;
	errx = NULL;
	erry = NULL;
	this->init();
}

void scatter_opencv::init()
{
	x = NULL;
	y = NULL;
	errx = NULL;
	erry = NULL;
	dimpoint = 0;
	selected = false;
	color = &color_normal;

	marker_size = 4;
	line_thickness = 2;
	line_type_internal_opencv = 8;
	line_type = LINE_NORMAL_OPENCV;

	set_marker_type(MARKER_TRIANGLE_OPENCV);

}

const scatter_opencv& scatter_opencv::operator = (const scatter_opencv& s)
{
	this->dimpoint = s.dimpoint;
	this->x = s.x;
	this->y = s.y;
	this->errx = s.errx;
	this->erry = s.erry;

	this->line_type_internal_opencv = s.line_type_internal_opencv;

	this->marker_size = s.marker_size;
	this->line_thickness = s.line_thickness;
	this->line_type = s.line_type;

	this->name = s.name;
	this->type = s.type;

	this->set_marker_type(s.marker_type);
	return *this;
}

void scatter_opencv::draw(cv::Mat &img, double start[], double scale[])
{
	cv::Point p[2];
	unsigned int i;

	if (MARKER_NONE_OPENCV != marker_type)
	{
		for (i = 0; i < dimpoint; ++i)
		{
			p[0].x = (int)((x[i] - start[0]) / scale[0]);
			p[0].y = (int)((y[i] - start[1]) / scale[1]);
			marker_draw(img, p[0], marker_size, color);
		}

	}

	
	if (0 != line_type)
	{
		p[0].x = (int)((x[0] - start[0]) / scale[0]);
		p[0].y = (int)((y[0] - start[1]) / scale[1]);

		for (i = 1; i < dimpoint; ++i)
		{
			p[1].x = (int)((x[i] - start[0]) / scale[0]);
			p[1].y = (int)((y[i] - start[1]) / scale[1]);
			cv::line(img, p[0], p[1], *color, line_thickness, line_type);
			p[0].x = p[1].x; p[0].y = p[1].y;
		}

	}

}

void scatter_opencv::set_marker_type(int _type)
{
	switch (_type)
	{
	case MARKER_NONE_OPENCV:
		marker_draw = NULL;
		break;
	case	MARKER_CIRCLE_OPENCV:
		marker_draw = draw_circle;
		break;
	case	MARKER_RECTANGLE_OPENCV:
		marker_draw = draw_rectangle;
		break;
	case	MARKER_CROSS_OPENCV:
		marker_draw = draw_cross;
		break;
	case	MARKER_TRIANGLE_OPENCV:
		marker_draw = draw_triangle;
		break;
	case	MARKER_DIAMOND_OPENCV:
		marker_draw = draw_diamond;
		break;
	default:
		return;
		break;
	}
	marker_type = _type;
}


void scatter_opencv::edit(void *in)
{
}

void scatter_opencv::getvalue(return_get_value_overlay_opencv *out)
{
	out->type = OOV_TYPE_STRING;
	out->val = &label;
}



axes_opencv::axes_opencv()
{
	name = "axes";
	type = OVERLAY_AXES_OPENCV;
	color_normal = cv::Scalar(255, 255, 255);
	color_selection = cv::Scalar(255, 0, 255);

	this->init();
}

void axes_opencv::init()
{
	dim_tick[0] = 5;
	dim_tick[1] = 10;

	font_number_size = 0.6;
	font_label_size = 0.6;

	//xlabel = "xlabel";
	//ylabel = "ylabel";
	label_return = "";

	selected = false;
	color = &color_normal;
}

const axes_opencv& axes_opencv::operator = (const axes_opencv& ax)
{
	int i;
	for(i=0; i<2; ++i)	this->dim_tick[i] = ax.dim_tick[i];

	this->font_number_size = ax.font_number_size;

	this->font_label_size = ax.font_label_size;
	this->xlabel = ax.xlabel;
	this->ylabel = ax.ylabel;
	
	this->name = ax.name;
	this->type = ax.type;

	return *this;
}

void axes_opencv::draw(cv::Mat &img, double start[], double scale[])
{
	cv::Point origin, end;
	float origind[2], endd[2];
	double start_tick, step_tick;
	float signum_scale;
	cv::Point tipx;
	cv::Point tipy;

	cv::Point p[2];
	unsigned int i,dim;
	std::stringstream st;

	origin.x = (int)((double)(img.cols) * 0.12);
	end.x = (int)((double)(img.cols) * 0.75);

	if (0 < scale[1])
	{
		origin.y = (int)((double)(img.rows) * 0.12);
		end.y = (int)((double)(img.rows) * 0.90);
		signum_scale = 1.0;
	}
	else
	{
		origin.y = (int)((double)(img.rows) * 0.9);
		end.y = (int)((double)(img.rows) * 0.10);
		signum_scale = -1.0;
	}

	origind[0] = (double)(origin.x)*scale[0]+start[0];
	origind[1] = (double)(origin.y)*scale[1]+start[1];
	endd[0] = (double)(end.x)*scale[0] + start[0];
	endd[1] = (double)(end.y)*scale[1] + start[1];

	//--------------------------------------------
	// X-axis

	p[0].x = end.x;
	p[0].y = origin.y;
	cv::line(img, origin, p[0], *color, 2, 8);
	num_axis(origind[0], endd[0], this->dim_tick[0], dim, start_tick, step_tick);
	
	p[0].x = (int)((double)(end.x) * 0.75);
	if (0 < scale[1]) p[0].y = (int)((float)(img.rows)* 0.05);
	else              p[0].y = (int)((float)(img.rows)* 0.98);
	if(0<xlabel.size())cv::putText(img, xlabel, p[0], cv::FONT_HERSHEY_SIMPLEX, font_label_size, *color, 2);

	p[1].y = (int)((float)(origin.y) + signum_scale*(float)(img.rows)* 0.01);
	for (i = 0; i < dim; ++i)
	{
		p[0].y = (int)((float)(origin.y) - signum_scale*(float)(img.rows)* 0.01);
		p[1].x = p[0].x = (int)((start_tick + step_tick*(double)(i)-start[0]) / scale[0]);
		cv::line(img, p[0], p[1], *color, 2, 8);

		if (0 < scale[1]) p[0].y = (int)((float)(img.rows) * 0.1);
		else              p[0].y = (int)((float)(img.rows) * 0.94);
		st.str("");
		st << (start_tick + step_tick*(double)(i));
		cv::putText(img, st.str(), p[0], cv::FONT_HERSHEY_SIMPLEX, font_label_size, *color);
	}

	//-------------------------------------------------
	// Y - axis

	p[0].x = origin.x;
	p[0].y = end.y;
	cv::line(img, origin, p[0], *color, 2, 8);
	num_axis(origind[1], endd[1], this->dim_tick[1], dim, start_tick, step_tick);

	p[0].x = (int)((float)(img.cols)* 0.07);
	if (0 < scale[1]) p[0].y = (int)((float)(img.rows)* 0.95);
	else              p[0].y = (int)((float)(img.rows)* 0.05);
	if(0<ylabel.size())cv::putText(img, ylabel, p[0], cv::FONT_HERSHEY_SIMPLEX, font_label_size, *color, 2);

	p[1].x = (int)((float)(origin.x) + (float)(img.cols)* 0.01);
	for (i = 0; i < dim; ++i)
	{
		p[0].x = (int)((float)(origin.x) - (float)(img.cols)* 0.01);
		p[1].y = p[0].y = (int)((start_tick + step_tick*(double)(i)-start[1]) / scale[1]);
		cv::line(img, p[0], p[1], *color, 2, 8);

		p[0].x = (int)((float)(img.cols) * 0.01);
		st.str("");
		st << (start_tick + step_tick*(double)(i));
		cv::putText(img, st.str(), p[0], cv::FONT_HERSHEY_SIMPLEX, font_label_size, *color);
	}
	

}

void axes_opencv::edit(void *in)
{
}

void axes_opencv::getvalue(return_get_value_overlay_opencv *out)
{
	out->type = OOV_TYPE_STRING;
	out->val = &label_return;
}

void axes_opencv::num_axis(double origin, double end, unsigned int n_wished_tick, unsigned int &ntick, double &start, double &step)
{
	char cifer;
	double delta;
	double pow10;

	delta = end - origin;

	step = delta / (double)(n_wished_tick);
	pow10 = pow(10, std::floor(std::log(step) / std::log(10.)));
	cifer = (char)(std::floor(step / pow10 + 0.5));

	if (3 < cifer)
	{
		if (7 < cifer) step = 10;
		else           step = 5.;
	}
	else
	{
		if (2 <= cifer) step = 2.;
		else         			step = 1.;
	}

	step *= pow10;

	start = std::ceil(origin / step)*step;

	ntick = (unsigned int)(std::floor((end - start) / step))+1;
}



legend_opencv::legend_opencv()
{
	name = "legend";
	type = OVERLAY_LEGEND_OPENCV;
	color_normal = cv::Scalar(255, 255, 255);
	color_selection = cv::Scalar(255, 0, 255);

	this->init();
}

void legend_opencv::init()
{
	font_size = 0.6;

	label_return = "";

	selected = false;
	color = &color_normal;
}

const legend_opencv& legend_opencv::operator = (const legend_opencv& ax)
{
	int i;

	this->font_size = ax.font_size;

	this->name = ax.name;
	this->type = ax.type;

	return *this;
}

void legend_opencv::draw(cv::Mat &img, double start[], double scale[])
{
	cv::Point origin;

	float signum_scale;
	return_get_value_overlay_opencv retval;
	cv::Point tmp;
	cv::Point step;
	cv::Point offset_text;
	cv::Point offset_line[2];
	cv::Point offset_half;
	//std::string write_label;

	overlay_opencv *ov;
	scatter_opencv *sc;
	barplot_opencv *bp;

	unsigned int i, dim;
	std::stringstream st;

	std::list<overlay_opencv*>::iterator it, end;

	origin.x = (int)((double)(img.cols) * 0.8);
//	end.x = (int)((double)(img.cols) * 0.75);

	step.x = 0;
	step.y = 30;
	offset_text.x = 30;
	offset_text.y = 0;

	offset_line[0].x = 0; offset_line[0].y = -step.y / 4;
	offset_line[1].x = (int)(0.8 *(float)(offset_text.x)); offset_line[1].y = offset_line[0].y;

	offset_half.x = offset_line[1].x;
	offset_half.y = -step.y / 2;

	if (0 < scale[1])
	{
		origin.y = (int)((double)(img.rows) * 0.3);
//		end.y = (int)((double)(img.rows) * 0.90);
		signum_scale = 1.0;
	}
	else
	{
		origin.y = (int)((double)(img.rows) * 0.3);
//		end.y = (int)((double)(img.rows) * 0.10);
		signum_scale = -1.0;
	}

	end = list_overlay->end();
	for (it = list_overlay->begin(); it != end; ++it)
	{
		switch ((*it)->type)
		{
			case OVERLAY_CONTOUR_OPENCV:
				// marker
				cv::circle(img, origin + (offset_line[0] + offset_line[1]) / 2, (offset_line[0].x + offset_line[1].x) / 2, *((*it)->color), 0);
				// label
				(*it)->getvalue(&retval);
				st.str("");
				st << *((double*)(retval.val));
				cv::putText(img, st.str(), origin+offset_text, cv::FONT_HERSHEY_SIMPLEX, font_size, *color, 2);

		


				break;
			case OVERLAY_SCATTER_OPENCV:
				// marker
				ov = *it;
				sc = (scatter_opencv*)(ov);
				//write_label = sc->label;
				if (0 != (sc->label).size())
				{
					if (NULL != sc->marker_draw)
					{
						tmp = origin + (offset_line[0] + offset_line[1]) / 2;
						sc->marker_draw(img, tmp, sc->marker_size, sc->color);
					}

					// line
					if (0 != sc->line_type)
					{
						cv::line(img, origin + offset_line[0], origin + offset_line[1], *(sc->color), 2, 8);
					}
					// label


					cv::putText(img, sc->label, origin + offset_text, cv::FONT_HERSHEY_SIMPLEX, font_size, *color, 2);
				}
	
				break;
			case OVERLAY_BARPLOT_OPENCV:
				ov = *it;
				bp = (barplot_opencv*)(ov);
				if (0 != (bp->label).size())
				{
					cv::rectangle(img, origin, origin + offset_half, *(bp->color), -1);

					cv::putText(img, bp->label, origin + offset_text, cv::FONT_HERSHEY_SIMPLEX, font_size, *color, 2);
				}
				break;
			default:
				break;
			}
		origin.y += step.y;
	}



}

void legend_opencv::edit(void *in)
{
}

void legend_opencv::getvalue(return_get_value_overlay_opencv *out)
{
	out->type = OOV_TYPE_STRING;
	out->val = &label_return;
}




barplot_opencv::barplot_opencv()
{
	name = "barplot";
	type = OVERLAY_BARPLOT_OPENCV;
	color_normal = cv::Scalar(255, 255, 255);
	color_selection = cv::Scalar(255, 0, 255);
	dimpoint = 0;
	x = NULL;
	y = NULL;
	errx = NULL;
	erry = NULL;
	this->init();
}

void  barplot_opencv::init()
{
	x = NULL;
	y = NULL;
	errx = NULL;
	erry = NULL;
	dimpoint = 0;
	selected = false;
	color = &color_normal;

	set_bar_type(BAR_SOLID_OPENCV);

}

const  barplot_opencv&  barplot_opencv::operator = (const  barplot_opencv& s)
{
	this->dimpoint = s.dimpoint;
	this->x = s.x;
	this->y = s.y;
	this->errx = s.errx;
	this->erry = s.erry;

	this->name = s.name;
	this->type = s.type;

	this->set_bar_type(s.bar_type);
	return *this;
}

void  barplot_opencv::draw(cv::Mat &img, double start[], double scale[])
{
	cv::Point p[3],*pbefore,*pp,*pafter, *ptmp;
	unsigned int i,j;

	pbefore = p;
	pp = &(p[1]);
	pafter = &(p[2]);


	if ((BAR_NONE_OPENCV != bar_type) && (2<dimpoint))
	{
		
		pp[0].x = (int)((x[0] - start[0]) / scale[0]);
		pp[0].y = (int)((y[0] - start[1]) / scale[1]);
		pafter[0].x = (int)((x[1] - start[0]) / scale[0]);
		pafter[0].y = (int)((0.0 - start[1]) / scale[1]);

		pbefore[0].x = pp[0].x - (pafter[0].x - pp[0].x);
		bar_draw(img, pbefore, pp, pafter, color);
		pafter[0].y = (int)((y[1] - start[1]) / scale[1]);
		for (i = 2; i < dimpoint; ++i)
		{
			// exchange pointers round
			ptmp = pbefore;		pbefore = pp;		pp = pafter;		pafter = ptmp;
			pafter[0].x = (int)((x[i] - start[0]) / scale[0]);
			pafter[0].y = (int)((0.0 - start[1]) / scale[1]);
			bar_draw(img, pbefore, pp, pafter, color);
			pafter[0].y = (int)((y[i] - start[1]) / scale[1]);
		}

		ptmp = pbefore;		pbefore = pp;		pp = pafter;		pafter = ptmp;
		pafter[0].x = pp[0].x + (pp[0].x-pbefore[0].x);
		pafter[0].y = (int)((0.0 - start[1]) / scale[1]);
		bar_draw(img, pbefore, pp, pafter, color);
	}

}

void  barplot_opencv::set_bar_type(int _type)
{
	switch (_type)
	{
	case BAR_NONE_OPENCV:
		bar_draw = NULL;
		break;
	case	BAR_SOLID_OPENCV:
		bar_draw = draw_solid_bar;
		break;
	case		BAR_EDGE_OPENCV:
		bar_draw = draw_edge_bar;
		break;
	case		BAR_SOLIDEDGE_OPENCV:
		bar_draw = draw_solidedge_bar;
		break;
	case	BAR_TOPLINE_OPENCV:
		bar_draw = draw_topline_bar;
		break;

	default:
		return;
		break;
	}
	bar_type = _type;
}


void barplot_opencv::edit(void *in)
{
}

void barplot_opencv::getvalue(return_get_value_overlay_opencv *out)
{
	out->type = OOV_TYPE_STRING;
	out->val = &label;
}


text_opencv::text_opencv()
{
	name = "text";

	color_normal = cv::Scalar(255, 255, 255);
	color_selection = cv::Scalar(255, 0, 255);
	
	this->init();
}

void  text_opencv::init()
{
	thickness = 2;
	font_scale = 0.6;
	this->select();
}

const  text_opencv&  text_opencv::operator = (const  text_opencv& tx)
{
	this->name = tx.name;
	this->type = tx.type;
	this->color = tx.color;
	this->selected = tx.selected;
	this->color_normal = tx.color_normal;
	this->color_selection = tx.color_selection;
	this->text = tx.text;
	return *this;
}

void  text_opencv::edit(void *in)
{
	text = *((std::string*)(in));
}

void  text_opencv::getvalue(return_get_value_overlay_opencv *out)
{
	out->type = OOV_TYPE_STRING;
	out->val = &text;
}

text_relative_opencv::text_relative_opencv() :text_opencv()
{
	type = OVERLAY_TEXT_RELATIVE_OPENCV;
}

void text_relative_opencv::draw(cv::Mat &img, double start[], double scale[])
{
	cv::Point p;

	p.x = (int)(x[0] * (double)(img.cols));
	p.y = (int)(x[1] * (double)(img.rows));

	cv::putText(img, text, p, cv::FONT_HERSHEY_SIMPLEX, font_scale, *color, thickness);
}

text_absolute_opencv::text_absolute_opencv() : text_opencv()
{
	type = OVERLAY_TEXT_ABSOLUTE_OPENCV;
}

void text_absolute_opencv::draw(cv::Mat &img, double start[], double scale[])
{
	cv::Point p;

	p.x = (int)((x[0] - start[0]) / scale[0]);
	p.y = (int)((x[1] - start[1]) / scale[1]);

	cv::putText(img, text, p, cv::FONT_HERSHEY_SIMPLEX, font_scale, *color, thickness);
}

