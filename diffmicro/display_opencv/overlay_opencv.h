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

#ifndef _OVERLAY_OPENCV_H_
#define _OVERLAY_OPENCV_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <list>

#include "marker_draw_opencv.h"
#include "my_math.h"


struct return_get_value_overlay_opencv
{
	int type;
	void *val;
};

enum type_return_get_value_overlay_opencv
{
	OOV_TYPE_DOUBLE = 0,
	OOV_TYPE_STRING = 1
};

/*!
Overlays are thought to be things drawn on top of the window_display_opencv
They are not intented to modify the image but just lay on top of it.
They can be anthing implemented through the virtual functions.
For further manipulation the possibility to select and deselect the overaly
is found to be important.

WARNING: the position for drawing inside may be intended in absolute frame.
The absolute frame is define by the property of image in window_display_opencv
*/

enum overlay_type_opencv
{
	OVERLAY_UNKNOWN_OPENCV = 0,
	OVERLAY_CIRLE_OPENCV = 1,
	OVERLAY_RECTANGLE_OPENCV = 2,
	OVERLAY_CONTOUR_OPENCV = 3,
	OVERLAY_SCATTER_OPENCV =4,
	OVERLAY_AXES_OPENCV = 5,
	OVERLAY_LEGEND_OPENCV = 6,
	OVERLAY_BARPLOT_OPENCV = 7,
	OVERLAY_TEXT_RELATIVE_OPENCV = 8,
	OVERLAY_TEXT_ABSOLUTE_OPENCV = 9
	//OVERLAY_CIRCLE_FASTAVG_OPENCV = 10
};

class overlay_opencv
{
public:
	overlay_opencv(std::string _name = ""){ name = _name; selected = false; }
	overlay_opencv(const overlay_opencv& ov){ *this = ov; }
	virtual ~overlay_opencv(){}

	const overlay_opencv& operator=(const overlay_opencv& ov) { this->name = ov.name; this->id = ov.id; return *this; }

	//! Is this overlay selected?
	bool selected;
	int id;
	int type;
	std::string name;
	//! Apperance color
	cv::Scalar *color;
	//! Apperance color in case selected == false
	cv::Scalar color_normal;
	//! Apperance color in case selected == true
	cv::Scalar color_selection;
	virtual void draw(cv::Mat &img, double start[], double scale[]) = 0;
	virtual void edit(void* in) = 0;
	virtual void getvalue(return_get_value_overlay_opencv *out) = 0;

	void select();
	void deselect();

};

overlay_opencv* select_overlay(int i_to_select, std::list<overlay_opencv*> lst);

enum geometric_shape_draw_type
{
	GEOMETRIC_SHAPE_DRAW_TYPE_SIMPLE = 0,
	GEOMETRIC_SHAPE_DRAW_TYPE_VERBOSE = 1
};

struct geometric_shape_edit_parameters
{
	double *x;
	int draw_type;
};

class circle_opencv : public overlay_opencv
{
public:
	circle_opencv();
	circle_opencv(const circle_opencv& cr) { *this = cr; }
	~circle_opencv(){}

	const circle_opencv& operator=(const circle_opencv& cr);
	void init();

	void draw(cv::Mat &img, double start[], double scale[]);
	void drawtext(int posx, int posy, cv::Mat &img, unsigned int dimx, unsigned int dimy, double scalex, double scaley, double *mat, unsigned int &n_pixel, double &avg, double &stddev);
	void edit(void *in);
	void getvalue(return_get_value_overlay_opencv *out);

	void average(unsigned int dimx, unsigned int dimy, double scalex, double scaley, double *mat, unsigned int &n_pixel, double &avg, double &stddev);

	bool find_interval(unsigned int dim, double pos, double scale, unsigned int &start, unsigned int &end);

	double r;
	/*!
	for drawing purpose of a radius two points are needed
	(x0, y0, x1, y1)
	x0 and y0 are the middle point of the circle
	*/
	double x[4];
	int draw_type;
};

class rectangle_opencv : public overlay_opencv
{
public:
	rectangle_opencv();
	rectangle_opencv(const rectangle_opencv& cr) { *this = cr; }
	~rectangle_opencv(){}

	const rectangle_opencv& operator=(const rectangle_opencv& cr);
	void init();

	void draw(cv::Mat &img, double start[], double scale[]);
	void edit(void *in);
	void getvalue(return_get_value_overlay_opencv *out);

	/*!
	for drawing purpose of the rectangle plus a diagonal two points are needed
	(x0, y0, x1, y1)
	*/
	double x[4];
	int draw_type;
	
protected:
	double returngetvalue;

};

struct contour_opencv_edit_parameters
{
	unsigned int dimx;
	unsigned int dimy;
	double *mat;
	double *scale;
	double val;
};

class contour_opencv : public overlay_opencv
{
public:
	contour_opencv();
	contour_opencv(const contour_opencv& cr) { *this = cr; }
	~contour_opencv();

	const contour_opencv& operator=(const contour_opencv& cr);
	void init();

	void draw(cv::Mat &img, double start[], double scale[]);
	void edit(void *in);
	void getvalue(return_get_value_overlay_opencv *out);

	void set_marker_type(int _type);
	int get_marker_type() { return marker_type; }
	void(*marker_draw)(cv::Mat &img, cv::Point &x, int size, cv::Scalar *color);

	void push_back(double *xx);
	void pop_back();

	unsigned int dimpoint;
	double *x;
	double val;

	int marker_size;

protected:
	int marker_type;
	unsigned int capacity;

};



enum line_type_opencv
{
	LINE_NONE_OPENCV = 0,
	LINE_NORMAL_OPENCV = 1,
};

/*!This overlay implements a scatter plot on the window_display_opencv. Plots are made as overalay to facilitate plotting over images.*/
class scatter_opencv : public overlay_opencv
{
public:
	scatter_opencv();
	scatter_opencv(const scatter_opencv& sc) { *this = sc; }
	~scatter_opencv(){}

	const scatter_opencv& operator=(const scatter_opencv& cr);
	void init();

	void draw(cv::Mat &img, double start[], double scale[]);
	void edit(void *in);
	void getvalue(return_get_value_overlay_opencv *out);
	 
	void set_marker_type(int _type);
	int get_marker_type() { return marker_type; }
	void(*marker_draw)(cv::Mat &img, cv::Point &x, int size, cv::Scalar *color);

	unsigned int dimpoint;
	double *x;
	double *y;
	double *errx;
	double *erry;

	int marker_size;
	int line_thickness;
	int line_type_internal_opencv;

	int line_type;

	std::string label;

protected:
	int marker_type;

};


/*!This overlay implements the 2-D axes on the window_display_opencv. It is used with scatter_opencv and barplot_opencv.
Plots are implemented as overalay to facilitate plotting over images.*/
class axes_opencv : public overlay_opencv
{
public:
	axes_opencv();
	axes_opencv(const axes_opencv& ax) { *this = ax; }
	~axes_opencv(){}

	const axes_opencv& operator=(const axes_opencv& ax);
	void init();

	void draw(cv::Mat &img, double start[], double scale[]);
	void edit(void *in);
	void getvalue(return_get_value_overlay_opencv *out);

	void num_axis(double origin, double end, unsigned int n_wished_tick, unsigned int &ntick, double &start, double &step);

	unsigned int dim_tick[2];
	
	int font_number_size;
	
	double font_label_size;
	std::string xlabel;
	std::string ylabel;

	std::string label_return;




};

class legend_opencv : public overlay_opencv
{
public:
	legend_opencv();
	legend_opencv(const legend_opencv& ax) { *this = ax; }
	~legend_opencv(){}

	const legend_opencv& operator=(const legend_opencv& ax);
	void init();

	void draw(cv::Mat &img, double start[], double scale[]);
	void edit(void *in);
	void getvalue(return_get_value_overlay_opencv *out);

	double font_size;
	std::list<overlay_opencv*> *list_overlay;
	std::string label_return;
};


/*!This overlay implements a bar plot on the window_display_opencv. Plots are made as overalay to facilitate plotting over images.*/
class barplot_opencv : public overlay_opencv
{
public:
	barplot_opencv();
	barplot_opencv(const barplot_opencv& ax) { *this = ax; }
	~barplot_opencv(){}

	const  barplot_opencv& operator=(const  barplot_opencv& ax);
	void init();

	void draw(cv::Mat &img, double start[], double scale[]);
	void edit(void *in);
	void getvalue(return_get_value_overlay_opencv *out);

	void set_bar_type(int _type);
	int get_bar_type() { return bar_type; }
	void(*bar_draw)(cv::Mat &img, cv::Point *xbefore, cv::Point *x, cv::Point *xafter, cv::Scalar *color);

	unsigned int dimpoint;
	double *x;
	double *y;
	double *errx;
	double *erry;

	

	std::string label;

protected:
	int bar_type;
};

/*!This overlay print a text on the window_display_opencv*/
class text_opencv : public overlay_opencv
{
public:
	text_opencv();
	text_opencv(const text_opencv& ax) { *this = ax; }
	~text_opencv(){}

	const  text_opencv& operator=(const  text_opencv& tx);
	void init();

	virtual void draw(cv::Mat &img, double start[], double scale[]) = 0;
	void edit(void *in);
	void getvalue(return_get_value_overlay_opencv *out);

	double x[2];
	std::string text;
	double font_scale;
	int thickness;

};

/*!This overlay print a text in relative coordinates on the window_display_opencv*/
class text_relative_opencv : public text_opencv
{
public:
	text_relative_opencv();
	text_relative_opencv(const text_relative_opencv& ax) { *this = ax; }
	~text_relative_opencv(){}

	void draw(cv::Mat &img, double start[], double scale[]);

};

/*!This overlay print a text in absolute coordinates on the window_display_opencv*/
class text_absolute_opencv : public text_opencv
{
public:
	text_absolute_opencv();
	text_absolute_opencv(const text_absolute_opencv& ax) { *this = ax; }
	~text_absolute_opencv(){}

	void draw(cv::Mat &img, double start[], double scale[]);

};



#endif

