/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date:02/ 2016

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

#include "marker_draw_opencv.h"


void draw_circle(cv::Mat &img, cv::Point &x, int size, cv::Scalar *color)
{
	cv::circle(img, x, size, *color,2,8);
}

void draw_rectangle(cv::Mat &img, cv::Point &x, int size, cv::Scalar *color)
{
	cv::Point p[2];
	p[0].x = x.x + size; p[0].y = x.y + size;
	p[1].x = x.x - size; p[1].y = x.y - size;
	cv::rectangle(img, p[0], p[1], *color, 2, 8);
}

void draw_cross(cv::Mat &img, cv::Point &x, int size, cv::Scalar *color)
{
	cv::Point p[2];
	p[0].x = size + x.x; p[0].y = size + x.y;
	p[1].x = x.x - size; p[1].y = x.y- size;
	cv::line(img, p[0], p[1], *color, 2, 8);
	p[1].y += 2 * size; p[0].y -= 2 * size;
	cv::line(img, p[0], p[1], *color, 2, 8);
}

void draw_triangle(cv::Mat &img, cv::Point &x, int size, cv::Scalar *color)
{
	cv::Point p[2];
	int shift;

	shift = (int)((float)(0.866025)*(float)(size));
	p[0].x = x.x; p[0].y = x.y-size;
	p[1].x = x.x + shift; p[1].y = x.y + size/2;
	cv::line(img, p[0], p[1], *color, 2, 8);
	p[1].x = x.x - shift;
	cv::line(img, p[0], p[1], *color, 2, 8);
	p[0].x = x.x + shift; p[0].y = p[1].y;
	cv::line(img, p[0], p[1], *color, 2, 8);

}

void draw_diamond(cv::Mat &img, cv::Point &x, int size, cv::Scalar *color)
{
	cv::Point p[2];
	p[0].x = x.x + size; p[0].y = x.y;
	p[1].x = x.x       ; p[1].y = x.y + size;
	cv::line(img, p[0], p[1], *color, 2, 8);
	p[0].x -= 2*size;
	cv::line(img, p[0], p[1], *color, 2, 8);
	p[1].y -= 2 * size;
	cv::line(img, p[0], p[1], *color, 2, 8);
	p[0].x += 2 * size;
	cv::line(img, p[0], p[1], *color, 2, 8);
}



void draw_solid_bar(cv::Mat &img, cv::Point *xbefore, cv::Point *x, cv::Point *xafter, cv::Scalar *color)
{
	cv::Point p[2];
	cv::Scalar black(0, 0, 0);

	p[0].x = (x[0].x + xbefore[0].x) / 2;
	p[0].y = x[0].y;
	p[1].x = (x[0].x + xafter[0].x) / 2;
	p[1].y = xafter[0].y;
	cv::rectangle(img, p[0], p[1], *color, -1);

}
void draw_edge_bar(cv::Mat &img, cv::Point *xbefore, cv::Point *x, cv::Point *xafter, cv::Scalar *color)
{
	cv::Point p[2];

	p[0].x = (x[0].x + xbefore[0].x) / 2;
	p[0].y = x[0].y;
	p[1].x = (x[0].x + xafter[0].x) / 2;
	p[1].y = xafter[0].y;
	cv::rectangle(img, p[0], p[1], *color, 2);
}

void draw_solidedge_bar(cv::Mat &img, cv::Point *xbefore, cv::Point *x, cv::Point *xafter, cv::Scalar *color)
{
	cv::Point p[2];
	cv::Scalar black(0, 0, 0);

	p[0].x = (x[0].x + xbefore[0].x) / 2;
	p[0].y = x[0].y;
	p[1].x = (x[0].x + xafter[0].x) / 2;
	p[1].y = xafter[0].y;
	cv::rectangle(img, p[0], p[1], *color, -1);
	cv::rectangle(img, p[0], p[1], black, 0);

}

void draw_topline_bar(cv::Mat &img, cv::Point *xbefore, cv::Point *x, cv::Point *xafter, cv::Scalar *color)
{
	cv::Point p[2];

	p[0].x = (x[0].x + xbefore[0].x) / 2;
	p[0].y = x[0].y;
	p[1].x = (x[0].x + xafter[0].x) / 2;
	p[1].y = x[0].y;
	cv::line(img, p[0], p[1], *color, -1);

}
