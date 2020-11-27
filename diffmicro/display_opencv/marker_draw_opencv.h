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

#ifndef _MARKER_DRAW_OPENCV_H_
#define _MARKER_DRAW_OPENCV_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

/*!
These functions are used to draw particular shapes on images.
They are intended to be used by overlays
*/

enum marker_type_opencv
{
	MARKER_NONE_OPENCV = 0,
	MARKER_CIRCLE_OPENCV = 1,
	MARKER_RECTANGLE_OPENCV = 2,
	MARKER_CROSS_OPENCV = 3,
	MARKER_TRIANGLE_OPENCV = 4,
	MARKER_DIAMOND_OPENCV = 5
};


void draw_circle(cv::Mat &img, cv::Point &x, int size, cv::Scalar *color);
void draw_rectangle(cv::Mat &img, cv::Point &x, int size, cv::Scalar *color);
void draw_cross(cv::Mat &img, cv::Point &x, int size, cv::Scalar *color);
void draw_triangle(cv::Mat &img, cv::Point &x, int size, cv::Scalar *color);
void draw_diamond(cv::Mat &img, cv::Point &x, int size, cv::Scalar *color);

enum bar_type_opencv
{
	BAR_NONE_OPENCV = 0,
	BAR_SOLID_OPENCV = 1,
	BAR_EDGE_OPENCV = 2,
	BAR_SOLIDEDGE_OPENCV = 3,
	BAR_TOPLINE_OPENCV = 4
};

void draw_solid_bar(cv::Mat &img, cv::Point *xbefore, cv::Point *x, cv::Point *xafter, cv::Scalar *color);
void draw_edge_bar(cv::Mat &img, cv::Point *xbefore, cv::Point *x, cv::Point *xafter, cv::Scalar *color);
void draw_solidedge_bar(cv::Mat &img, cv::Point *xbefore, cv::Point *x, cv::Point *xafter, cv::Scalar *color);
void draw_topline_bar(cv::Mat &img, cv::Point *xbefore, cv::Point *x, cv::Point *xafter, cv::Scalar *color);

#endif