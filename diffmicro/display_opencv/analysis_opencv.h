

/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 12/2015

implemented with opencv v 3.0
*/

/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com

update:05/2020 - 09/2020
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


/*! The function contained in this file are used to make quick evaluation on diplayed object inside a window_display_opencv*/
#ifndef _ANALYSIS_OPENCV_H_
#define _ANALYSIS_OPENCV_H_

#include <string>
#include <iostream>


enum analysis_opencv_type
{
	INTEGRATOR_CIRCLE_ANALYSIS_OPENCV = 0,
	INTEGRATOR_RECTANGLE_ANALYSIS_OPENCV = 1,
	AVERAGER_CIRCLE_ANALYSIS_OPENCV = 2,
	AVERAGER_RECTANGLE_ANALYSIS_OPENCV = 3,
	ANGULAR_AVERAGER_ANALYSIS_OPENCV = 4
};


/*! Analysis opencv should be an operation applied on the data contained into a window_display_opencv */

class analysis_opencv
{
public:
	analysis_opencv();
	~analysis_opencv() { this->clear(); }

	virtual void run() = 0;
	virtual void get_value(void *val) = 0;

	int type;
	int id;
	std::string name;

	void clear();
	void alloc(unsigned int _dimx, unsigned int _dimvalue, unsigned int _dimlut);

protected:

	double dimx;
	double *x;
	
	unsigned int dimvalue;
	double *value;

	unsigned int dimlut;
	unsigned int *lut;
};




#endif
