

/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 12/2015
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
#ifndef _CONTROL_OPENCV_H_
#define _CONTROL_OPENCV_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <list>

/*!
This is the mother class for buttons, trackbars... and what can
be used for graphcal user interface.
This controls are thought to be displayed on an image and the programmers should
implement the appearence and the behaviour by using the appropriate virtual functions (see comments below).
*/

class control_opencv
{
public:
	control_opencv(int _id = -1, std::string _name = "");
	control_opencv(const control_opencv& ctrl);
	~control_opencv();

	virtual void select() = 0;
	virtual void deselect() = 0;
	virtual void enable() = 0;
	virtual void disable() = 0;

	virtual void draw(cv::Mat &img) = 0;
	/*!
	Multiple actions are possible and multiple grabbing point for the pointer are possible as well
	This method should return any negative number in case no action should take place.
	*/
	virtual int handle(float pos[]) = 0;
	virtual control_opencv& operator=(const control_opencv &ctrl) = 0;
	//!retrive the value of the control
	virtual void get_value(void *out) = 0;
	//!Implementation for setting the value of the control
	virtual void set_value(void *in) = 0;

	/*!
	This method is intended to be called in case control_opencv::handle would return a value >= 0
	and arranges the behaviour of the control when the mouse button of choice is pressed the first time
	*/
	virtual void onclick(float pos[]) = 0;
	/*!
	This method is intended to be called in case control_opencv::onclick was already called.
	It defines the behviour in case of mouse move with button pressed.
	*/
	virtual void onmove(float pos[]) = 0;
	/*!
	This method is intended to be called in case control_opencv::onclick was already called.
	It arranges the behaviour of the control when the mouse button of choice is released
	*/
	virtual void onrelease(float pos[]) = 0;


	bool selected;
	bool enabled;
	//! assign the value you prefer to this id, it might be usefull for identification
	int id;
	//! this variable can be used in combination with enum control_types to know the specific implementation
	int type;
	//! the string name can be usefull in opencv enviroment as well as to draw some string
	std::string name;
	/*! variable to store the geometrical position of the point on the window
	The value should be intended as relative values for a window size of 1x1
	This allows for easy resize of the window that would at the same time resize
	the entire aspect of the control
	*/
	float *x;

};

/*! It might be useful to quikly know which type of control is from time to time
in order not to fail the pointer cast. This enum is inteded to facilitate this process
and the specified number should be assigned in any control constructor to the variable
any_control_opencv::type
*/
enum control_types
{
	CTRL_BUTTON = 0,
	CTRL_SINGLE_TRACKBAR = 1,
	CTRL_MINMAX_TRACKBAR = 2,
	CTRL_TEXTBOX = 3
};

/*!This function toggles the selection option of a control_opencv inside the vector
so that only one control_opencv is selected and all others are deselected.*/
control_opencv* select_control(control_opencv*, std::vector<control_opencv*> lst);

#endif