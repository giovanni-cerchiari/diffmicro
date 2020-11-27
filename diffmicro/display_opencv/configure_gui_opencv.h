

/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 2016
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

#ifndef _CONFIGURE_GUI_OPENCV_H_
#define _CONFIGURE_GUI_OPENCV_H_

#include "dir.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <list>
#include <string>
#include <sstream>
#include "global_define.h"
#include "prep_string.h"
#include "controls_opencv.h"

/*!
This enumerator is used to assign an integer number to a number type so that it will be possible
to recongnize the number type at runtime inside the execution.
*/
enum ui_variable_type_enum
{
	UI_TYPE_INT8_OPENCV = 0,
	UI_TYPE_UINT8_OPENCV = 1,
	UI_TYPE_INT16_OPENCV = 2,
	UI_TYPE_UINT16_OPENCV = 3,
	UI_TYPE_INT32_OPENCV = 4,
	UI_TYPE_UINT32_OPENCV = 5,
	UI_TYPE_INT64_OPENCV = 6,
	UI_TYPE_UINT64_OPENCV = 7,
	UI_TYPE_INT128_OPENCV = 8,
	UI_TYPE_UINT128_OPENCV = 9,
	UI_TYPE_FLOAT32_OPENCV = 10,
	UI_TYPE_FLOAT64_OPENCV = 11,//double
	UI_TYPE_FLOAT128_OPENCV = 12,//long double
	UI_TYPE_BOOL_OPENCV = 20,
	UI_TYPE_STRING_OPENCV = 30
};

/*!
This class implements a user interface variable
*/
class ui_variable
{
public:
	ui_variable(){}
	~ui_variable(){ this->clear(); }

	void clear();

	std::vector<control_opencv*> control;

	void draw(cv::Mat &mat);

	void enable();
	void disable();

	void get(std::string &value);
	void get(void *value);
	bool set(std::string &value);
	void set(void *value);

	void refresh();

	int id;
	int type;
	void *ptr;

protected:
	virtual bool check() = 0;
	bool set_box();
};




bool ui_variable_check_default(std::vector<control_opencv*> &v);
bool ui_variable_check_file(std::vector<control_opencv*> &v);
bool ui_variable_check_folder(std::vector<control_opencv*> &v);
bool ui_variable_check_numeric(std::vector<control_opencv*> &v);
bool ui_variable_check_positive(std::vector<control_opencv*> &v);
bool ui_variable_check_integer(std::vector<control_opencv*> &v);
bool ui_variable_check_unsigned_integer(std::vector<control_opencv*> &v);


/*!
This class is written for diffmicro application.
This class stores all the user preferences about the execution of the program.
*/
class configure_gui_opencv
{
public:
	/*!constructor*/
	configure_gui_opencv();
	/*!destructor*/
	~configure_gui_opencv();

	bool flg_start;



	void show();

	void resize_window(unsigned int display_dimx, unsigned int display_dimy);
	void move_window(float x, float y);
	void move_window(int x, int y);

	static void callback_mouse(int ev, int x, int y, int flags, void* userdata);

	//! active object of the user interface
	ui_variable* active_variable;

	ui_variable* get_variable(int id);

	/*!
	vector of the files to be processed
	*/
	std::vector< std::string> file_list;
	/*!flg to know if an image is valid or not*/
	std::vector<bool> flg_valid_image;
	/*!time stamp for each image*/
	std::vector<MY_REAL> time;
	/*!*/
	std::list<ui_variable*> option;

	/*!
	class initialization from file. The input entry must be store in diffrenet The file is made with a series of tags in char* format that sould be satisfaid.
	After the tag a tab ("\t") character should be left to separate the entry value. Tags can be placed in any order
	*/
	bool load(std::string file_ui);

	bool save(std::string filename = "");

	static int keyboard_refresh(int key);
	bool all_setted();

	void variables_to_gui();
	void gui_to_variables();

	void write_time_sequence_to_file();

	void close_window();

protected:

	std::string name;

	cv::Mat panel;

	// mouse position for control manipulation (x_current, y_current, x_previous, y_previous, x_onclick, y_onclick)
	float xx[8];
	float *xx_current;
	float *xx_previous;
	float *xx_onclick;
	float *xx_onrelease;

private:

	/*!
	This functions returns alla the files contained in path with extension ".dat" sorted in respect to the last number present in the file name
	*/
	bool prep_file_list(std::string path, std::vector<std::string > &list_file);

	/*! method used by all_setted*/
	int check_path_set_box(int check_box_id, int variable_id);
	int check_path_button_set_box(int check_box_id, int button_id, int variable_id);
	int check_positive_integer_set_box(int check_box_id, int string_id);
	int check_positive_integer_button_set_box(int check_box_id, int button_id, int path_id);
	int check_filename_set_box(int check_box_id, int path_id, int filename_id);

	void add_input_line(int n_cols, int n_rows, int line, int check_box_id, int label_id, std::string label, int variable_id, int variable_type, std::string defaultvalue = "");
	void add_button_line(int n_cols, int n_rows, int line, int check_box_id, int button_id, std::string label_true, std::string label_false, int variable_id, int variable_type, std::string defaultvalue = "");

	bool adjust_time_spacing(std::vector<std::string> &file_list, std::vector<MY_REAL> time_input, MY_REAL time_step, std::vector<bool> &flg_valid_image);

	void initialize();
	void translate();

};

#endif
