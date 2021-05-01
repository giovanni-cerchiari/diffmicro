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
#include "keyboard_opencv.h"

void diffmicro_user_interface::add_input_line(int n_cols, int n_rows, int line, int check_box_id, int label_id, std::string label, int variable_id, int variable_type, std::string defaultvalue)
{
	bool flg;
	ui_variable *uivar;
	button_opencv *button;
	textbox_opencv *textbox;

	uivar = new ui_variable(UI_TYPE_BOOL_DIFFMICRO);
	button = new button_opencv(check_box_id, label + "_checkbox");
	button->label_false = "NO";
	button->label_true = "OK";
	button->disable();
	button->x[0] = 0.1 / (double)(n_cols);
	button->x[1] = (double)(line) / (double)(n_rows);
	button->x[2] = button->x[0] + 0.7 / (double)(n_cols);
	button->x[3] = button->x[1] + 0.9 / (double)(n_rows);
	flg = false;	button->set_value(&flg);
	uivar->control = (control_opencv*)(button);
	option.push_back(uivar);

	uivar = new ui_variable(UI_TYPE_STRING_DIFFMICRO);
	textbox = new textbox_opencv(label_id, label);
	textbox->text = textbox->name;
	textbox->x[0] = 1. / (double)(n_cols);
	textbox->x[1] = (double)(line) / (double)(n_rows);
	textbox->x[2] = textbox->x[0] + 3. / (double)(n_cols);
	textbox->x[3] = textbox->x[1] + 0.9 / (double)(n_rows);
	textbox->disable();
	uivar->control = (control_opencv*)(textbox);
	option.push_back(uivar);

	uivar = new ui_variable(variable_type);
	textbox = new textbox_opencv(variable_id, label + "_var");
	textbox->text = textbox->name;
	textbox->color_label_normal = cv::Scalar(0, 255, 255);
	textbox->x[0] = 4. / (double)(n_cols);
	textbox->x[1] = (double)(line) / (double)(n_rows);
	textbox->x[2] = textbox->x[0] + 10. / (double)(n_cols);
	textbox->x[3] = textbox->x[1] + 0.9 / (double)(n_rows);
	textbox->enable();
	textbox->set_value(&defaultvalue);
	uivar->control = (control_opencv*)(textbox);
	option.push_back(uivar);

}

void diffmicro_user_interface::add_button_line(int n_cols, int n_rows, int line, int check_box_id, int button_id, std::string label_true, std::string label_false, int variable_id, int variable_type, std::string defaultvalue)
{
	bool flg;
	ui_variable *uivar;
	button_opencv *button;
	textbox_opencv *textbox;

	uivar = new ui_variable(UI_TYPE_BOOL_DIFFMICRO);
	button = new button_opencv(check_box_id, label_true + "_checkbox");
	button->label_false = "NO";
	button->label_true = "OK";
	button->disable();
	button->x[0] = 0.1 / (double)(n_cols);
	button->x[1] = (double)(line) / (double)(n_rows);
	button->x[2] = button->x[0] + 0.7 / (double)(n_cols);
	button->x[3] = button->x[1] + 0.9 / (double)(n_rows);
	flg = false;	button->set_value(&flg);
	uivar->control = (control_opencv*)(button);
	option.push_back(uivar);

	uivar = new ui_variable(UI_TYPE_BOOL_DIFFMICRO);
	button = new button_opencv(button_id, label_true);
	button->x[0] = 1. / (double)(n_cols);
	button->x[1] = (double)(line) / (double)(n_rows);
	button->x[2] = button->x[0] + 3. / (double)(n_cols);
	button->x[3] = button->x[1] + 0.9 / (double)(n_rows);
	flg = false;	button->set_value(&flg);
	button->label_true = label_true;
	button->label_false = label_false;
	button->enable();
	uivar->control = (control_opencv*)(button);
	option.push_back(uivar);

	uivar = new ui_variable(variable_type);
	textbox = new textbox_opencv(variable_id, label_true + "_var");
	textbox->text = textbox->name;
	textbox->color_label_normal = cv::Scalar(0, 255, 255);
	textbox->x[0] = 4. / (double)(n_cols);
	textbox->x[1] = (double)(line) / (double)(n_rows);
	textbox->x[2] = textbox->x[0] + 10. / (double)(n_cols);
	textbox->x[3] = textbox->x[1] + 0.9 / (double)(n_rows);
	textbox->enable();
	textbox->set_value(&defaultvalue);
	uivar->control = (control_opencv*)(textbox);
	option.push_back(uivar);
}


void diffmicro_user_interface::initialize()
{
	refresh = diffmicro_user_interface::keyboard_refresh;
	//------------------------------------------------------------
	/*!graphical initialization*/
	int i;
	int dimx_panel;
	int dimy_panel;
	int n_rows = 20;
	int n_cols = 15;
	int line = 1;
	bool flg;
	unsigned char* panel_data_i_uc;

	name = "optionsw";

	xx_current = xx;
	xx_previous = &(xx[2]);
	xx_onclick = &(xx[4]);
	xx_onrelease = &(xx[6]);

	cv::namedWindow(name, CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

	GetDesktopResolution(dimx_panel, dimy_panel);
	dimx_panel = (dimx_panel*9) / 10;
	dimy_panel = (dimy_panel*8) / 10;

	panel.create(dimy_panel, dimx_panel, CV_8U);
	for (i = 0; i < panel.rows * panel.cols; ++i)
	{
		panel_data_i_uc = (unsigned char*)(&(panel.data[i]));
		panel_data_i_uc[0] = 0;
	}

	cv::cvtColor(panel, panel, CV_GRAY2RGB);

	cv::setMouseCallback(name, diffmicro_user_interface::callback_mouse, NULL);



	ui_variable *uivar;
	button_opencv *button;
	textbox_opencv *textbox;

	uivar = new ui_variable(UI_TYPE_BOOL_DIFFMICRO);
	button = new button_opencv(FLG_START,"start");
	button->x[0] = 2./3.;
	button->x[1] = (double)(n_rows-1)/(double)(n_rows);
	button->x[2] = 2. / 3. + 3. / (double)(n_cols);
	button->x[3] = button->x[1] + 0.9 / (double)(n_rows);
	flg = false;	button->set_value(&flg);
	uivar->control = (control_opencv*)(button);
	option.push_back(uivar);

	add_input_line(n_cols, n_rows, line, CHECKBOX_PATH_IMAGES, 10000 + PATH_IMAGES, "input directory:", PATH_IMAGES, UI_TYPE_STRING_DIFFMICRO);
	++line;
	add_input_line(n_cols, n_rows, line, CHECKBOX_FIRST_IMAGE, 10000 + FIRST_IMAGE, "first image:", FIRST_IMAGE, UI_TYPE_INDEX_DIFFMICRO);
	++line;
	add_input_line(n_cols, n_rows, line, CHECKBOX_LAST_IMAGE, 10000 + LAST_IMAGE, "last image:", LAST_IMAGE, UI_TYPE_INDEX_DIFFMICRO);
	++line;
	/*add_button_line(n_cols, n_rows, line, CHECKBOX_TIME_FILENAME, FLG_TIME_FILENAME, "time file:", "equispaced images", TIME_FILENAME, UI_TYPE_STRING_DIFFMICRO);
	++line;*/
	/*add_input_line(n_cols, n_rows, line, CHECKBOX_TIME_STEP, 10000 + TIME_STEP, "time step:", TIME_STEP, UI_TYPE_INDEX_DIFFMICRO);
	++line;*/
	add_button_line(n_cols, n_rows, line, CHECKBOX_VALID_IMAGE_FILENAME, FLG_VALID_IMAGE_FILENAME, "file valid images:", "all valid images",VALID_IMAGE_FILENAME, UI_TYPE_STRING_DIFFMICRO);
	++line;

	add_input_line(n_cols, n_rows, line, CHECKBOX_PATH_POWER_SPECTRA, 10000 + PATH_POWER_SPECTRA, "output directory:", PATH_POWER_SPECTRA, UI_TYPE_STRING_DIFFMICRO);
	++line;
//	add_button_line(n_cols, n_rows, line, CHECKBOX_POWER_SPECTRA_FILENAME, FLG_WRITE_POWER_SPECTRA, "write power spectra to :", "do not write power spectra", POWER_SPECTRA_FILENAME);
	add_button_line(n_cols, n_rows, line, CHECKBOX_POWER_SPECTRA_FILENAME, FLG_WRITE_POWER_SPECTRA, "write power spectra", "do not write power spectra", POWER_SPECTRA_FILENAME, UI_TYPE_STRING_DIFFMICRO);
	++line;
//	add_button_line(n_cols, n_rows, line, CHECKBOX_AZIMUTHAL_AVG_FILENAME, FLG_WRITE_AZIMUTHAL_AVG, "write angular average to :", "do not write angular average", AZIMUTHAL_AVG_FILENAME);
	add_button_line(n_cols, n_rows, line, CHECKBOX_AZIMUTHAL_AVG_FILENAME, FLG_WRITE_AZIMUTHAL_AVG, "write angular average", "do not write angular average", AZIMUTHAL_AVG_FILENAME, UI_TYPE_STRING_DIFFMICRO);
	
	++line;
	add_input_line(n_cols, n_rows, line, CHECKBOX_DIST_MAX, 10000 + DIST_MAX, "maximum delay :", DIST_MAX, UI_TYPE_INDEX_DIFFMICRO);
	++line;
	add_input_line(n_cols, n_rows, line, CHECKBOX_N_PW_AVERAGES, 10000 + N_PW_AVERAGES, "maximum number of average :", N_PW_AVERAGES, UI_TYPE_INDEX_DIFFMICRO);
	++line;
	add_input_line(n_cols, n_rows, line, CHECKBOX_FREQUENCY_MAX, 10000 + FREQUENCY_MAX, "maximum spatial frequency :", FREQUENCY_MAX, UI_TYPE_INDEX_DIFFMICRO);
	++line;
	add_button_line(n_cols, n_rows, line, CHECKBOX_EXECUTION_MODE, EXECUTION_MODE, "FIFO mode", "Time correlation mode", EXECUTION_MODE, UI_TYPE_STRING_DIFFMICRO);
	//++line;
	//add_input_line(n_cols, n_rows, line, CHECKBOX_VERSION, 10000 + VERSION, "Version :", VERSION, UI_TYPE_INDEX_DIFFMICRO);

	++line;
	add_button_line(n_cols, n_rows, line, CHECKBOX_HARDWARE_SELECTION, HARDWARE_SELECTION, "Run on CPU", "Run on GPU", HARDWARE_SELECTION, UI_TYPE_STRING_DIFFMICRO);
	++line;
	add_input_line(n_cols, n_rows, line, CHECKBOX_NUMBER_THREAD_GPU, 10000 + NUMBER_THREAD_GPU, "number of threads of gpu :", NUMBER_THREAD_GPU, UI_TYPE_INDEX_DIFFMICRO);
	++line;
	add_input_line(n_cols, n_rows, line, CHECKBOX_NUMBER_THREAD_CPU, 10000 + NUMBER_THREAD_CPU, "number of threads of cpu :", NUMBER_THREAD_CPU, UI_TYPE_INDEX_DIFFMICRO);
	++line;
	add_input_line(n_cols, n_rows, line, CHECKBOX_RAM_CPU, 10000 + RAM_CPU, "memory used by cpu :", RAM_CPU, UI_TYPE_INDEX_DIFFMICRO);
	++line;
	add_button_line(n_cols, n_rows, line, CHECKBOX_GRAPH_MODE, GRAPH_MODE, "Graph on", "Graph off", GRAPH_MODE, UI_TYPE_STRING_DIFFMICRO);
	++line;
	/*

	uivar = new ui_variable(UI_TYPE_BOOL_DIFFMICRO);
	button = new button_opencv(FLG_WRITE_POWER_SPECTRA,"write power spectra");
	button->x[0] = 1. / (double)(n_cols);
	button->x[1] = (double)(line) / (double)(n_rows);
	button->x[2] = button->x[0] + 4. / (double)(n_cols);
	button->x[3] = button->x[1] + 0.9 / (double)(n_rows);
	button->state = false;
	button->label_true = "write power spectra to :";
	button->label_false = "do not write power spectra";
	uivar->control = (control_opencv*)(button);
	option.push_back(uivar);

	++line;

	uivar = new ui_variable(UI_TYPE_BOOL_DIFFMICRO);
	button = new button_opencv(FLG_WRITE_AZIMUTHAL_AVG,"write angulat average");
	button->x[0] = 1. / (double)(n_cols);
	button->x[1] = (double)(line) / (double)(n_rows);
	button->x[2] = button->x[0] + 4. / (double)(n_cols);
	button->x[3] = button->x[1] + 0.9 / (double)(n_rows);
	button->state = false;
	button->label_true = "write angular avg to :";
	button->label_false = "do not write angular avg";
	uivar->control = (control_opencv*)(button);
	option.push_back(uivar);
	*/
#if 0
	/*!index position in the  directory of the first file to be processed*/
	
	FIRST_IMAGE = 0,
		//INDEX first_image;
		/*!index position in the  directory of the last file to be processed*/
		LAST_IMAGE = 1,
		//INDEX last_image;
		/*!number of maxium averages to calculate a power spectrum*/
		N_PW_AVERAGES = 2,
		//INDEX n_pw_averages;
		/*!maxium temporal delay that must be consider*/
		DIST_MAX = 3,
		//INDEX dist_max;

		/*!output flag: if ==true write on file the azimuthal averages*/
		FLG_WRITE_AZIMUTHAL_AVG = 11,
		//bool flg_write_azimuthal_avg;
		/*!output flag: if ==true write on file all the averaged power spectrum*/
		FLG_WRITE_POWER_SPECTRA = 12,
		//bool flg_write_power_spectra;
		/*!output flag: if ==true write on file the average intensity value of all the images*/
		FLG_WRITE_IMAGES_AVG = 13,
		//bool flg_write_images_avg;
		FLG_START = 14,

		/*!output path*/
		PATH_POWER_SPECTRA = 21,
		//std::string path_power_spectra;
		/*!path and generic root name for power spectrum output*/
		POWER_SPECTRA_FILENAME = 22,
		//std::string power_spectra_filename;
		/*!path and file name for azimuthal averages output*/
		AZIMUTHAL_AVG_FILENAME = 23,
		//std::string azimuthal_avg_filename;
		/*!path and file name for image averages output*/
		IMAGES_AVG_FILENAME = 24,
		//std::string images_avg_filename;
		/*!path and file name for elapsed time output*/
		LOG_FILENAME = 25,
		//std::string log_filename;
		/*!
		temporary directory where are store the images that can be loaded with
		bool load_binary_image(std::string &filename, INDEX &dimy, INDEX &dimx, bool read_im, unsigned short im[]);
		*/
		PATH_TEMPORARY = 26
		//std::string path_temporary;
#endif
	

}
