/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 8/2011
update: 1/2015
*/

/*
Copyright: Mojtaba Norouzisadeh, Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
e-mail: norouzi.mojtaba.sade@gmail.com

date: 05/2020 - 09/2020
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

/*!
This functions are written for diffmicro.exe application.
*/


#ifndef _DIFF_MICRO_IO_H_
#define _DIFF_MICRO_IO_H_

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
#include "diffmicro_display.h"
#include "controls_opencv.h"

/*!Enumerator that is used to track the variable types*/
enum ui_variable_type_enum
{
	UI_TYPE_INDEX_DIFFMICRO = 0,
	UI_TYPE_MY_REAL_DIFFMICRO = 1,
	UI_TYPE_BOOL_DIFFMICRO = 2,
	UI_TYPE_STRING_DIFFMICRO = 3
};

/*!Enumerator that defines the possibile mode of execution according to different algorithms*/
enum ui_execution_mode_enum
{
	/*!FIFO mode is called in the article WITHOUT_FT algorithm.*/
	DIFFMICRO_MODE_FIFO = 0,
	/*!TIMECORRELATION mode is called in the article WITH_FT algorithm.*/
	DIFFMICRO_MODE_TIMECORRELATION = 1
};

/*!Enumerator used as id of the entries of the graphical user interface window*/
enum ui_variable_enum
{
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
	//* maximum frequency to save in gpu for each power spectra
	FREQUENCY_MAX = 4,

	TIME_STEP = 5,
	/*!Number of parallel cuda-threads to be used in the GPU core execution*/
	NUMBER_THREAD_GPU = 6,
	/*!Number of parallel cpu threads that will be used in the CPU execution*/
	NUMBER_THREAD_CPU = 7,
	/*!Ammount of memory dedicated on the RAM for the program execution (only valid in CPU mode)*/
	RAM_CPU = 8,

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
	
	/*!Execution mode selection*/
	EXECUTION_MODE=15,
	
	/*!Selection between CPU and GPU execution*/
	HARDWARE_SELECTION = 16,

	/*!option to pop up the graphs*/
	GRAPH_MODE = 17,

	/*!input path*/
	PATH_IMAGES = 20,
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
	PATH_TEMPORARY = 26,
	//std::string path_temporary;

	TIME_FILENAME = 27,

	VALID_IMAGE_FILENAME = 28,


	CHECKBOX_PATH_IMAGES = 100,
	CHECKBOX_FIRST_IMAGE = 101,
	CHECKBOX_LAST_IMAGE = 102,
	CHECKBOX_PATH_POWER_SPECTRA = 103,
	CHECKBOX_FREQUENCY_MAX = 104,
	CHECKBOX_TIME_STEP = 105,
	CHECKBOX_POWER_SPECTRA_FILENAME = 122,
	CHECKBOX_AZIMUTHAL_AVG_FILENAME = 123,
	CHECKBOX_LOG_FILENAME = 124,
	CHECKBOX_DIST_MAX = 125,
	CHECKBOX_N_PW_AVERAGES = 126,
	CHECKBOX_TIME_FILENAME = 127,
	CHECKBOX_VALID_IMAGE_FILENAME = 128,

	CHECKBOX_EXECUTION_MODE = 129,
	CHECKBOX_HARDWARE_SELECTION = 130,
	CHECKBOX_NUMBER_THREAD_GPU = 131,
	CHECKBOX_NUMBER_THREAD_CPU = 132,
	CHECKBOX_RAM_CPU = 133,
	CHECKBOX_GRAPH_MODE = 129,


	FLG_TIME_FILENAME = 1027,
	FLG_VALID_IMAGE_FILENAME = 1028

};

class ui_variable
{
public:
	ui_variable(int _type){ flg = false; type = _type; control = NULL; }
	virtual ~ui_variable(){ if (NULL != control)delete control; }

	control_opencv *control;

	int type;
	bool flg;
};



/*!
this structure is used to sort the file names by associating the string of the name to a sorting number
*/
struct file_num
{
	std::string filename;
	unsigned int num;
};

/*!
function used with std::sort for comparisions of file_num structures
*/
bool comp_filenum(file_num &f1, file_num &f2);
/*!
This function calculates the number to assign to the structure filenum starting from the string file.
filenum.num will be the last number readable in the string file
*/
void file_num_assign(std::string &file, file_num &filenum);


/*!
This class is written for diffmicro application.
This class stores all the user preferences about the execution of the program.
*/
class diffmicro_user_interface
{
	public:
		/*!constructor*/
		diffmicro_user_interface();
		/*!destructor*/
		~diffmicro_user_interface();

		bool flg_start;

		/*!index position in the  directory of the first file to be processed*/
		INDEX first_image;
		/*!index position in the  directory of the last file to be processed*/
		INDEX last_image;
		/*!number of maxium averages to calculate a power spectrum*/
		INDEX n_pw_averages;
		/*!maxium temporal delay that must be consider*/
		INDEX dist_max;
		/*!maximum frequency to save for each power spectra*/
		INDEX frequency_max;
		/*!quantization time step. It must fit the minimum time difference between to images so that round(min/time_step)>0*/
		MY_REAL time_step;

		/*!output flag: if ==true write on file the azimuthal averages*/
		bool flg_write_azimuthal_avg;
		/*!output flag: if ==true write on file all the averaged power spectrum*/
		bool flg_write_power_spectra;
		/*!output flag: if ==true write on file the average intensity value of all the images*/
		bool flg_write_images_avg;
		
		/*!Execution flag. See ui_execution_mode_enum for further details.*/
		bool flg_execution_mode;
		/*!Flag that indicates if display of the output should be made.*/
		bool flg_graph_mode;
		/*!Flag that indicates the hardware platform of execution of the algorithm.*/
		bool flg_hardware_selection;

		/*! flag. Does the user provide a file with the times?*/
		bool flg_time_file;
		/*! flag. Does the user provide a file indicating the valid images?*/
		bool flg_valid_image_file;

		/*!execution mode. Use ui_execution_mode_enum to assign meaningful values*/
		int execution_mode;

		/*!output path*/
		std::string path_power_spectra;
		/*!path and generic root name for power spectrum output*/
		std::string power_spectra_filename;
		/*!path and file name for azimuthal averages output*/
		std::string azimuthal_avg_filename;
		/*!path and file name for image averages output*/
		std::string images_avg_filename;
		/*!path and file name for elapsed time output*/
		std::string log_filename;
		/*!path and file name for time sequence interpretation*/
		std::string time_sequence_filename;

		/*!maximum number of threads for CPU execution*/
		INDEX nthread;
		/*!maximum number of threads for GPU execution*/
		INDEX nthread_gpu;
		/*!Ammount of RAM memory that diffmicro can use in case of CPU execution*/
		INDEX RAM;
		/*!Decide the hardware where the computation will be performed*/
		INDEX hardware;
		/*!Input path*/
		std::string path_input;


		std::string file_times;
		/*File that indicates which images are valid. It can only be used in FIFO mode.*/
		std::string file_valid_images;

		bool get_bool(int id);
		INDEX get_int(int id);
		std::string get_string(int id);

		bool set_bool(int id, bool value);
		bool set_int(int id, INDEX value);
		bool set_string(int id, std::string value);

		/*!refresh the window on screen*/
		void show();
		/*!resize the window on screen*/
		void resize_window(unsigned int display_dimx, unsigned int display_dimy);
		/*!move window on screen*/
		void move_window(float x, float y);
		/*!move window on screen*/
		void move_window(int x, int y);
		/*!This static method is called on mouse action*/
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
		/*!This method check if all the values of the GUI is resonable (numeric if it should, if the indicated folders exists...)*/
		bool all_setted();

		/*!This method converts the text useri interface into the graphical user interface*/
		void variables_to_gui();
		/*!This method converts the graphical user interface into the text based one*/
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
	/*! method used by all_setted*/
	int check_path_button_set_box(int check_box_id, int button_id, int variable_id);
	/*! method used by all_setted*/
	int check_positive_integer_set_box(int check_box_id, int string_id);
	/*! method used by all_setted*/
	int check_filename_set_box(int check_box_id, int path_id, int filename_id);
	/*!Used at creation of the window of the GUI*/
	void add_input_line(int n_cols, int n_rows, int line, int check_box_id, int label_id, std::string label, int variable_id, int variable_type, std::string defaultvalue = "");
	/*!Used at creation of the window of the GUI*/
	void add_button_line(int n_cols, int n_rows, int line, int check_box_id, int button_id, std::string label_true, std::string label_false, int variable_id, int variable_type, std::string defaultvalue = "");

	bool adjust_time_spacing(std::vector<std::string> &file_list, std::vector<MY_REAL> time_input, MY_REAL time_step, std::vector<bool> &flg_valid_image);

	void initialize();
	void translate();

};

/*!
this is the global instance for the user interfac that is used in the program
*/
extern diffmicro_user_interface useri;

/*!
This function writes a vector to a binary file
*/
template<typename FLOAT>
bool write_vet_to_file(std::string &filename, INDEX dim, FLOAT vet[])
{
	unsigned short size;
	FILE *pfile;

	pfile = fopen(filename.c_str(), "wb");
	if(NULL == pfile)
		{
			std::cerr <<"error attempting to write : "<<filename<<std::endl;
			return false;
		}

	size = sizeof(FLOAT);
	fwrite(&size, sizeof(unsigned short), 1, pfile);

	size = (unsigned short)(dim);
	fwrite(&size, sizeof(unsigned short), 1, pfile);

	fwrite(vet, sizeof(FLOAT), dim, pfile);

	fclose(pfile);

	return true;
}

/*!
This function writes a matrix to a binary file
*/
template<typename FLOAT>
bool write_mat_to_file(std::string &filename, INDEX dimy, INDEX dimx, FLOAT mat[])
{
	
	unsigned short size;
	FILE *pfile;

	pfile = fopen(filename.c_str(), "wb");

	if(NULL == pfile)
		{
			std::cerr <<"error attempting to write : "<<filename<<std::endl;
			return false;
		}

	size = sizeof(FLOAT);
	fwrite(&size, sizeof(unsigned short), 1, pfile);

	size = (unsigned short)(dimy);
	fwrite(&size, sizeof(unsigned short), 1, pfile);
	size = (unsigned short)(dimx);
	fwrite(&size, sizeof(unsigned short), 1, pfile);

	fwrite(mat, sizeof(FLOAT), dimy * dimx, pfile);

	fclose(pfile);

	return true;
}

/*!This function is used to save the partial results of the timeseries analysis to file*/
template<typename FLOAT>
void writeappend_partial_lutpw(INDEX ichunck, INDEX dist, INDEX dimchunck, FLOAT power_spectrum[])
{
	std::string filename;
	FILE* pfile;

	filename = useri.power_spectra_filename + convert_to_string(dist) + ".mat";
	// if it is the first chunck we create the file, otherwise we append the data
	if (ichunck > 0)
		pfile = fopen(filename.c_str(), "ab");
	else
		pfile = fopen(filename.c_str(), "wb");

	fwrite(power_spectrum, sizeof(FLOAT), dimchunck, pfile);

	fclose(pfile);
}


/*!This function is used to save the partial results of the timeseries analysis to file*/
template<typename FLOAT>
void read_mergedpartial_lutpw(INDEX dist, INDEX dimlut, FLOAT power_spectrum[])
{
	std::string filename;
	FILE* pfile;

	filename = useri.power_spectra_filename + convert_to_string(dist) + ".mat";

	pfile = fopen(filename.c_str(), "rb");

	fread(power_spectrum, sizeof(FLOAT), dimlut, pfile);

	fclose(pfile);
}


/*!
With the help of : bool write_mat_to_file(std::string &filename, INDEX dimy, INDEX dimx, FLOAT mat[])
this function writes to file the power spectra stored in power_spectra[].
The name of the file is fixed and has an increasing index number that starts from start_dist value
*/
void write_power_spectra(INDEX start_dist, INDEX npw, INDEX dimy, INDEX dimx, STORE_REAL power_spectra[]);

/*!
This functon reads images that are store with this binary format:
- uint16 dimy x1
- uint16 dimx x1
- uint16 images x(dimx*dimy)

the memory area unsigned short im[] should be prepared in advance. So if you call the function with
read_im == false the image is not read from the hard drive and yuo can recollect the image size
information
*/
bool load_binary_image(std::string &filename, INDEX &dimy, INDEX &dimx, bool read_im, unsigned short im[], bool flg_display = false);

/*!Load any grayscale image format to matrix of unsigned char using opencv*/
bool load_image(std::string &filename, INDEX &dimy, INDEX &dimx, bool read_im, unsigned short im[], bool flg_display = false);


bool get_file_lines_number(std::string filename, INDEX &n_lines);

template<typename MY_TYPE>
bool load_one_column_file(std::string filename, std::vector<MY_TYPE> &vet)
{
	std::fstream fin;
	std::string line;
	INDEX dim;

	if (false == get_file_lines_number(filename, dim))
	{
		std::cerr << "invalid time file: " << filename << std::endl;
		return false;
	}

	vet.clear();
	vet.reserve(dim);

	fin.open(filename.c_str(), std::ios::in);
	while (fin.good())
	{
		std::getline(fin, line);
		vet.push_back(convert_string<MY_TYPE>(line));
	}
	fin.close();
	return true;
}

ui_variable* select_control(ui_variable* to_be_selected, std::list<ui_variable*> lst);

/*This function starts the graphical user interface.*/
void start_gui(std::string file_ui, bool flg_force_start_gui = false);

#endif