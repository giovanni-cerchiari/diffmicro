/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 8/2011
update: 1/2016
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

/*!
This functions and classes are written for diffmicro.exe application.
*/
#include "stdafx.h"
#include "diffmicro_io.h"
#include "figure_opencv.h"


diffmicro_user_interface useri;

bool comp_filenum(file_num &f1, file_num &f2)
{
	return (f1.num < f2.num);
}


void file_num_assign(std::string &file, file_num &filenum)
{
	INDEX i, size;
	filenum.filename = file;
	std::string character;
	std::string numstr;
	bool isnumeric;
	bool flg_first_number = true;;

	size = (INDEX) (file.size());

	for(i=1; i<size; ++i)
		{
			character = file[size - i];
			isnumeric = is_numeric(character);

			if((true == isnumeric) && (true == flg_first_number))
					flg_first_number = false;

			if(true == isnumeric)	numstr.push_back(character[0]);
			else	                 if(false == flg_first_number) break;

		}

	if(0 != numstr.size())
		{
			std::reverse(numstr.begin(), numstr.end());
			filenum.num = convert_string<unsigned int>(numstr);
		}
	else
		{
			filenum.num = 0;
		}
}


/*!
string comparision for sorting pourposes
*/
bool comp_str(std::string &st1, std::string &st2)
{
	INDEX i;
	i=0;
	while(st1[i] == st2[i]) ++i;

	return (st1[i] < st2[i]);
}




diffmicro_user_interface::diffmicro_user_interface()
{

	//initialize();
	
	first_image = 0;
	last_image = 0;
	n_pw_averages = 0;
	dist_max = 0;
	frequency_max = 0;
	nthread = N_MAX_THREADS;
	nthread_gpu = 1024;
	RAM = (INDEX)(1) << 30;
	RAM = (INDEX)(1) << 27;
	//useri.hardware = HARDWARE_GPU,

	// default init
	flg_write_azimuthal_avg = true;
	flg_write_power_spectra = false;
	flg_write_images_avg = true;

	flg_start = false;
	flg_execution_mode = true;
	//execution_mode = DIFFMICRO_MODE_FIFO;
	//execution_mode = DIFFMICRO_MODE_TIMECORRELATION;
}


diffmicro_user_interface::~diffmicro_user_interface()
{
	while (0 < option.size())
	{
		delete *(option.begin());
		option.pop_front();
	}
	panel.deallocate();
	if (0<name.size())	cv::destroyWindow(name);
}

ui_variable* diffmicro_user_interface::get_variable(int id)
{
	std::list<ui_variable*>::iterator it, end;

	end = option.end();
	for (it = option.begin(); it != end; ++it)
	{
		if (id == (*it)->control->id)
		{
			return *it;
		}
	}

	return NULL;
}



bool diffmicro_user_interface::get_bool(int id)
{
	bool flg;
	ui_variable* var = get_variable(id);
	if (NULL == var) return false;
	if (UI_TYPE_BOOL_DIFFMICRO != var->type) return false;
	var->control->get_value(&flg);
	return flg;
}

bool diffmicro_user_interface::set_bool(int id, bool value)
{
	ui_variable* var = get_variable(id);
	if (NULL == var) return false;
	if (UI_TYPE_BOOL_DIFFMICRO != var->type) return false;
	var->control->set_value(&value);
	return true;
}


INDEX diffmicro_user_interface::get_int(int id)
{
	std::string text;
	INDEX val;
	ui_variable* var = get_variable(id);
	if (NULL == var) return 0;
	if (UI_TYPE_INDEX_DIFFMICRO != var->type) return 0;
	var->control->get_value(&text);
	if (false == is_int(text)) return 0;

	return convert_string<INDEX>(text);
}

bool diffmicro_user_interface::set_int(int id, INDEX value)
{
	std::string valuestr;
	ui_variable* var = get_variable(id);
	if (NULL == var) return false;
	if (UI_TYPE_INDEX_DIFFMICRO != var->type) return false;
	valuestr = convert_to_string(value);
	var->control->set_value(&valuestr);
	return true;
}

std::string diffmicro_user_interface::get_string(int id)
{
	std::string text;
	ui_variable* var = get_variable(id);
	if (NULL == var) return "";
	if (UI_TYPE_STRING_DIFFMICRO != var->type) return "";
	var->control->get_value(&text);
	return text;
}
bool diffmicro_user_interface::set_string(int id, std::string value)
{
	ui_variable* var = get_variable(id);
	if (NULL == var) return false;
	if (UI_TYPE_STRING_DIFFMICRO != var->type) return false;
	var->control->set_value(&value);
	return true;
}





bool load_binary_image(std::string &filename, INDEX &dimy, INDEX &dimx, bool read_im, unsigned short im[], bool flg_display)
{
	FILE *pfile;
	unsigned short size;

	pfile = fopen(filename.c_str(), "rb");
	if(NULL == pfile)
		{
			std::cerr <<"error attempting to read : "<<filename<<std::endl;
			return false;
		}
	fread(&size, sizeof(unsigned short), 1, pfile);
	dimy = (INDEX)(size);
	fread(&size, sizeof(unsigned short), 1, pfile);
 dimx = (INDEX)(size);
	if(true == read_im) fread(im, sizeof(unsigned short), dimx*dimy, pfile);
	fclose(pfile);


	if (true == flg_display) display_read(im);

	return true;
}





/*!
 This functions returns alla the files contained in path with extension ".dat"
*/
bool diffmicro_user_interface::prep_file_list(std::string path, std::vector<std::string > &list_file)
{
	INDEX i,size_tmp, size;
	std::vector<std::string > list_dir;
	std::vector<std::string > list_file_tmp;
	std::vector< file_num> filenum;
	std::vector< file_num>::iterator it, end;
	file_num filenum_tmp;
	std::string extension;

	if(false == list_directory(path, list_dir, list_file_tmp))
		{
			return false;
		}

	//sort(list_file_tmp.begin(), list_file_tmp.end(), comp_str);

//	for(i=0; i<list_file_tmp.size(); ++i)
//	std::cerr <<list_file_tmp[i]<<std::endl;

	list_file.clear();
	size_tmp = (INDEX)(list_file_tmp.size());
	for(i=0; i<size_tmp; ++i)
		{
			size = (INDEX)(list_file_tmp[i].size());

			if (size > 4)
			{
				extension = list_file_tmp[i].substr(list_file_tmp[i].size() - 4, 4);
				if ((0 == strcmp(".dat", extension.c_str())) ||
					(0 == strcmp(".bin", extension.c_str())) ||
		   			(0 == strcmp(".tif", extension.c_str())) ||
			   		(0 == strcmp("tiff", extension.c_str())) )
					{
						file_num_assign(list_file_tmp[i], filenum_tmp);
						filenum.push_back(filenum_tmp);
					}
				}
		}

	sort(filenum.begin(), filenum.end(), comp_filenum);

	end = filenum.end();
	for(it = filenum.begin(); it!=end; ++it)
		{
			//std::cerr <<it->num<<std::endl;
			list_file.push_back(path + it->filename);
		}

	//std::cout <<"# of files : "<<list_file.size()<<std::endl;

//	for(i=0; i<list_file.size(); ++i)
//	std::cerr <<list_file[i]<<std::endl;

	return true;
}


void write_power_spectra(INDEX start_dist, INDEX npw, INDEX dimy, INDEX dimx, STORE_REAL power_spectra[])
{
	INDEX i;
	INDEX dim = dimx * dimy;
	std::string filename;

	for(i=0; i<npw; ++i)
		{
			filename = useri.power_spectra_filename + convert_to_string(start_dist + i) + ".mat";
			write_mat_to_file(filename, dimy, dimx, &(power_spectra[i * dim]));
		}
}

bool load_image(std::string &filename, INDEX &dimy, INDEX &dimx, bool read_im, unsigned short im[], bool flg_display)
{
	//int Binary = 1;
	FILE* file_bin;
	INDEX i, j;
	cv::Mat img_cv;
	int nb_val_lues;
	unsigned short dx, dy;
	
	if ((true == read_im) && (useri.binary == 1)) {
		file_bin = fopen(filename.c_str(), "rb");
		if (file_bin == NULL)
			std::cout << "ERROR" << std::endl;
		nb_val_lues = fread(&dx, sizeof(unsigned short), 1, file_bin);
		nb_val_lues = fread(&dy, sizeof(unsigned short), 1, file_bin);
		dimy = dy;
		dimx = dx;
		nb_val_lues = fread(im, sizeof(unsigned short), dimx*dimy, file_bin);
		fclose(file_bin);
	}
	else {
		if (useri.binary != 1) {
			img_cv = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH);

			if (!img_cv.data)                              // Check for invalid input
			{
				std::cerr << "Could not open or find image : " << filename << std::endl;
				return false;
			}

			//std::cerr << "img_cv.elemSize() = " << img_cv.elemSize() << std::endl;

			dimx = img_cv.cols;
			dimy = img_cv.rows;
			if ((true == read_im) && (1 == img_cv.elemSize()))
			{
				for (j = 0; j < img_cv.rows; ++j)
					for (i = 0; i < img_cv.cols; ++i)
						im[j * dimx + i] = (unsigned short)(img_cv.data[j * img_cv.step[0] + i]);
			}
			if ((true == read_im) && (2 == img_cv.elemSize()))
			{
				for (j = 0; j < img_cv.rows; ++j)
					for (i = 0; i < img_cv.cols; ++i)
						im[j * dimx + i] = *((unsigned short*)(&(img_cv.data[j * img_cv.step[0] + i * img_cv.step[1]])));
			}
		}
	}
	if (true == flg_display) display_read(im);
	return true;
}



void diffmicro_user_interface::resize_window(unsigned int display_dimx, unsigned int display_dimy)
{
	unsigned int i;
	unsigned char* panel_data_i_uc;
	panel.deallocate();
	panel.create(display_dimy, display_dimx, CV_8U);
	for (i = 0; i < panel.rows * panel.cols; ++i)
	{
		panel_data_i_uc = (unsigned char*)(&(panel.data[i]));
		panel_data_i_uc[0] = 0;
	}
	cv::cvtColor(panel, panel, CV_GRAY2RGB);
}

void diffmicro_user_interface::move_window(int x, int y)
{
	int dimx, dimy;
	GetDesktopResolution(dimx, dimy);
	if ((x >= 0) && (y >= 0) && (x < dimx) && (y < dimy))
	{
		cv::moveWindow(name, x, y);
	}
}

void diffmicro_user_interface::move_window(float x, float y)
{
	int dimx, dimy;
	GetDesktopResolution(dimx, dimy);
	if ((x >= 0) && (y >= 0) && (x < 1) && (y < 1))
	{
		this->move_window((int)(x*(float)(dimx)), (int)(y*(float)(dimy)));
	}
}

ui_variable* select_control(ui_variable* to_be_selected, std::list<ui_variable*> lst)
{
	int i;
	std::list<ui_variable*>::iterator it, end;
	ui_variable* ct = NULL;

	end = lst.end();
	for (it = lst.begin(), i = 0; it != end; ++it, ++i)
	{
		if (to_be_selected != *it) (*it)->control->deselect();
		else
		{
			ct = *it;
			ct->control->select();
		}
	}

	return ct;
}
void diffmicro_user_interface::callback_mouse(int ev, int x, int y, int flags, void* afig)
{
	int i;
	std::list<ui_variable*>::iterator it, end;

	useri.xx[0] = (float)(x) / (float)(useri.panel.cols);
	useri.xx[1] = (float)(y) / (float)(useri.panel.rows);

	if (cv::EVENT_LBUTTONDOWN == ev)
	{
		useri.active_variable = NULL;
		// set onclick position parameters
		for (i = 0; i < 2; ++i) useri.xx[4 + i] = useri.xx[i];

		end = useri.option.end();
		for (it = useri.option.begin(); it != end; ++it)
		{
			if (0 <= (*it)->control->handle(useri.xx))
			{
				useri.active_variable = *it;
				select_control(useri.active_variable, useri.option);
				break;
			}
		}
		if (NULL != useri.active_variable) useri.active_variable->control->onclick(useri.xx);

	}

	if (cv::EVENT_MOUSEMOVE == ev && cv::EVENT_FLAG_LBUTTON == flags)
	{
		if (NULL != useri.active_variable) useri.active_variable->control->onmove(useri.xx);
	}

	if (cv::EVENT_LBUTTONUP == ev)
	{
		if (NULL != useri.active_variable)
		{
			useri.active_variable->control->onrelease(useri.xx);
			if (CTRL_TEXTBOX != useri.active_variable->control->type)
			{
				useri.active_variable->control->deselect();
				useri.active_variable = NULL;
			}
		}
	}

	// set previous position parameters
	for (i = 0; i < 2; ++i) useri.xx[2 + i] = useri.xx[i];
	useri.show();

}


void diffmicro_user_interface::show()
{
	std::list<ui_variable*>::iterator it, end;
	end = option.end();
	for (it = option.begin(); it != end; ++it)
	{
		(*it)->control->draw(this->panel);
	}

	imshow(name, panel);
}

int diffmicro_user_interface::keyboard_refresh(int key)
{ 
	std::string txt;

	if (true == useri.all_setted()) return 27;
	else
	{
		if (0 < key)
		{
			if (NULL != useri.active_variable)
			{
				if (CTRL_TEXTBOX == useri.active_variable->control->type)
				{
					useri.active_variable->control->get_value(&txt);
					switch (key)
					{
					case 8: // backspace
						if (0 < txt.size()) txt.pop_back();
						break;
					default:
						txt.push_back(key);
						break;
					}
					
					useri.active_variable->control->set_value(&txt);
				}
			}

			useri.show();
		}
		return key;
	}
}

void diffmicro_user_interface::close_window()
{
	if (0<name.size())	cv::destroyWindow(name);
	name = "";
}

void start_gui(std::string file_ui, bool force_start_gui)
{
	ui_variable *var;

	if (false == force_start_gui)
	{
		if (0 == file_ui.size())
		{
			file_ui = "user.txt";
			force_start_gui = true;
		}
	}

	while (false == useri.load(file_ui) || true == force_start_gui)
	{
		useri.variables_to_gui();
		if (true == force_start_gui)
		{
			force_start_gui = false;
			var = useri.get_variable(FLG_START); var->control->set_value(&force_start_gui);
		}
		while (false == useri.all_setted())
		{
			useri.show();
			waitkeyboard(100);
		}
		useri.gui_to_variables();
		useri.save(file_ui);
	}

	useri.close_window();
}


void diffmicro_user_interface::write_time_sequence_to_file()
{
	INDEX i, dim, j;
	std::fstream fout;
	bool flg_write_valid_image = false;

	fout.open(time_sequence_filename.c_str(), std::ios::out);

	dim = file_list.size();

	if (flg_valid_image.size() == dim) flg_write_valid_image = true;

	for (i = 0; i < dim; ++i)
	{
		fout << i << "\t";
		if (true == flg_write_valid_image)
		{
			if (true == flg_valid_image[i]) j = 1;
			else                            j = 0;
			fout << j << "\t";
		}
		fout << file_list[i] << std::endl;
	}


	fout.close();
}

