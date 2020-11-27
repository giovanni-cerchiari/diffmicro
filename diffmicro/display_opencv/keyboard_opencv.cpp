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
#include "stdafx.h"
#include <iomanip>
#include <sstream>
#include "mouse_opencv.h"
#include "controls_opencv.h"
#include "figure_opencv.h"
#include "keyboard_opencv.h"


void(*handle_on_matrix_callback_F2_paralize_save)(int *x, double *xx, void *fig) = NULL;

std::string snapshot_folder(".\\");
unsigned int snapshot_sequential_number(0);

int refresh_idle(int key)
{
	return key;
}

int refresh_exit(int key)
{
	return 27;
}

int(*refresh)(int key) = refresh_idle;

void waitkeyboard(int delay)
{
	unsigned int i, j;
	int old_delay_F2_save = delay;
	int key=0;
	std::string text;
	std::stringstream num;
	bool F2_pause_toggle = false;

	handle_on_matrix_callback_F2_paralize_save = handle_on_matrix_callback;

	while (27 != key) //ESC
	{
		key = cv::waitKey(delay);

		//if(-1!=key)std::cout << "key = " << key << std::endl;
		switch (key)
		{
		case 43: //+
			i = (unsigned int)((float)(active_figure->display_dimx())*(float)(1.1));
			j = (unsigned int)((float)(active_figure->display_dimy())*(float)(1.1));
			active_figure->create_display_window(i, j);
			active_figure->show();
			break;
		case 45: //-
			i = (unsigned int)((float)(active_figure->display_dimx())*(float)(0.9));
			j = (unsigned int)((float)(active_figure->display_dimy())*(float)(0.9));
			active_figure->create_display_window(i, j);
			active_figure->show();
			break;
		case 8:// backspace
			if (NULL != mouse.manual_points)	mouse.manual_points->pop_back();
			break;
		case 13: //enter
			if (NULL != mouse.manual_points)
			{
				mouse.manual_points->color_normal = cv::Scalar(50,100,200);
				active_figure->overlay.push_back((overlay_opencv*)(mouse.manual_points));
				mouse.manual_points = NULL;
			}
		case 7340032://F1
			num.str("");
			num << std::setw(3) << std::setfill('0') << std::internal << snapshot_sequential_number;
			active_figure->save_display(snapshot_folder + "snapshot"+num.str()+".bmp");
			++snapshot_sequential_number;
			
			break;
		case 7405568: //F2
			if (false == F2_pause_toggle)
			{
				handle_on_matrix_callback_F2_paralize_save = handle_on_matrix_callback;
				handle_on_matrix_callback = NULL;
				old_delay_F2_save = delay;
				delay = 0;
			}
			else
			{
				handle_on_matrix_callback = handle_on_matrix_callback_F2_paralize_save;
				delay = old_delay_F2_save;
			}
			F2_pause_toggle = !F2_pause_toggle;
			break;
		default:
			break;
		}


		/*
		if (NULL != active_figure->active_control)
		{
			if (CTRL_TEXTBOX == active_figure->active_control->type)
			{
				active_figure->active_control->get_value(&text);

				switch (key)
				{
				case 8: //BACKSPACE
					text.pop_back();
					break;
				case 3014656: //DEL
					break;
				default:
					text.push_back((char)(key));
					std::cerr << "pressed key = " << (char)(key) << "(" << (int)(key) << ")" << std::endl;
					break;
				}
				active_figure->active_control->set_value(&text);
			}
		}
		*/
		key = refresh(key);
	}


}
