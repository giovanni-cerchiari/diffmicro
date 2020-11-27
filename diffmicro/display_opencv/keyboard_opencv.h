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

#ifndef _KEYBOARD_OPENCV_H_
#define _KEYBOARD_OPENCV_H_

#include <string>

/*!This variable can be used to indicate where the snapshot of the program are stored*/
extern std::string snapshot_folder;
/*!This variable indicates the number of the snapshot that will appear in the file name.*/
extern unsigned int snapshot_sequential_number;

/*!This is an example refresh function. The refresh function is necessary to force display.*/
int refresh_idle(int key);
/*!This is an example refresh function. The refresh function is necessary to force display.*/
int refresh_exit(int key);
/*! refresh function is called after a certain delay or when a button is pressed*/
extern int(*refresh)(int key);
/*!we grab the keyboard key press in case of event. The delay allows to call function refresh even if no button is pressed*/
void waitkeyboard(int delay = 0);

#endif
