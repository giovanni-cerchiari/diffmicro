/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com

date: 2011
update: 2016
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


/*!This file contains the functions and classes used to communicate with the program by the user.
The user interface is meant to be a text file with a specific structure that can be reconstructed 
via the provided graphical user interface.*/
#ifndef _DIFFMICRO_UI_H_
#define _DIFFMICRO_UI_H_

#include "control_opencv.h"

class diffmicro_user_interface
{
public:
	diffmicro_user_interface();
	~diffmicro_user_interface();

  /*!This function parses the text file of the user interface and correct non-compatible values.*/
	void load(std::string filename);

};



#endif
