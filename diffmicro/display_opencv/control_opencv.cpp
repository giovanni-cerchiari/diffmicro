
/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 12/2015

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
#include "control_opencv.h"

control_opencv* select_control(control_opencv* to_be_selected, std::vector<control_opencv*> lst)
{
	int i;
	std::vector<control_opencv*>::iterator it, end;
	control_opencv* ct = NULL;

	end = lst.end();
	for (it = lst.begin(), i = 0; it != end; ++it, ++i)
	{
		if (to_be_selected != *it) (*it)->deselect();
		else
		{
			ct = *it;
			ct->select();
		}
	}

	return ct;
}


control_opencv::control_opencv(int _id, std::string _name)
{
	std::stringstream ss;

	id = _id;
	x = NULL;

	if (0 == _name.size())
	{
		ss << "control " << id;
		name = ss.str();
	}
	else
	{
		name = _name;
	}

}
control_opencv::control_opencv(const control_opencv& ctrl)
{
	*this = ctrl;
}


control_opencv::~control_opencv()
{
	if (NULL != this->x) delete[] this->x;
}



