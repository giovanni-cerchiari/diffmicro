
/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 8/2011
*/

/*
Copyright:  Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com

update: 11/2018
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


#ifndef _DIR_H_
#define _DIR_H_

#include <string>
#include <vector>
#include <list>

#include "dirent.h"

/*!
 this function list the subdirectories and the files contained in the directory whose path is path
 if return value = true the folder in path exist
*/
bool list_directory(std::string path, std::vector<std::string > &list_dir, std::vector<std::string > &list_file);

/*!This function gives back all the files that contains a certain string pattern inside*/
void load_file_pattern(std::string directory_to_search, std::string &pattern, std::list<std::string> &list_load, bool flg_recursive = true);

#endif
