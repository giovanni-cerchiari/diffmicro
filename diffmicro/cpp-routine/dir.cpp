/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 8/2011
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

#include <iostream>
#include "dir.h"


/*!
 this function list the subdirectories and the files contained in the directory whose path is path
 if return value = true the folder in path exist
*/
bool list_directory(std::string path, std::vector<std::string > &list_dir, std::vector<std::string > &list_file)
{
 DIR *dirp;
 dirent *file;
 std::string file_name;

 list_dir.clear();
 list_file.clear();

 dirp = dirent_opendir(path.c_str());
 if (dirp == NULL) return false;
 
 file = dirent_readdir(dirp);
 while(file != NULL)
  {
   file_name = file->d_name;

   if(file->d_type == DT_DIR) list_dir.push_back(file_name);
   if(file->d_type == DT_REG) list_file.push_back(file_name);

   file = dirent_readdir(dirp);
  }
 
 dirent_closedir(dirp);
 
 return true;
}

/*!This function gives back all the files that contains a certain string pattern inside*/
void load_file_pattern(std::string directory_to_search, std::string &pattern, std::list<std::string> &list_load, bool flg_recursive)
{
	DIR *directory;
	dirent *entry;
	std::string path;
	std::string name;

	path = directory_to_search;
	if ('\\' != path[path.size() - 1]) path.push_back('\\');

//	std::cerr << "path = " << path << std::endl << std::endl;

	directory = dirent_opendir(directory_to_search.c_str());

	entry = dirent_readdir(directory); // reading .
	entry = dirent_readdir(directory); // reading ..
	entry = dirent_readdir(directory);
	while (NULL != entry)
	{
		if (DT_DIR == entry->d_type && true == flg_recursive)
			load_file_pattern(path + std::string(entry->d_name), pattern, list_load);

		if (DT_REG == entry->d_type)
		{
			name = entry->d_name;
			if (std::string::npos!=name.find(pattern)) // if we found the pattern
				list_load.push_back(path + name);
		}
	
		entry = dirent_readdir(directory);
	}

	dirent_closedir(directory);
}
