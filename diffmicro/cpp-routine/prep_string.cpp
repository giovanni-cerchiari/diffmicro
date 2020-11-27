
/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 8/2009
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

#include "prep_string.h"

bool is_numeric(std::string &str)
{
 unsigned int i, dim, start;
	unsigned int n_point = 0;
 char c;

 dim = str.size();
	if (0 == dim) return false;

	if ('-' != str[0])
	{
		start = 0;
	}
	else
	{
		if (1 >= dim) return false;
		start = 1;
	}

 for(i=start ; i<dim; ++i)
 {
  c = str[i];
  if ( ((c < '0') || (c > '9')) && (c != '.') )
   return false;

		if (c == '.') ++n_point;
 }

	if (2>n_point) return true;
	else           return false;
   
}

bool is_int(std::string &str)
{
	unsigned int i, dim, count;
	count = 0;
	
	if (true == is_numeric(str))
	{
		dim = str.size();
		for (i = 0; i < dim; ++i)
		{
			if ('.' == str[i])++count;
		}
		if (0 == count) return true;
		else            return false;
	}
	else
	{
		return false;
	}

}

void split(char separator, std::string in, std::vector<std::string> &out)
{
	if (0 == in.size()) return;

 unsigned int i, dim;
 std::string field;
 
 if(in[in.size()-1]!=separator) in.push_back(separator);
  
 dim = in.size();
 out.clear();
	//out.resize(0);
 for(i=0; i<dim; ++i)
  {
	if(in[i] != separator)
	 {
	  field.push_back(in[i]);
	 }
	else
	 {
	  out.push_back(field);
	  field.clear();
	 }
  }
 
}

bool extract_folder_from_filename(std::string filename, std::string &folder)
{
	int i, dim;

	dim = (int)(filename.size());

	for (i = dim - 1; i >= 0; --i)
	{
		if ('\\' == filename[i] || '/' == filename[i]) break;
	}

	if (0 > i) return false;
	if (i == dim) return false;

	folder = filename.erase(i+1, dim - i);

	return true;
}
