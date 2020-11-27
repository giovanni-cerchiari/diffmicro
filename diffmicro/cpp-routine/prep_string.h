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



#ifndef _PREP_STRING_H_
#define _PREP_STRING_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

/*
This function removes front and back spaces from an input string

void del_front_back_space(std::string &str)
{
 unsigned int i, dim;
 std::string str2;

 str2 = str;
 i=0;
 while( (str.size()>0) and ( ('\t' == str[i]) or (' ' == str[i]) ) )
  {
	
  }
  
  while( (str.size()>0) and ( ('\t' == str[str.size()-1]) or (' ' == str[str.size()-1]) ) ) str.pop_back();
  
}*/

/*!Returns true if the string can be interpreted as a number otherwise false*/
bool is_numeric(std::string &str);
bool is_int(std::string &str);

/*!
This function is used to convert numbers written into a string into number-types of C++  
*/
template <typename TYPE>
TYPE convert_string(std::string &str)
{

 std::stringstream in;
 TYPE var;
 in <<str;
 in >> var;

 return (var);
}

/*!
This function is used to convert numbers into a string 
*/
template <typename TYPE>
std::string convert_to_string(TYPE val)
{

 std::stringstream in;
 in <<val;

 return (in.str());
}

/*!
This function split the input string in a vector of strings called out. separator is the character that is used
to separate the sub-strings. The separator is not included in any of the out[i] strings and empty strings are allowed
*/
void split(char separator, std::string in, std::vector<std::string> &out);

/*!
This function attempts to extract the foldername from a complete filename using '\' character
If folder name is not found returns false
*/
bool extract_folder_from_filename(std::string filename, std::string &folder);

/*!This function can be used to parse option of a configuration file*/
template<typename TYPE>
bool parse_option(std::string entry, std::vector<std::string> &splitted, TYPE &val)
{
	if (2 != splitted.size()) return false;
	if (0 != strcmp(entry.c_str(), splitted[0].c_str()))
	{
		return false;
	}
	else
	{
		val = convert_string<TYPE>(splitted[1]);
		return true;
	}
}

#endif
