
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

#ifndef _CIRCULAR_INDEX_H_
#define _CIRCULAR_INDEX_H_

#include <cstdlib>
#include <fstream>
#include <iostream>

/*!
This class implements modular arithmetics on index values of a specified index in a defined interval [min, max] .
First and last listed values are included.

Only incremental operation has been implemented like operators +, -, ++, -- . 

A circular index must incremented or decremented by its methods, without making operations directly on .i attribute, that has been
left public to speed up the variable reading!!!

Pay attention that no error is returned if you try to assign to a circular_index a value outside [min, max] interval,
because it is forced, using periodicity criteria, into the [min, max] interval.

circular_index is unsigned safe (INT = unsigned int).

Initialization (init(INT _max, INT _min = 0, INT val = 0)) is not safe,
choose min<=val<=max and if you assign INT = unsigned and do not try to initialize min < 0.

Comparison operators has been implemented, but use <, >, <=, >= with extream care because theoretically a circular_index
cannot be grater than another due to the periodicity condition.
*/
template <typename INT>
class circular_index
{
 public:
  
  circular_index(INT _max = 0, INT _min = 0) { this->init(_max, _min); }
  circular_index(circular_index &ic) { this->copy(ic); }
  
  ~circular_index() {}
  
  void init(INT _max, INT _min = 0, INT val = 0)
  {
/*   if(_min == _max) i = val;
   if(_min > _max)
    {
     std::cerr <<"invalid circular_index::init inputs   min = "<<_min<<"  and max = "<<_max<<std::endl;
     _min = 0; _max = 0; val = 0;
    }
    
   if(val > max)
    {
     std::cerr <<"invalid circular_index::init inputs   index = "<<val<<"  and max = "<<_max<<std::endl;
     val = max;
    }
    
   if(val < min)
    {
     std::cerr <<"invalid circular_index::init inputs   index = "<<val<<"  and min = "<<_min<<std::endl;
     val = min;
    }
*/    
   min = _min;
   max = _max;
   dim = max - min + 1;
   i = val;
   
  }
  
  circular_index& copy(circular_index& ic)
  {
   this->dim = ic.dim;
   this->max = ic.max;
   this->min = ic.min;
   this->i   = ic.i;
   return (*this);
  }
  
  INT dim;
  INT min;
  INT max;
  INT i;
  
  inline INT& operator()() {return i;}
  
  INT operator++()
   {
  
    if(i==max) i = min;
    else           ++i;
    
    return i;
   }

  INT operator--()
   {
  
    if(i==min) i = max;
    else           --i;
    
    return i;
   }

/*  friend INT& operator++(circular_index &index)
   {
    if(index.i == index.max) index.i = index.min;
    else               ++index.i;
    
    return index.i;
   }
   
  friend INT& operator--(circular_index &index)
   {
    if(index.i == index.min) index.i = index.max;
    else                      --index.i;
    
    return index.i;
   }
  */ 
  template<typename INT2>
  bool operator== (INT2 i2) {return i == i2;}
  
  template<typename INT2>
  bool operator!= (INT2 i2) {return i != i2;}
  
  template<typename INT2>
  bool operator<  (INT2 i2) {return i <  i2;}
  
  template<typename INT2>
  bool operator<= (INT2 i2) {return i <= i2;}
  
  template<typename INT2>
  bool operator>  (INT2 i2) {return i >  i2;}
  
  template<typename INT2>
  bool operator>= (INT2 i2) {return i >= i2;}
  
  template<typename INT2>
  bool operator== (circular_index<INT2>& ic) {return this->i == ic.i;}
  
  template<typename INT2>
  bool operator!= (circular_index<INT2>& ic) {return this->i != ic.i;}
  
  template<typename INT2>
  bool operator<  (circular_index<INT2>& ic) {return this->i <  ic.i;}
  
  template<typename INT2>
  bool operator<= (circular_index<INT2>& ic) {return this->i <= ic.i;}
  
  template<typename INT2>
  bool operator>  (circular_index<INT2>& ic) {return this->i >  ic.i;}
  
  template<typename INT2>
  bool operator>= (circular_index<INT2>& ic) {return this->i >= ic.i;}
  
  
  
  
  template<typename INT2>
  inline INT& operator=(INT2 i2)
  {
   if( ((signed long)(min) <= i2) && (i2 <= (signed long)(max) ) )
    {
     i = i2;
    }
   else
    {
     div_t d;
     INT ii;
     
     if(i2 > (signed long)(min))
      {
       ii = i2 - min;
       d = div((int)ii,(int)dim);
       i = d.rem + min;
      }
     else //(ii<0)
      {
       long iii = i2 - min;
       ii = abs(iii);
       d = div((int)ii,(int)dim);
       i = dim - d.rem;
      }
    }
    
   return i;
  }
  
  template<typename INT2>
  inline INT& operator=(circular_index<INT2>& ic)
  {
   return this->operator=(ic.i);
  }


  
  
  
  
  template<typename INT2>
  inline INT operator+(INT2 i2)
  {
   INT2 ii;
   INT i_o;
   
   if(i2<0)
    {
     ii = abs(i2);
     this->operator-(ii);
    }
   else
    {
     ii = i2;
    }
   
   if(ii>dim)
    {
     div_t d = div(ii,dim);
     ii = d.rem;
    }
    
   i_o = i + ii;
   if(i_o > max) i_o = i_o - dim;

   return i_o;
  }
  
//  template<typename INT2>
//  inline INT operator+(INT2 i2)
//  {
//   return this->operator+(i2);
//  }
  
  template<typename INT2>
  inline INT operator+(circular_index<INT2>& ic)
  { 
   return this->operator+(ic.i);
  }
  
  template<typename INT2>
  inline INT operator-(INT2 i2)
  {
   INT2 ii;
   INT i_o;
   
   if(i2<0)
    {
     ii = abs(i2);
     return this->operator+(ii);
    }
   else
    {
     ii = i2;
    }
   
   if(ii>dim)
    {
     div_t d = div(ii,dim);
     ii = d.rem;
    }
    
   if(i < ii + min)
    i_o = (i + dim) - ii;
   else
    i_o = i - ii;

   return i_o;
  }
  
//  template<typename INT2>
//  inline INT operator-(INT2 i2)
//  {
//   return this->operator-(i2);
//  }
  
  template<typename INT2>
  inline INT operator-(circular_index<INT2>& ic)
  { 
   return this->operator-(ic.i);
  }
  
 template <typename INT2>
 inline friend INT& operator+( INT2 ii, circular_index &i_c)
 {
  return (ii + i_c);
 }
 
 template <typename INT2>
 inline friend INT& operator-( INT2 ii, circular_index &i_c)
 {
  return (ii - i_c);
 }
 
 friend std::ostream& operator<<(std::ostream& out, circular_index &ii)
 {
  out <<ii.i;
  return out;
 }
  
 friend std::istream& operator<<(std::istream& in, circular_index &ii)
 {
  long iii;
  in >>iii;
  ii = iii;
  return in;
 }
  
 private:
    
 
};

#endif

