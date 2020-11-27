
/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 2010
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

#ifndef _TENSOR_H_
#define _TENSOR_H_

// this is the include that allows to write functions with an indefinite
// number of argments
#include <cstdarg>
// how to use
// find a way to know how many arguments there will be
// define a
//    va_list it
// variables to iterate throught the arguments
// then use this stament
//    x = va_arg(ap, int); 
// iterate ap throught the arguments and assign the arguments
// casted to int in the varible x.
// do not forget to
//    va_end(it)

#include "global_define.h"
#include <cstdlib>
#include <iostream>
//#include <fstream>
#include <string>
//#include "prep_vet.h"

/*!
This class implements a tensor of generic type for memory managment. Particularly useful for high-performance computing or to operate
directly inside the memory is the possibility to use high-level function for allocation and delete, but at the same time to get the
pointer to memory when it is needed.
*/
template<typename FLOAT>
class tensor
{

	public:

  tensor()
   {
	m_dimspace = 0;
	m_size = 0;
	m_ptr = NULL; 
	m_dim = NULL;
    m_jump = NULL;
   }
   
  tensor(INDEX space_dimension, ...)
   {

	m_dimspace = 0;
	m_size = 0;
	m_ptr = NULL; 
	m_dim = NULL;
    m_jump = NULL;

    va_list ap;
    va_start(ap, space_dimension);
				this->m_resize_valist(space_dimension, ap);
				va_end(ap);
   }

	//position of the arguments disentagle the overload
	tensor(INDEX *dim, INDEX space_dimension)
   {   
	m_ptr = NULL; 
	m_dim = NULL;
    m_jump = NULL;
      
    this->m_resize(space_dimension, dim);
    
   }

  virtual void resize(INDEX space_dimension = 1, ...)
   {
    va_list ap;
    va_start(ap, space_dimension);
				this->m_resize_valist(space_dimension, ap);
				va_end(ap);
   }

  virtual void resize(INDEX space_dimension, INDEX *dim)
   {
    if(space_dimension == 0)
     {
      space_dimension = 1;
      std::cerr <<"error dimension==0 in tensor::tensor"<<std::endl;
     }
    this->m_resize(space_dimension, dim);
   }


  virtual void clear()
			{
				m_size = 0;
				m_dimspace = 0;
				if(m_ptr  != NULL)
				 {
				  delete[] m_ptr;
				  m_ptr = NULL;
				 }
    if(m_dim  != NULL)
					{
					 delete[] m_dim;
					 m_dim = NULL;
					}
    if(m_jump != NULL)
					{
					 delete[] m_jump;
					 m_jump = NULL;
					}
			}
				
  //!destructor 
  ~tensor()
   {
				this->clear();
   }
  
  
  virtual FLOAT operator()(INDEX index, ...) const
   {
    //no control is made if the index is really existing
   
    INDEX ele = 0;
    INDEX i;
				va_list ap;
    
	  // jumping to the proper location
				ele = index * m_jump[0];
				va_start(ap, index); 
				for (i = 1; i < m_dimspace; ++i) ele += va_arg(ap, INDEX) * m_jump[i];
				va_end(ap);
    
    return m_ptr[ele];
   }
   
  virtual FLOAT &operator()(INDEX index, ...)
   {
    //no control is made if the index is really existing
   
    INDEX ele = 0;
    INDEX i;
				va_list ap;
    
	  // jumping to the proper location
				ele = index * this->m_jump[0];
				va_start(ap, index); 
				for (i = 1; i < m_dimspace; ++i) ele += va_arg(ap, INDEX) * this->m_jump[i];
				va_end(ap);
    
    return m_ptr[ele];
   }
  
		  virtual FLOAT operator()(INDEX *index) const
   {
    //no control is made if the index is really existing
   
    INDEX ele = 0;
    INDEX i;
    
	  // jumping to the proper location
				ele = index[0] * m_jump[0];

				for (i = 1; i < m_dimspace; ++i) ele += index[i] * m_jump[i];
    
    return m_ptr[ele];
   }
   
  virtual FLOAT &operator()(INDEX *index)
   {
    //no control is made if the index is really existing
   
    INDEX ele = 0;
    INDEX i;
    
	  // jumping to the proper location
				ele = index[0] * m_jump[0];

				for (i = 1; i < m_dimspace; ++i) ele += index[i] * m_jump[i];
    
    return m_ptr[ele];
   }


  virtual tensor& operator=(tensor &t)
			{
				
				this->clear();
				this->m_ptr  = new FLOAT[t.m_size];
    this->m_dim  = new INDEX[t.m_dimspace];
    this->m_jump = new INDEX[t.m_dimspace];
    
    this->m_size = t.m_size;
    this->m_dimspace = t.m_dimspace;
    
				memcpy(this->m_dim, t.m_dim, this->m_dimspace * sizeof(INDEX));
				memcpy(this->m_jump, t.m_jump, this->m_dimspace * sizeof(INDEX));

    memcpy(this->m_ptr, t.m_ptr, this->m_size * sizeof(FLOAT));
    
				return *this;
			}
	   
   void save_file(std::string file_name)
				{


				}
				
			virtual bool load_file(std::string file_name)
				{

				 return true;
				}

  //----------------------- 
  // use these function to optimize your program getting protected datas... but use with care!!! 
  FLOAT*        ptr()      {return m_ptr;}
  INDEX* dim()      {return m_dim;}
  INDEX* jump()     {return m_jump;}
  //-----------------------------
  INDEX  dimspace() {return m_dimspace;}
  INDEX  size()     {return m_size;}
   
   
 protected:
 
		// memory pointer to the tensor datas
  FLOAT *m_ptr;
		// subdimensions of the spaced spanned by the various indices
  INDEX *m_dim;
		// jump step for different indices
  INDEX *m_jump;
		// number of different indices to be used to browse the tensor
  INDEX m_dimspace;
		// total number of elements stored in m_ptr
  INDEX m_size;



		void m_resize_valist(INDEX &space_dimension, va_list &ap)
		{
			INDEX i,arg;
			INDEX *dim;
			// dimension check
    if(space_dimension == 0)
     {
      space_dimension = 1;
      std::cerr <<"error dimension==0 in tensor::tensor"<<std::endl;
     }
				dim = new INDEX[space_dimension];
				for (i = 0; i < space_dimension; ++i)
				 {
				  arg = va_arg(ap, INDEX);
				  dim[i] = arg;
				 }
			this->m_resize(space_dimension, dim);
			delete[] dim;
		}

  void m_resize(INDEX &space_dimension, INDEX *dim)
			{
    
    // definitions
    if(m_dimspace != space_dimension)
					{
						m_dimspace = space_dimension;
						if(m_dim!=NULL) delete[] m_dim;
						m_dim  = new INDEX[m_dimspace];
						if(m_jump!=NULL) delete[] m_dim;
						m_jump = new INDEX[m_dimspace];
					}
    
				INDEX i, size;
		 
		  // reading arguments and assign the proper dimension
				size = 1;
				for (i = 0; i < m_dimspace; ++i)
				 {
				  m_dim[i] = dim[i];
				  size = size * m_dim[i];
				 }

    // calculus of jump array (see element reading process)
    
    m_jump[m_dimspace - 1] = 1;
    if(m_dimspace > 1)
     {
      m_jump[m_dimspace - 2] = m_dim[m_dimspace - 1];
		    for (i = m_dimspace-2; i>0 ; --i) m_jump[i - 1] = m_dim[i] * m_jump[i];
		   }

				if(size != m_size)
					{
						m_size = size;
						if(m_ptr != NULL) delete[] m_ptr;
						m_ptr = new FLOAT[m_size];
					}
			}
 

  
};

#endif

