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



#ifndef _PREP_VET_H_
#define _PREP_VET_H_

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <string>
#include <algorithm>

#include "global_define.h"

/*!
This function copy an array into another
*/
template<typename UINT, typename TYPEIN, typename TYPEOUT>
void copy_a(UINT dim, TYPEIN *in, TYPEOUT *out)
{
	UINT i;
	for(i=0; i<dim; ++i) out[i] = (TYPEOUT)(in[i]);
}

/*!
This function prints a std::vector to video using std::cerr
*/
template<typename ELE>
void cerr_vet(std::string name, std::vector<ELE> &vet)
{
 unsigned int i, dim;
 dim = vet.size();
 for(i=0; i<dim; ++i) std::cerr <<name<<"["<<i<<"] = "<<vet[i]<<std::endl;
}

/*!
This function prints a matrix to video using std::cerr
*/
template<typename ELE>
void cerr_mat(unsigned int dimrow, unsigned int dimcol, ELE *mat)
{
	unsigned int i,j;
	for(j=0; j<dimrow; ++j)
		{
			std::cerr <<mat[dimcol * j];
			for(i=1; i<dimcol; ++i)
				{
					std::cerr <<"\t"<<mat[dimcol * j + i];
				}
			std::cerr <<std::endl;
		}
}

/*!
This function prints a matrix to video using a generic std::ostream
*/
template<typename ELE>
void fout_mat(std::ostream &fout, unsigned int dimrow, unsigned int dimcol, ELE *mat)
{
	unsigned int i,j;
	for(j=0; j<dimrow; ++j)
		{
			fout <<*(mat + dimcol * j);
			for(i=1; i<dimcol; ++i)
				{
					fout <<"\t"<<*(mat + dimcol * j + i);
				}
			fout <<std::endl;
		}
}

/*!
This function prints an array to video using std::cerr
*/
template<typename INT, typename ELE>
void cerr_ary(std::string name, INT dim, ELE *vet)
{
 unsigned int i;
 for(i=0; i<dim; ++i) std::cerr <<name<<"["<<i<<"] = "<<*(vet + i)<<std::endl;
}


/*!
This function prints a vector to video using std::cerr
*/
template<typename INT, typename ELE>
void fout_ary(std::ostream &fout, std::string name, INT dim, ELE vet[])
{
 unsigned int i;
 for(i=0; i<dim; ++i) fout <<name<<"["<<i<<"] = "<<vet[i]<<std::endl;
}

/*!This function returns the element value having maximum frequancy in vet array*/
template <typename INT, typename FLOAT>
inline FLOAT moda(INT dimvet, FLOAT *vet)
{
 INT i,j, dim_freq;
 INT max, min, pos_max, pos_min;
 FLOAT *val, vall;
 FLOAT new_val;
 INT *freq;
 bool flg_found;
 
	val = new FLOAT[dimvet];
	freq = new INT[dimvet];

 for(i=0; i<dimvet; ++i)
  {
   freq[i] = 0.;
  }
 dim_freq = 0;
 for(i=0; i<dimvet; ++i)
  {
   new_val = *(vet + i);
   flg_found = false;
   for(j=0; j<dim_freq; ++j)
    {
     if(fabs(val[j] - new_val) < (0.0000000001 * val[j]))
      {
       ++freq[j];
       flg_found = true;
       break;
      }
    }
   if(flg_found == false)
    {
     val[dim_freq] = new_val;
     ++freq[dim_freq];
     ++dim_freq;
    }
  }
 maxminary(dim_freq, freq, max, pos_max, min, pos_min);
 
	vall = val[pos_max];
	delete[] val;
	delete[] freq;

 return (vall);
}


/*!
This function evaluates maximum and minimum elements and their index-positions into a vector
*/
template <typename INT, typename ELEVET>
void
maxminvet(std::vector<ELEVET> &vet, ELEVET &max, INT &pos_max, ELEVET &min, INT &pos_min)
{
 // maxminvet finds maximum and minimum values and their positions.
 INT i, beginvet, dimvet;
 beginvet = 0;
 dimvet = vet.size();
 max = vet[beginvet];
 min = vet[beginvet];
 pos_max = beginvet;
 pos_min = beginvet;
 for(i=beginvet; i<dimvet; ++i)
 {
  if (vet[i] > max) 
  {
	max = vet[i];
	pos_max = i;
  }
  if (vet[i] < min)
  {
	min = vet[i];
	pos_min = i;
  }
 }
}//void maxminvet


/*!
This function evaluates maximum and minimum elements and their index-positions into an array
*/
template <typename INT, typename ELEVET>
void
maxminary(INT dimvet, ELEVET vet[], ELEVET &max, INT &pos_max, ELEVET &min, INT &pos_min)
{
	if (0 == dimvet) return;
 // maxminvet finds maximum and minimum values and their positions.
 INT i, beginvet;
 beginvet = 0;
 max = vet[beginvet];
 min = vet[beginvet];
 pos_max = beginvet;
 pos_min = beginvet;
 for(i=beginvet; i<dimvet; ++i)
 {
  if (vet[i] > max) 
  {
			max = vet[i];
			pos_max = i;
  }
  if (vet[i] < min)
  {
			min = vet[i];
			pos_min = i;
  }
 }
}//void maxminvet

/*!
This function implements the binary search alghoritm into an array.
v is supposed to be in true order
return index is the index of the last element in v which is
ele <= val
if val<=v[0] zero is returned
*/
template <typename FLOAT>
unsigned int binary_search(unsigned int size, FLOAT *v, FLOAT val)
{
 int sign;

 if(val <= *v) return 0;
 size = size-1;
 if(*(v+size) <= val) return size; 
 size /= 2;
 while(size != 1)
 {
		if(*(v + size) < val) size += size/2;
		else                  size -= size/2;
 }
 
 while( val < *(v + size)) --size;
 while( *(v + size) <= val) ++size;
 
 return (size-1);
 
}

/*! This function copies the second vector as trailing to the first. It is like the push_back function for more than one element. */
template<typename ELE>
void append_vet(std::vector<ELE> &vet, std::vector<ELE> &app)
{
 unsigned int i, dimold, dimnew;
 
 dimold = vet.size();
 dimnew = dimold + app.size();
 vet.resize(dimnew);
 for(i=dimold; i<dimnew; ++i) vet[i] = app[i-dimold]; 
}

/*!
This function deletes the single element of index ii compacting the vet vector
*/
template<typename ELE, typename INT>
bool del_vet_element(std::vector<ELE> &vet, INT ii)
{
 INT kk;
 INT dimvet = vet.size();
 if(ii!=dimvet-1)
 {
  for(kk=ii+1; kk<dimvet; ++kk) vet[kk-1]=vet[kk];
 }
  vet.pop_back();
  
  if(ii==vet.size()) return (false);
  else               return (true);
}

/*!
It deletes the single element of index ii compacting the vet array
*/
template<typename ELE, typename INT>
bool del_ary_element(INT &dimvet, ELE *vet, INT ii)
{
 INT kk;
 if(ii!=dimvet-1)
 {
  for(kk=ii+1; kk<dimvet; ++kk) vet[kk-1] = vet[kk];
 }
 --dimvet;
   
 if(ii==dimvet) return (false);
 else           return (true);
}

/*!
This function is used in del_vet_elements. The function prepares a vector of indices with the following properties:
- sorted ascending
- no multiple indices
- 0 <= index < dim_extern_vet

*/
INDEX prep_vet_del(std::vector<INDEX> &vet, INDEX dim_extern_vet);

/*!
The aim of this function is to generate a look-up-table (lut).
This function receives a vector of indices of elements that should be deleted (del) in an external vector.
This external vector is not passed to the function and has dimension dimold.
Given these two inputs, the function elaborates lut having

Output:
- lut[i] = new_address_for_i_element
- lut[i] = -1  (element to be deleted)

uses: prep_vet_del
*/
void look_up_table_del(INDEX dimold, std::vector<INDEX> &del, std::vector<INDEX> &lut);


/*!
This function deletes in the vector vet all the elements whose index are in the vector ele_to_del.
Working process:
- prepearing ele_to_del sorting indices and deleting repeated indices
- preparing new address in the vector for the elements that survives
- copy the elements that must survive compacting the vector

uses: look_up_table_del
*/
template<typename ELE>
void del_vet_elements(std::vector<ELE> &vet, std::vector<long> &ele_to_del)
{
 long i,j,dimvet;
 std::vector<long> lut;

 look_up_table_del(vet.size(), ele_to_del, lut);

 if(ele_to_del.size() == 0) return;
 dimvet = vet.size();
 for(i=0; i<dimvet; ++i)
  {
   j = lut[i];
   if(j>=0) vet[j] = vet[i];
  }

 dimvet = ele_to_del.size();
 for(i=0; i<dimvet; ++i) vet.pop_back();
}


/*!
This function shuffles in seq vector the elements of vet vector in random order. Pay attention not to use vet
by-ref, because vet vector is going to be modified
*/
template <typename ELE>
void random_sequence_vet(std::vector<ELE> vet, std::vector<ELE> &seq)
{
 unsigned int i,j,dim,dimm1,ind;
 i=0;
 dim = vet.size();
 if(dim==0) return;
 seq.resize(dim);
 dimm1 = dim - 1;
 for(j=0; j<dim; ++j)
  {
   ind = (unsigned int)((double)(dimm1) * (double)(rand()/((double)(RAND_MAX))));
   seq[i] = vet[ind]; ++i;
   del_vet_element(vet, ind); --dimm1;
  }
}

/*!
This function shuffles in seq array the elements of vet array in random order. Vet array
will be modified also
*/
template <typename INT, typename ELE>
void random_sequence_ary(INT dim, ELE *vet, ELE *seq)
{
 INT i,j,dim2,ind;
 i=0;
 if(dim==0) return;
 dim2 = dim;
 for(j=0; j<dim; ++j)
 {
  ind = (unsigned int)((double)(dim2-1) * ((double)(rand())/(double)(RAND_MAX)));
  *(seq + i) = *(vet + ind); ++i;
  del_ary_element(dim2, vet, ind);
 }
}

#endif

