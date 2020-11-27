
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



#ifndef _PREP_MATRIX_H_
#define _PREP_MATRIX_H_

#include "circular_index.h"
#include <iostream>
#include <string>

/*!This function print a matrix stored into an array on an output stream*/
template<typename UINT, typename ELE>
void print_mat(std::ostream &out, UINT dimx, UINT dimy, ELE mat[])
{
unsigned int i,j;	
for(j=0;j<dimy;++j)
{
	i=0;
	out <<mat[j*dimx+i];
	for(i=1;i<dimx;++i)
	{
		out <<"\t"<<mat[j*dimx+i];
	}
	out <<std::endl;
}
}


void edge_vet_init_indices(unsigned int dimin, unsigned int dimout,
                           unsigned int &start_in, unsigned int &start_out,
                           circular_index<unsigned int> &index_in,
                           circular_index<unsigned int> &index_out,
                           unsigned int &dimmim);

/*!
 add or delete a cross of elements in or from the middle of the matrix
*/
template<typename FLOAT>
void edge_matrix(unsigned int dimrowin,  unsigned int dimcolin,  FLOAT *in,
                 unsigned int dimrowout, unsigned int dimcolout, FLOAT *out)
{
	unsigned int i,j;
	unsigned int dimrow, dimcol;
	FLOAT *out_row, *in_row;

	circular_index<unsigned int> i_in;
	circular_index<unsigned int> i_out;
	unsigned int i_in_start, i_out_start;
	unsigned int j_in_start, j_out_start;
	circular_index<unsigned int> j_in;
	circular_index<unsigned int> j_out;

	edge_vet_init_indices(dimrowin, dimrowout, j_in_start, j_out_start, j_in, j_out, dimrow);
	edge_vet_init_indices(dimcolin, dimcolout, i_in_start, i_out_start, i_in, i_out, dimcol);

	for(j=0; j<dimrow; ++j , ++j_in,	++j_out)
		{
			out_row = out + j_out.i * dimcolout;
			in_row  = in  + j_in.i  * dimcolin;	
			
			i_out.i = i_out_start;	i_in.i = i_in_start;
			for(i=0; i<dimcol; ++i, ++i_in,	++i_out)	*(out_row + i_out.i) = *(in_row + i_in.i);
		}
		
}

/*!
matrix trasposition.Safe self-transposition
*/
template<typename ELE>
inline void trasp_mat(unsigned int dimrow, unsigned int dimcol, ELE *in, ELE *out)
{
	unsigned int i,j,jj,ii;
	ELE *row_in, *col_in;
	ELE *row_out, *col_out;
	ELE temp;
	
	for(j=0; j<dimrow; ++j)
		{
			jj = j * dimcol;
			row_in = in + jj;
			col_in = in + j;
			col_out = out + jj;
			row_out = out + j;
			for(i=j+1; i<dimcol; ++i)
				{
					ii = i * dimrow;
					temp = *(row_in + i);
					*(col_out + i) = *(col_in + ii);
					*(row_out + ii) = temp;
				}
		}
}

/*! submatrix copy with periodiacally boundary condition */
template<typename INT, typename FLOATIN, typename FLOATOUT>
void copy_submatrix_periodical(INT dimxin, INT dimyin, FLOATIN matin[], INT startx, INT starty, INT dimxout, INT dimyout, FLOATOUT matout[])
{
	UINT i, j;
	circular_index<INT> iin, jin;
	FLOATOUT *row_out;
	FLOATIN *row_in;

	iin.init(dimxin - 1);
	jin.init(dimyin - 1);

	for (j = 0, jin = starty; j < dimyout; ++j, ++jin)
	{
		row_out = &(matout[j*dimxout]);
		row_in = &(matin[jin()*dimxin]);

		for (i = 0, iin = startx; i < dimxout; ++i, ++iin)
		{
			row_out[i] = row_in[iin()];
		}
	}

}

#endif
