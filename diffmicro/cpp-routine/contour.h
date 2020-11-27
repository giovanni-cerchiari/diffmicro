
/*
Copyright: Giovanni Cerchiari
date: 03/12/2015
e-mail: giovanni.cerchiari@gmail.com
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

#ifndef _CONTOUR_H_
#define _CONTOUR_H_

#include <iostream>
#include <list>
#include <vector>
#include <algorithm>

#include "prep_matrix.h"
#include "my_math.h"


/*!
This set of funtions are inteded to find contour courve in a contour-plot like behaviour
*/

/*!
This function might be used togheter with equipoints.
It makes an estimation about the deviation of the points that could constitue a line
of equal-value points. In practice it is worthless to ask for the same value, but better to similar value.
This function attempts a definition of similarity.
The idea is that in a courour plot it is possible to draw a line.
Therefore considering a matrix around a coutour point the deviation is estimated as max and min.
The deviation to be considered is the one that allows to pick up approximately the points
that would allow a line to go through the matrix.

dimx, dimy -> dimension of the matrix of values
mat -> function matrix
val -> value of the of the equivalent points
similarity -> calculated deviation of similarity
windowside -> dimension of the submatrix where the alghoritm will run

*/
template<typename UINT, typename FLOAT>
void equipoints_estimate_deviation(UINT dimx, UINT dimy, FLOAT *mat, FLOAT val, FLOAT &similarity, UINT windowside = 7)
{
	UINT i,j;
	UINT x,y, dim;
	UINT xbegin, xend, ybegin, yend;
	FLOAT min,max;
	FLOAT most_similar;
	std::vector<FLOAT> euristic;
	FLOAT *row;
	
	UINT windowborder = (windowside-1)/2;
	euristic.resize(windowside*windowside);
	
	
	xbegin = windowborder + 1; xend = dimx-(windowborder + 1);
	ybegin = windowborder + 1; yend = dimy-(windowborder + 1);
	
	dim = dimx*dimy;
	most_similar = mat[ybegin*dimx+xbegin];
	similarity = fabs(mat[ybegin*dimx+xbegin]-val);
	i=xbegin; j=ybegin;
	for(y=ybegin; y<yend; ++y)
	{
		row = &(mat[y*dimx]);
		for(x=xbegin; x<xend; ++x)
		{
			if(similarity > fabs(row[x]-val))
			{
				similarity = fabs(row[x]-val);
				most_similar = row[x];
				i = x;	j = y;
			}
		}
	}
	
	i -= windowborder;
	j -= windowborder;
	
	for(y=0; y<windowside; ++y)
		for(x=0; x<windowside; ++x)
			euristic[y*windowside+x]=mat[(y+j)*dimx+(x+i)];
	
	sort(euristic.begin(), euristic.end());
	
	max = euristic[0];
	min = euristic[0];
	for(i=1; i<2*windowside; ++i)
	{
		if(euristic[i]>max)max = euristic[i];
		if(euristic[i]<min)min = euristic[i];
	}
	
	if(fabs(max-val)>fabs(val-min)) similarity = fabs(max-val);
	else                            similarity = fabs(min-val);
}

/*!
Find in a matrix all the position where the value is similar to val
Two values v1 and v2 are similar if deviation>=fabs(v1-v2)
dimx, dimy -> matrix dimension
mat -> matrix
val -> value of the points to be found
dimpoints -> number of found point
point -> found point indicated as (x[0], y[0], x[1], y[1] ...)
         if NULL only dimpoints is returned.
deviation -> see above description. Can be stimated with equipoints_estimate_deviation
*/
template<typename UINT, typename FLOAT>
void equipoints(UINT dimx, UINT dimy, FLOAT *mat, FLOAT val, UINT &dimpoints, UINT *point, FLOAT deviation = -1)
{
	UINT x,y;
	bool end = false;
	FLOAT *row;
	UINT dim;
	
	dim = dimx*dimy;
	
	if(0>deviation)
		equipoints_estimate_deviation<UINT, FLOAT>(dimx, dimy, mat, val, deviation, 3);
	
//	std::cout <<"val = "<<val<<"\tdeviation = "<<deviation<<std::endl;
	
	dimpoints = 0;
	if(NULL == point)
	{
		for(x=0;x<dim;++x)
			if(deviation>=fabs(mat[x]-val))
				++dimpoints;
//		std::cout <<"dimpoints = "<<dimpoints<<std::endl;
	}
	else
	{
		for(y=0; y<dimy; ++y)
		{
			row = &(mat[y*dimx]);
			for(x=0;x<dimx;++x)
			{
				if(deviation>=fabs(row[x]-val))
				{
					point[2*dimpoints]   = x;
					point[2*dimpoints+1] = y;
					++dimpoints;
				}
			}
		}
	}
	
	
}


/*!
Uses equipoints function to elaborate a matrix with 0 where no points are matching and 1
where equivalent points are found.
dimx, dimy -> matri dimension
mat -> matrix
val -> value to look for
mask -> matrix of the same dimension of to be used as output
*/
template<typename UINT, typename FLOAT, typename MASKTYPE>
void equipoints_mask(UINT dimx, UINT dimy, FLOAT *mat, FLOAT val, MASKTYPE *out, FLOAT deviation = -1)
{
 UINT i, dimpoint, dim;
 UINT *point;

 dimpoint = 0;
 dim = dimx*dimy;
 point = new UINT[2*dim];
 
 //point = NULL;
 equipoints(dimx, dimy, mat, val, dimpoint, point, deviation); 
 
 for(i=0; i<dim; ++i) out[i]=(MASKTYPE)(0.0);
 for(i=0; i<dimpoint; ++i) out[point[2*i+1]*dimx+point[2*i]] = (MASKTYPE)(1);
 
 delete[] point;
}


#endif

