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



#ifndef _ANGULAR_INTEGRAL_AVERAGE_H_
#define _ANGULAR_INTEGRAL_AVERAGE_H_

#include "global_define.h"
#include "histogram.h"
#include "prep_matrix.h"

template<typename FLOAT>
void angular_integral_average(INDEX dimx, INDEX dimy, FLOAT *mat, FLOAT *center, INDEX dimout, FLOAT stepout, FLOAT *integral, FLOAT *weight, FLOAT *average, FLOAT *average_std, FLOAT *mask=NULL, INDEX subdivision = 8)
{
	INDEX i,j, i2, j2, ii;
	INDEX dimx2, dimy2;
	FLOAT *r;
	FLOAT *row, *rowmask;
	INDEX row2i;
	FLOAT *mat2, *mask2(NULL);
	INDEX pixeli;
	FLOAT val, valmask;
	FLOAT step,x,y;
	FLOAT y1, y2;
	FLOAT rr;
	SIGNED_INDEX rru, rrd;
	FLOAT ifl,jfl;
	FLOAT zero = (FLOAT)(0);
	INDEX *one;
	FLOAT *avg_img;
	FLOAT normalization;
	histogram<FLOAT, FLOAT> hist;
	histogram<FLOAT, FLOAT> hist_std;
	histogram<FLOAT, INDEX> histi;
	histogram<FLOAT, FLOAT> histmask;
	FLOAT *hist_ptr;
	FLOAT *hist_std_ptr;
	INDEX *histi_ptr;
	FLOAT *histmask_ptr(NULL);
	bool flg_delete_mask = false;
	
	if (NULL == mask)
	{
		ii = dimx * dimy;
		mask = new double[ii];
		for (i = 0; i < ii; ++i) mask[i] = (FLOAT)(1.0);
		flg_delete_mask = true;
	}

	normalization = (FLOAT)(1.0)/(FLOAT)(subdivision*subdivision);
	
	dimx2 = subdivision*dimx;
	dimy2 = subdivision*dimy;
	
	ii = dimx2*dimy2;
	r = new FLOAT[ii];
	mat2 = new FLOAT[ii];
	mask2 = new FLOAT[ii];

	one = new INDEX[ii];
	avg_img = new FLOAT[ii];
	
	step = (FLOAT)(1.0)/(FLOAT)(subdivision);
	
	for(j=0; j<dimy; ++j)
	{
		row = &(mat[j*dimx]);
		rowmask = &(mask[j*dimx]);
		jfl = (FLOAT)(j);
		for(i=0;i<dimx;++i)
		{
			ifl = (FLOAT)(i);
			val = row[i];
			valmask = rowmask[i];
			pixeli =(j*dimx2+i)*subdivision;
			for(j2=0;j2<subdivision;++j2)
			{
				y = (jfl + step * (FLOAT)(j2))-center[1];
				y*=y;
				row2i = pixeli + j2*dimx2;
				for(i2=0;i2<subdivision;++i2)
				{
					ii =row2i + i2; 
					x = (ifl + step * (FLOAT)(i2))-center[0];
					mat2[ii] = val*valmask;
					mask2[ii] = valmask;

					one[ii] = 1;
					r[ii] = sqrt(y+x*x);
				}
			}
		}
	}
		
//	std::cerr <<"--------------------------"<<std::endl;
//	print_mat(std::cout, dimx2, dimy2, mat2);
	//std::cerr <<"--------------------------"<<std::endl;
	//print_mat(std::cout, dimx2, dimy2, r);
//std::cerr <<"--------------------------"<<std::endl;
	
	y = (FLOAT)(dimy);
	x = (FLOAT)(dimx);
//std::cerr <<"ciao"<<std::endl;
	hist.init(1, &dimout, &zero, &stepout);
	histmask.init(1, &dimout, &zero, &stepout);
	histi.init(1, &dimout, &zero, &stepout);
	hist_std.init(1, &dimout, &zero, &stepout);

	hist.update(dimx2*dimy2, r, mat2);
	hist_ptr = hist.freq.ptr();
	
	histi.update(dimx2*dimy2, r, one);
	histi_ptr = histi.freq.ptr();

	histmask.update(dimx2*dimy2, r, mask2);
	histmask_ptr = histmask.freq.ptr();

	for (i = 0; i < dimx2*dimy2; ++i) mat2[i] *= mat2[i];

	hist_std.update(dimx2*dimy2, r, mat2);
	hist_std_ptr = hist_std.freq.ptr();

	ii = hist.freq.dim()[0];
	for (i = 0; i < dimout; ++i)
	{
		if (i < ii)
		{
			integral[i] = hist_ptr[i] * normalization;
			if (NULL == mask) weight[i] = (FLOAT)(histi_ptr[i]);
			else              weight[i] = histmask_ptr[i];
			if (0 != histi_ptr[i] && 0.0000001 < fabs(weight[i]))
			{
				average[i] = hist_ptr[i] / weight[i];
				average_std[i] = sqrt(hist_std_ptr[i] / weight[i] - average[i] * average[i]);
				std::cerr << "[" << i << "] -> avg = " << average[i] << "\tstd = " << average_std[i] << std::endl;
			}
			else
			{
				average[i] = (FLOAT)(0.0);
				average_std[i] = (FLOAT)(0.0);
			}
		}
		else
		{
			integral[i] = (FLOAT)(0.0);
			average[i] = (FLOAT)(0.0);
		}
	}

	


	delete[] r;
	delete[] mat2;
	delete[] one;
	delete[] mask2;
	if (true == flg_delete_mask) delete[] mask;
}




#endif
