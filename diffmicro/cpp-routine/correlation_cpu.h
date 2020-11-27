
/*
Copyright: Giovanni Cerchiari
date: Feb 2019
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


#ifndef _CORRELATION_CPU_H_
#define _CORRELATION_CPU_H_

#include "my_math.h"

/*! This function is needed by one point correlation to evaluate antialising correction*/
template<typename FLOAT>
void calculate_linear_overlap_weight(int &dim, FLOAT shift, int shift_index[], FLOAT weight[])
{

	int i;
	FLOAT shift_pure;

	shift_pure = shift - std::floor(shift + (FLOAT)(0.5));

	if (0.0000001 < std::fabs(shift_pure))
	{
		dim = 2;

		for (i = 0; i < 2; ++i)
		{
			shift_index[i] = (int)(std::floor(shift_pure)) + i;
			//			std::cout << i << " -> " << shift_index[i] << std::endl;
		}

		if (0 < shift_pure)
		{
			weight[0] = (FLOAT)(1.0) - (std::fabs(shift_pure));
			weight[1] = std::fabs(shift_pure);
		}
		else
		{
			weight[1] = (FLOAT)(1.0) - (std::fabs(shift_pure));
			weight[0] = std::fabs(shift_pure);
		}
	}
	else
	{
		dim = 1;
		shift_index[0] = 0;
		weight[0] = 1;
	}
	
}

/*! This function is needed by one point correlation to evaluate antialising correction*/
template <typename FLOAT>
void get_index_shift_and_weight(FLOAT shift[], int &dim, int shift_index[], FLOAT weight[])
{
	
	int i;
	int dimx, dimy;
	int shift_index_x[2], shift_index_y[2];
	FLOAT weight_x[2], weight_y[2];

	calculate_linear_overlap_weight(dimx, shift[0], shift_index_x, weight_x);
	calculate_linear_overlap_weight(dimy, shift[1], shift_index_y, weight_y);

	dim = dimx*dimy;

	switch (dim)
	{
	case 1:
		weight[0] = 1;
		for (i = 0; i < 2; ++i) shift_index[i] = 0;
		break;
	case 2:

		for (i = 0; i < dim; ++i)
		{
			if (1 == dimy)
			{
				shift_index[2 * i] = shift_index_x[i];
				shift_index[2 * i + 1] = 0;
				weight[i] = weight_x[i];
			}
			else
			{
				shift_index[2 * i] = 0;
				shift_index[2 * i + 1] = shift_index_y[i];
				weight[i] = weight_y[i];
			}
		}


		break;
	case 4:

		shift_index[0] = shift_index_x[0];	shift_index[1] = shift_index_y[0];
		weight[0] = weight_x[0] * weight_y[0];
		shift_index[2] = shift_index_x[1];	shift_index[3] = shift_index_y[0];
		weight[1] = weight_x[1] * weight_y[0];
		shift_index[4] = shift_index_x[0];	shift_index[5] = shift_index_y[1];
		weight[2] = weight_x[0] * weight_y[1];
		shift_index[6] = shift_index_x[1];	shift_index[7] = shift_index_y[1];
		weight[3] = weight_x[1] * weight_y[1];

		break;
	default:
		break;
	}

	for (i = 0; i < dim; ++i)
	{
		shift_index[2 * i] += (int)(std::floor(shift[0] + 0.5));
		shift_index[2 * i + 1] += (int)(std::floor(shift[1] + 0.5));
	}
	/**/
	/*
	FLOAT sum = 0;
	for (i = 0; i < dim; ++i)
	{
	std::cout << shift_index[2 * i] << ", " << shift_index[2 * i + 1] << " ->\t" << weight[i] << std::endl;
	sum += weight[i];
	}
	std::cout << "sum = " << sum << std::endl;
	*/
}

/*!
This function performs the one point shift correlation between to images correcting for antialisiaing
The pixel is splitted according to the overlap with the other picture and the scalar product is weighted by 
the area of overlap.
*/
template<typename FLOATOUT, typename FLOAT>
FLOATOUT one_point_correlation(int dimx, int dimy, FLOAT mat[], int dimx_kernel, int dimy_kernel, FLOAT kernel[], FLOAT point_shift[])
{
	
	int i, j, k, dim_shift, dimx_kernel_d2, dimy_kernel_d2, dimtot;
	int ik, jk;
	FLOATOUT weight[4];
	FLOATOUT corr_to_weight[4];
	int shift_index[4 * 2];
	FLOAT val_kernel;
	FLOATOUT sum = (FLOATOUT)(0.0);
	FLOAT *tmpmat = new FLOAT[dimx_kernel*dimy_kernel];

	dimx_kernel_d2 = dimx_kernel / 2;
	dimy_kernel_d2 = dimy_kernel / 2;

	for (i = 0; i < 2; ++i)
		get_index_shift_and_weight(point_shift, dim_shift, shift_index, weight);


	for (i = 0; i < dim_shift; ++i)
	{
		copy_submatrix_periodical(dimx, dimy, mat, shift_index[2 * i] - dimx_kernel_d2, shift_index[2 * i + 1] - dimy_kernel_d2, dimx_kernel, dimy_kernel, tmpmat);
	
		corr_to_weight[i] = (FLOATOUT)(0.0);
		dimtot = dimx_kernel * dimy_kernel;
		for (int iii = 0; iii < dimtot; ++iii) corr_to_weight[i] += kernel[iii] * tmpmat[iii];
		sum += corr_to_weight[i] * weight[i];
	}


	delete[] tmpmat;
	return sum;
}


/*! By using one_point_correlation, this function performs the correlation of two images along a line. */
template<typename FLOAT>
void line_correlation(int dimx, int dimy, FLOAT mat[], int dimx_kernel, int dimy_kernel, FLOAT kernel[], int dimline, FLOAT line_points[], FLOAT corr_line[])
{
	
	int i, j;

	for (i = 0; i < dimline; ++i)
	{
		corr_line[i] = (CUFFT_REAL)(one_point_correlation<double, CUFFT_REAL>(dimx, dimy, mat, dimx_kernel, dimy_kernel, kernel, &(line_points[2*i])));
	}
	

}

#endif