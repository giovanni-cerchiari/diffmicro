
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

#ifndef _SPHERICAL_HYPERBOLICAL_PROJECTION_H_
#define _SPHERICAL_HYPERBOLICAL_PROJECTION_H_

enum spheric_hyperbolic_project_enum
{
	SPHERICAL_PROJECTION = 0,
	HYPERBOLICAL_PROJECTION = 1
};

#include <cmath>

/*!
This function returns the mapped indices coordinates for spherical or hyperbolical projection of images.
It is intended to allow the reconstruction of images deformed by fish-eye lenses.
The output indices are thought to be one where the value of the matrix pixel has to be read.
The decision how to behave with the output indices (if interpolating or not) is deferred outside this function
The indices are stored this way : x0,y0,x1,y1,...

- r radius in the third dimension out of the picture of the sphere
- dimx, dimy dimensions of the indout matrix
- center[] (centerx, centery) bias position of the center of the sphere in image plane
- scale[] (scalex, scaley) pixel lateral dimension
- indout output
- type (0 ==  spherical), (1== hyperbolical)
*/
template<typename FLOAT, typename UINT>
void spheric_hyperbolic_indices_for_projection(FLOAT r, UINT dimx, UINT dimy, FLOAT scale[], FLOAT *indout, int type = 0)
{
	UINT i, j;
	FLOAT *indoutrow;
	FLOAT x, y;
	FLOAT theta;

	FLOAT(*_sin)(FLOAT);
	FLOAT(*_cos)(FLOAT);
	FLOAT(*_tan)(FLOAT);
	FLOAT(*_atan)(FLOAT);
	FLOAT(*_asin)(FLOAT);

	switch (type)
	{
	case HYPERBOLICAL_PROJECTION:
		_sin = sinh;
		_cos = cosh;
		_tan = tanh;
		_asin = asinh;
		_atan = atanh;
		break;
	case SPHERICAL_PROJECTION:
		_sin = sin;
		_cos = cos;
		_tan = tan;
		_asin = asin;
		_atan = atan;

	default:
		break;
	}

	for (j = 0; j < dimy; ++j)
	{
		indoutrow = &(indout[2 * j * dimx]);
		for (i = 0; i < dimx; ++i)
		{
			y = ((FLOAT)(j)-0.5*(FLOAT)(dimy-1)) * scale[1];
			x = ((FLOAT)(i)-0.5*(FLOAT)(dimx-1)) * scale[0];

			theta = _atan(sqrt(x*x + y*y) / r);

//			std::cerr << "cos(theta = " << theta <<") = "<<_cos(theta)<< std::endl;

			indoutrow[2 * i    ] = x*_cos(theta);
			indoutrow[2 * i + 1] = y*_cos(theta);
//			std::cerr << "(" << i << "," << j << ") \t (" << x << "," << y << ") \t (" << x*_cos(theta) << "," << y*_cos(theta) << ")" << std::endl;
		}
	}

}

/*!
This function take into consideration the copy job and reduces all the redirection of the matrices in terms of two indices array:

indok is the array of pixel where an effective copy is taking place
indzero is the array of pixel where no corresponding pixel exist and might be assigned value zero

indok and indzero indices are the already prepared global indices to jump directly into the matrices. No row-column calculation is further needed.
indok memory layout -> (indok_in_0, indok_out_0, indok_in_1, indok_out_1,...)
indzero memory layout -> (indzero_0, indzero_1, indero_2...)

center[] (centerx, centery) is the center of the sphere on the input matrix in pixel scale
*/
template<typename FLOAT, typename UINT>
void spheric_hyperbolic_indices_reduction(UINT dimxin, UINT dimyin, FLOAT in[], FLOAT center[], FLOAT scale[], UINT dimxout, UINT dimyout, FLOAT ind[],
	                                                    UINT &dimindok, UINT indok[], UINT &dimindzero, UINT indzero[])
{
	UINT i, j;
	FLOAT x, y;
	UINT ii, jj;

	dimindok = 0;
	dimindzero = 0;

	for (j = 0; j < dimyout; ++j)
	{
		for (i = 0; i < dimxout; ++i)
		{
			x = ind[2 * (j*dimxout + i)] + center[0];
			y = ind[2 * (j*dimxout + i) + 1] + center[1];

			ii = (UINT)(std::round(x));
			jj = (UINT)(std::round(y));

			// warning unsigned trick
			if (ii < dimxin && jj < dimyin)
			{
				indok[2 * dimindok    ] = jj*dimxin +ii;
				indok[2 * dimindok + 1] = j *dimxout+i;
				++dimindok;
			}
			else
			{
				indzero[dimindzero] = j *dimxout + i;
				++dimindzero;
			}

		}
	}
}

/*!
The easy part of the projection :). Copy what has to be copied and put to zero what would be non-defined

*/

template<typename FLOAT, typename UINT>
void spheric_hyperbolic_projection_with_indices(FLOAT in[], UINT dimok, UINT indok[], UINT dimzero, UINT indzero[], FLOAT out[])
{
	UINT i;

	for (i = 0; i < dimzero; ++i) out[indzero[i]] = (FLOAT)(0);
	for (i = 0; i < dimok; ++i) out[indok[2 * i + 1]] = in[indok[2 * i]];

}

/*!
Example function to show the usage of the functions contained in this file
To speed up execution the indices may be saved
*/
template<typename FLOAT, typename UINT>
void spheric_hyperbolic_projection_easy(FLOAT r, UINT dimxin, UINT dimyin, FLOAT in[], FLOAT center[], FLOAT scale[],UINT dimxout, UINT dimyout, FLOAT out[], int type = 0)
{
	FLOAT *ind;
	UINT dimzero, dimok;
	UINT *indzero, *indok;

	ind = new FLOAT[2*dimxout*dimyout];
	indzero = new UINT[dimxout*dimyout];
	indok = new UINT[2 * dimxout*dimyout];

	spheric_hyperbolic_indices_for_projection(r, dimxout, dimyout, scale, ind, type);

//	for (int j = 0; j < dimyout; ++j)
//	{
//		for (int i = 0; i < dimxout; ++i)
//		{
//			std::cout << "(" << i << "," << j << ")\t(" << ind[2 * (j*dimxout+i)] << "," << ind[2 * (j*dimxout+i) + 1]<<")" << std::endl;
//		}
//	}

	spheric_hyperbolic_indices_reduction(dimxin, dimyin, in, center, scale, dimxout, dimyout, ind, dimok, indok, dimzero, indzero);
	spheric_hyperbolic_projection_with_indices(in, dimok, indok, dimzero, indzero, out);

	delete[] ind;
	delete[] indzero;
	delete[] indok;
}



#endif

