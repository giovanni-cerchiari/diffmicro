/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com

date: 2016
update: 05/2020-09/2020
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

/*!
This set of function allows converting a matrix into a vector via a look up table and viceversa.
The look up table is used to save preferentially low frequency components in frequency space.
This components are located radially out from the center of the Fourier transform, which is in
the corner of the Fourier transform matrix. In fact, due to periodically boundary conditions,
both the upper left and upper right corners of the image are saved. The lower left and lower right
corners may be discarded because we start from real valued images and the symmetry of the Fourier 
tranform makes those corners redoundant.
*/

#ifndef _RADIAL_STORAGE_LUT_H_
#define _RADIAL_STORAGE_LUT_H_

#include <iostream>
#include <vector>
#include <algorithm>

#include "global_define.h"
#include "my_math.h"

/*!Length of ram_radial_lut array*/
extern INDEX dim_radial_lut;
/*!memory area to store the radial look up table in CPU RAM*/
extern unsigned int *ram_radial_lut;

/*!
This struct associates the index of the Fourier transform matrix to the wavenumber. Here the wavenumber is
the absolute value of the frequency discarding the direction. It is the radial distance of the Fourier 
transform pixel from the fundamental frequency.
*/
struct radial_storage_lut_struct
{
	INDEX i;
	float r;
};

/*!This function compares two radial_storage_lut_struct by comparing the radius. It is used to apply std::sort*/
bool comp_radial_storage_lut_struct(radial_storage_lut_struct &f1, radial_storage_lut_struct &f2);

/*!This function calculates the dimension of the radial look up table to allow for allocating the correct memory area*/
void radial_lut_dimr_from_max_freq(INDEX dimy, INDEX dimx, float max_freq, INDEX &dimr);
/*!This function fill the variable lut with the look up table. Please, use radial_lut_dimr_from_max_freq beforehand 
and prepare lut of correct length.*/
void radial_storage_lut(INDEX dimy, INDEX dimx, INDEX dimr, unsigned int *lut);

void index_shiftedFFT(INDEX dimx, INDEX dim_freq, unsigned int* ram_radial_lut);


/*!This function converts a vector into a matrix by using the look up table. It is the inverse transform of radial_lut_normal_to_rad.*/
template<typename TYPEIN, typename TYPEOUT>
void radial_lut_rad_to_normal(INDEX dimnormal, INDEX dimr, unsigned int *lut, TYPEIN *in, TYPEOUT *out)
{
	INDEX i;

	memset(out, 0, sizeof(TYPEOUT)* dimnormal);
	for (i = 0; i < dimr; ++i)
	{
		out[lut[i]] = (TYPEOUT)(in[i]);
	}
}

/*!This function converts a matrix into a vector by using the look up table. It is the inverse transform of radial_lut_rad_to_normal.*/
template<typename TYPEIN, typename TYPEOUT>
void radial_lut_normal_to_rad(INDEX dimr, unsigned int *lut, TYPEIN *in, TYPEOUT *out)
{
	INDEX i;

	for (i = 0; i < dimr; ++i)
	{
		out[i] = (TYPEOUT)(in[lut[i]]);
	}
}



#endif

