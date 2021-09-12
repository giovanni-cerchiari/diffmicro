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


#include "stdafx.h"

#include "radial_storage_lut.h"

INDEX dim_radial_lut(0);
unsigned int *ram_radial_lut;

bool comp_radial_storage_lut_struct(radial_storage_lut_struct &f1, radial_storage_lut_struct &f2)
{
	return (f1.r < f2.r);
}

void radial_lut_prepare_vet(INDEX dimx, INDEX dimy, std::vector<radial_storage_lut_struct> &vet)
{
	INDEX ii, dim;
	__int64 i, j;
	double rr;

	dim = dimx*dimy;
	vet.resize(dim);
	for (ii = 0; ii < dim; ++ii)
	{
		i = (int)(ii % dimx);
		j = (int)(ii / dimx);

		if (i >= dimx / 2) i = i - dimx;
		rr = std::sqrt((float)(i*i + j*j));

		vet[ii].i = ii;
		vet[ii].r = rr;
	}

	std::sort(vet.begin(), vet.end(), comp_radial_storage_lut_struct);

}


void radial_lut_dimr_from_max_freq(INDEX dimy, INDEX dimx, float max_freq, INDEX &dimr)
{
	INDEX dim;
	std::vector<radial_storage_lut_struct> vet;

	dim = dimx*dimy;

	radial_lut_prepare_vet(dimx, dimy, vet);

	dimr = 0;
	while ((dimr < dim) && (vet[dimr].r < max_freq)) ++dimr;
	
}


void radial_storage_lut(INDEX dimy, INDEX dimx, INDEX dimr, unsigned int *lut)
{
	INDEX i, dim;
	std::vector<radial_storage_lut_struct> vet;

	dim = dimx*dimy;

	radial_lut_prepare_vet(dimx, dimy, vet);

	for (i = 0; i < dimr; ++i)
	{
		lut[i] = vet[i].i;
	}

}

void index_shiftedFFT(INDEX dimx,INDEX dim_freq, unsigned int* ram_radial_lut) {

	int crop_image = dimx; 
	int azim_crop = dim_freq;
	int xx = (crop_image-1) / 2 - (azim_crop - 1) / 2;//((crop_image-1)/2-(azim_crop-1)/2+1)
	int yy = (crop_image-1) / 2 + (azim_crop - 1) / 2;
	//std::vector<int> v;
	for (int i = 0; i < dim_freq; i++) {
		for (int j = 0; j < dim_freq; j++) {

			ram_radial_lut[j + i * dim_freq] = xx + xx * dimx + j + i * dimx;
			//v.push_back(xx + xx * dimx + j + i *dimx);
		}
	}

}



