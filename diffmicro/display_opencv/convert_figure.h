/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 12/2015

implemented with opencv v 3.0
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


#ifndef _CONVERT_FIGURE_H_
#define _CONVERT_FIGURE_H_

#include "figure_opencv.h"

template<typename FLOATIN, typename FLOATOUT>
void copy_vet(unsigned int dim, FLOATIN *in, FLOATOUT *out)
{
	unsigned int i;

	for (i = 0; i < dim; ++i) out[i] = (FLOATOUT)(in[i]);
}

template<typename FLOATIN>
void new_figure_t(unsigned int dimx, unsigned int dimy, FLOATIN *img, double *buffer)
{
	copy_vet<FLOATIN, double>(dimx*dimy, img, buffer);
	new_figure(dimx, dimy, buffer);
}


#endif