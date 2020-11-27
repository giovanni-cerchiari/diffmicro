
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


#include "stdafx.h"

#include "prep_matrix.h"

void edge_vet_init_indices(unsigned int dimin, unsigned int dimout,
                           unsigned int &start_in, unsigned int &start_out,
                           circular_index<unsigned int> &index_in,
                           circular_index<unsigned int> &index_out,
                           unsigned int &dimmin)
{
 if(dimin >= dimout)
		{
			start_out = dimout / 2;
			start_in = dimin - start_out;
			dimmin = dimout;
		}
	else
		{
			start_in = dimin / 2;
			start_out = dimout - start_in;
			dimmin = dimin;
		}

	index_out.init(dimout-1, 0, start_out);
	index_in.init(dimin-1, 0, start_in);
}
