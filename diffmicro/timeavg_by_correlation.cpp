/*
Copyright: Mojtaba Norouzi, Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
e-mail: norouzi.mojtaba.sade@gmail.com

date: 05/2020 - 09/2020
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
#include "timeavg_by_correlation.h"
#include <thread>


void averagesabs2_array_cpu(INDEX dim, FFTW_COMPLEX *in, FFTW_REAL *out)
{
	FFTW_REAL avg = 0.0;
	FFTW_REAL coef1, coef2, abs2_fromstart, abs2_fromend;
	INDEX i, ii;
	for (i = 0; i < dim; ++i)
	{
		// next absolute value from the beginning of the array
		abs2_fromstart = in[i][0] * in[i][0] + in[i][1] * in[i][1];

		// next absolute value from the end of the array
		ii = dim - 1 - i;
		abs2_fromend = in[ii][0] * in[ii][0] + in[ii][1] * in[ii][1];

		// in-place average
		coef2 = (FFTW_REAL)(1.0) / (FFTW_REAL)(i + 1);
		coef1 = (FFTW_REAL)(i)*coef2;
		avg = coef1 * avg + coef2 * (abs2_fromstart + abs2_fromend);

		// save the result in the output array
		// ATTENTION! note the index
		out[ii] = avg;
	}
}


void averagesabs2_array_cpu(INDEX dim, INDEX dim_t, FFTW_COMPLEX* in, FFTW_REAL* out)
{
	for (INDEX j = 0; j < dim_t; ++j)
	{
		FFTW_REAL avg = 0.0;
		FFTW_REAL coef1, coef2, abs2_fromstart, abs2_fromend;
		INDEX i, ii;
		for (i = 0; i < dim; ++i)
		{
			// next absolute value from the beginning of the array
			abs2_fromstart = in[i + j * dim][0] * in[i + j * dim][0] + in[i + j * dim][1] * in[i + j * dim][1];

			// next absolute value from the end of the array
			ii = dim - 1 - i ;
			abs2_fromend = in[ii + j * dim][0] * in[ii + j * dim][0] + in[ii + j * dim][1] * in[ii + j * dim][1];

			// in-place average
			coef2 = (FFTW_REAL)(1.0) / (FFTW_REAL)(i + 1);
			coef1 = (FFTW_REAL)(i)*coef2;
			avg = coef1 * avg + coef2 * (abs2_fromstart + abs2_fromend);

			// save the result in the output array
			// ATTENTION! note the index
			out[ii+j*dim] = avg;
		}
	}
}


void updatewithdivrebyramp_cpu(INDEX dim, INDEX ramp_start, FFTW_COMPLEX* in, FFTW_REAL* update)
{
	INDEX i;
	for (i = 0; i < dim; ++i)
		update[i] -= (2. / (FFTW_REAL)(ramp_start - i)) * in[i][0];
}

void update_with_divrebyramp_cpu(INDEX nth, INDEX dim, FFTW_COMPLEX* in, FFTW_REAL* update)
{
	INDEX i;
	lldiv_t d;
	std::thread** th;
	th = new std::thread * [nth];

	d = div((long long)dim, nth);

	// starting all threads
	for (i = 0; i < nth; ++i)
		th[i] = new std::thread(updatewithdivrebyramp_cpu, d.quot, dim-i * d.quot, &(in[i * d.quot]), &(update[i * d.quot]));
	if (0 < d.rem)
		updatewithdivrebyramp_cpu(d.rem, dim-i * d.quot, &(in[i * d.quot]), &(update[i * d.quot]));

	// waiting for all threads to finish
	for (i = 0; i < nth; ++i)
		th[i]->join();

	delete[] th;
}


void complexabs2_cpu(INDEX dim, FFTW_COMPLEX* in, FFTW_COMPLEX* out)
{
	INDEX i;
	for (i = 0; i < dim; ++i)
	{
		out[i][0] = in[i][0] * in[i][0] + in[i][1] * in[i][1];
		out[i][1] = 0.0;
	}
}

void complex_abs2_cpu(INDEX nth, INDEX dim, FFTW_COMPLEX* in, FFTW_COMPLEX* out)
{
	INDEX i;
	lldiv_t d;
	std::thread** th;
	th = new std::thread * [nth];

	d = div((long long)dim, nth);

	// starting all threads
	for (i = 0; i < nth; ++i)
		th[i] = new std::thread(complexabs2_cpu, d.quot, &(in[i * d.quot]), &(out[i * d.quot]));
	if (0 < d.rem)
		complexabs2_cpu(d.rem, &(in[i * d.quot]), &(out[i * d.quot]));

	// waiting for all threads to finish
	for (i = 0; i < nth; ++i)
		th[i]->join();

	delete[] th;
}


void gaincomplex_cpu(INDEX dim, FFTW_COMPLEX in[], FFTW_REAL gain, FFTW_COMPLEX out[])
{
	for (INDEX i = 0; i < dim; ++i)
	{
		out[i][0] = gain * in[i][0];
		out[i][1] = gain * in[i][1];
	}
}

void gain_complex_cpu(INDEX nth, INDEX dim, FFTW_COMPLEX in[], FFTW_REAL gain, FFTW_COMPLEX out[])
{
	INDEX i;
	lldiv_t d;
	std::thread** th;
	th = new std::thread * [nth];

	d = div((long long)dim, nth);

	// starting all threads
	for (i = 0; i < nth; ++i)
		th[i] = new std::thread(gaincomplex_cpu, d.quot, &(in[i * d.quot]),gain, &(out[i * d.quot]));
	if (0 < d.rem)
		gaincomplex_cpu(d.rem, &(in[i * d.quot]), gain, &(out[i * d.quot]));

	// waiting for all threads to finish
	for (i = 0; i < nth; ++i)
		th[i]->join();

	delete[] th;
}