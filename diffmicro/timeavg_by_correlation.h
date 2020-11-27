/*
Copyright: Mojtaba Norouzisadeh, Giovanni Cerchiari
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

#ifndef _TIMEAVG_CORRELATION_H_
#define _TIMEAVG_CORRELATION_H_

#include "global_define.h"

/*!
This function calculates the average vector of abssolute values as needed in the calculation
of the average power spectra over time. It is the sum of two averages.
In out[dim-1-i] the averages of the first i and last i square absolute values are added together.
The reverse ordering is needed in the final calculation.
- dim -> number of elements in the arrays
- in -> input
- out -> output
WARNING: Do not use this function in multithreading!!!
*/
void averagesabs2_array_cpu(INDEX dim, FFTW_COMPLEX* in, FFTW_REAL* out);
void averagesabs2_array_cpu(INDEX dim, INDEX dim_t, FFTW_COMPLEX* in, FFTW_REAL* out);

/*!
This function divides the real part of the complex vector in input by the linear decreasing
ramp of integers.
- dim -> number of elements in the arrays
- ramp_start -> index at which the decreasing ramp starts
- in -> input
- update -> to this array the result of the calculation will be subtracted
*/
void updatewithdivrebyramp_cpu(INDEX dim, INDEX ramp_start, FFTW_COMPLEX* in, FFTW_REAL* update);

/*!
Multithreading version of updatewithdivrebyramp_cpu.
This function divides the real part of the complex vector in input by the linear decreasing
ramp of integers. The ramp starts at dim
- nth -> number of threads
- dim -> number of elements in the arrays
- in -> input
- update -> to this array the result of the calculation will be subtracted
*/
void update_with_divrebyramp_cpu(INDEX nth, INDEX dim, FFTW_COMPLEX* in, FFTW_REAL* update);

/*!
This function calculates the absolute modulus square of a complex array into the real part
of an output array
- dim -> number of elements in the arrays
- in -> input
- out -> output
*/
void complexabs2_cpu(INDEX dim, FFTW_COMPLEX* in, FFTW_COMPLEX* out);

/*!
Multithreading version of complex_abs2_cpu.
This function calculates the absolute modulus square of a complex array into the real part
of an output array
- nth -> number of threads
- dim -> number of elements in the arrays
- in -> input
- out -> output
*/
void complex_abs2_cpu(INDEX nth, INDEX dim, FFTW_COMPLEX* in, FFTW_COMPLEX* out);

/*!
multiplies a complex array by a real number.
- dim -> number of elements in the array
- in -> input
- gain -> multiplicative term
- out -> output
*/
void gaincomplex_cpu(INDEX dim, FFTW_COMPLEX in[], FFTW_REAL gain, FFTW_COMPLEX out[]);

/*!
Multithreading version of gaincomplex_cpu.
multiplies a complex array by a real number.
- nth -> number of threads
- dim -> number of elements in the array
- in -> input
- gain -> multiplicative term
- out -> output
*/
void gain_complex_cpu(INDEX nth, INDEX dim, FFTW_COMPLEX in[], FFTW_REAL gain, FFTW_COMPLEX out[]);

#endif



