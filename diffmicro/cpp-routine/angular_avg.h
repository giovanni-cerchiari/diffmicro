/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 8/2011
updated: 6/2020

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


#ifndef _ANGULAR_AVG_H_
#define _ANGULAR_AVG_H_

#include <iostream>
#include <cstdlib>
#include <thread>

#include "global_define.h"

#include "angular_integ.h"

/*!
initialization of the azimuthal averages calculus
- Initialization of one :	angular_integrator<REAL, STORE_REAL>
- Initialization of the Threads arguments (WARNING! -> there must be one angular integrator and one temporary matrix for each thread)
- nth -> number of threads
*/
void power_spectra_to_azhavg_init(INDEX nth, INDEX _dimside, INDEX &_dimr);

/*!
memory free of the threads
*/
void power_spectra_to_azhavg_free();


/*!
This function organize the trivial parallelization over the azimuthal averages.

It runs N_MAX_THREADS threads over N_MAX_THREADS different power spectra and compute separately
each azimuthal average
*/
void power_spectra_to_azhavg(INDEX npw, STORE_REAL* pw, MY_REAL* azh_avgs);
void power_spectra_to_azhavg_test(int indx,INDEX npw, STORE_REAL* pw, FFTW_COMPLEX* dev_images_cpu, unsigned int* lut,MY_REAL* azh_avgs);


/*!
This function organize the parallelization over the azimuthal averages.

It runs N_MAX_THREADS threads over N_MAX_THREADS different power spectra and compute separately
the azimuthal average of each image
*/
void power_spectrum_to_azhavg(STORE_REAL *pw, MY_REAL *azh_avg, STORE_REAL *mat, angular_integrator<MY_REAL, STORE_REAL> *ang_int);

#endif

