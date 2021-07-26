/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 8/2011
updated: 6/2020
*/
/*!
This functions are written for diffmicro.exe application.
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
#include "angular_avg.h"
//#include <vector>
//#include <algorithm>

INDEX dimside(0);
INDEX dim(0);
INDEX dimr(0);

INDEX angavg_nth(0);

struct th_arg_st
{
	STORE_REAL* tmp_mat;
	angular_integrator<MY_REAL, STORE_REAL>* ang_int;
} ;

th_arg_st *th_arg(NULL);

void power_spectra_to_azhavg_init(INDEX nth, INDEX _dimside, INDEX &_dimr)
{
	INDEX i;

	dimside = _dimside;
	dim = dimside * dimside;

	angavg_nth = nth;

	th_arg = new th_arg_st[angavg_nth];

	for(i=0; i< angavg_nth; ++i)
		{
			th_arg[i].tmp_mat = new STORE_REAL[dim];
			th_arg[i].ang_int = new angular_integrator<MY_REAL, STORE_REAL>;
		}

	th_arg[0].ang_int->init(dimside);
	for(i=1; i< angavg_nth; ++i)	*(th_arg[i].ang_int) = *(th_arg[0].ang_int);

	dimr = th_arg[0].ang_int->dimr();
	_dimr = dimr;

}

void power_spectra_to_azhavg_free()
{
 INDEX i;

	for(i=0; i<angavg_nth; ++i)
		{
			delete[] th_arg[i].tmp_mat;
			delete th_arg[i].ang_int;
		}
	if (NULL != th_arg)
		delete[] th_arg;
}

void power_spectrum_to_azhavg(STORE_REAL *pw, MY_REAL *azh_avg, STORE_REAL *mat, angular_integrator<MY_REAL, STORE_REAL> *ang_int)
{
	//---------------------------------------------------------------------------
	// declaration
	int i, j;
	INDEX dims = ang_int->dimside();
	INDEX dims_b_2 = dims * 2;
	INDEX dimrad = ang_int->dimr();
	STORE_REAL *row_mat, *row_pw;
	STORE_REAL zp5 = 0.5;
	MY_REAL *wr = ang_int->wr();

	//----------------------------------------------------------------------------
	// reducing half power spectrum to one quarter power spectrum
//#pragma omp parallel for
	
	for (j = 0; j<dims; ++j)
	{
		row_pw = &(pw[j * dims_b_2]);
		row_mat = &(mat[j * dims]);
		row_mat[0] = row_pw[0];
		for (i = 1; i < dims; ++i) {

			row_mat[i] = zp5 * (row_pw[i] + row_pw[dims_b_2 - i]);
			//std::cout <<i<<"    "<< row_pw[i] <<"    "<< row_pw[dims_b_2 - i] << std::endl;
		}

	}

	//--------------------------------------------------------------------------
	// angular integral and division to obtain the average 
	ang_int->operator()(mat, azh_avg);
#pragma omp parallel for
	for (i = 0; i < dimrad; ++i)
	{
		azh_avg[i] = azh_avg[i] / wr[i];
	}
}


void power_spectra_to_azhavg(INDEX npw, STORE_REAL* pw, MY_REAL* azh_avgs)
{
	// declaration
	INDEX ind_cycle,j,jj;
	INDEX cycle = (INDEX)(std::div((int)(npw),(int)(angavg_nth)).quot);
	INDEX dim_b_2 = dim * 2;
	std::thread** th;
	th = new std::thread * [angavg_nth];
		//----------------------------------------------------------------------
	// threads ititialization


	for(ind_cycle = 0; ind_cycle < cycle; ++ind_cycle)
		{
			for(j=0; j< angavg_nth; ++j)
				{
					jj = j + ind_cycle * angavg_nth;
					
					th[j] = new std::thread(power_spectrum_to_azhavg, &(pw[jj * dim_b_2]), &(azh_avgs[jj * dimr]),
						th_arg[j].tmp_mat, th_arg[j].ang_int);
				}
			// wait
			for(j=0; j< angavg_nth; ++j) th[j]->join();
		}

	//----------------------------------------------------------
	// out of threads elements
	for(j=cycle* angavg_nth; j<npw; ++j)
		{
			power_spectrum_to_azhavg(&(pw[j * dim_b_2]), &(azh_avgs[j * dimr]),
				th_arg[0].tmp_mat, th_arg[0].ang_int);
		}

	delete[] th;
}

void power_spectra_to_azhavg_test(int indx,INDEX npw, STORE_REAL* pw, FFTW_COMPLEX* dev_images_cpu, unsigned int* lut, MY_REAL* azh_avgs)
{

	// declaration
	int i, j;
	INDEX dims = th_arg[0].ang_int->dimside();
	INDEX dims_b_2 = dims * 2;
	INDEX dimrad = th_arg[0].ang_int->dimr();
	STORE_REAL* row_mat, * row_pw;
	STORE_REAL zp5 = 0.5;
	MY_REAL* wr = th_arg[0].ang_int->wr();
	//STORE_REAL a=0, b=0;
	//int c;
	//typedef IntContainer::iterator IntIterator;
	//----------------------------------------------------------------------------
	// reducing half power spectrum to one quarter power spectrum
//#pragma omp parallel for

	for (j = 0; j < dims; ++j)
	{
		//row_pw = &(pw[j * dims_b_2]);
		row_mat = &(th_arg[0].tmp_mat[j * dims]);
		//row_mat[0] = row_pw[0];
		for (i = 1; i < dims; ++i) {
			
			/*for (int k = 0; k < 15798; k++) {
				if ((i + j * dims_b_2) == lut[k]) {
					a = (STORE_REAL)dev_images_cpu[indx + k * 100][0]; //pw[i + j * dims_b_2]
					break;
				}
			}*/

			row_mat[i] = zp5 * (pw[i + j * dims_b_2] + pw[dims_b_2 - i + j * dims_b_2]);


		}
	}

	//--------------------------------------------------------------------------
	// angular integral and division to obtain the average 
	th_arg[0].ang_int->operator()(th_arg[0].tmp_mat, azh_avgs);
#pragma omp parallel for
	for (i = 0; i < dimrad; ++i)
	{
		azh_avgs[i] = azh_avgs[i] / wr[i];
	}
}