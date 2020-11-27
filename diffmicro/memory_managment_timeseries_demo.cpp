/*
Copyright: Mojtaba Norouzisadeh, Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
e-mail:norouzi.mojtaba.sade@gmail.com

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

#include "figure_opencv.h"
//#include "correlation.h"
#include "power_spectra.h"
#include "diffmicro_io.h"
#include <cmath>
#include <iostream>
#include "cuda_init.h"
#include "device_launch_parameters.h"


__global__ void cpx_col2row_gain__gpu(INDEX dimcopy, INDEX dimx_in, INDEX i_col_in, CUFFT_COMPLEX in[], CUFFT_REAL gain, INDEX dimx_out, INDEX i_row_out, CUFFT_COMPLEX out[])
{

	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;


	INDEX i_in, i_out;

	if (i < dimcopy)
	{
		i_in = i * dimx_in + i_col_in;
		i_out = i_row_out * dimx_out + i;

		out[i_out].x = gain * in[i_in].x;
		out[i_out].y = gain * in[i_in].y;
	}
}


__global__ void cpx_row2col_gain__gpu(INDEX dim, INDEX dimx_in, INDEX i_row_in, CUFFT_COMPLEX in[], FFTW_REAL gain, INDEX dimx_out, INDEX i_col_out, CUFFT_COMPLEX out[])
{
	INDEX i = blockDim.x * blockIdx.x + threadIdx.x;
	INDEX i_in, i_out;
	if ( i < dim)
	{
		i_in = i_row_in * dimx_in + i;
		i_out = i * dimx_out + i_col_out;

		out[i_out].x = gain * in[i_in].x;
		out[i_out].y = gain * in[i_in].y;
	}
}

void printcpxmat(INDEX dimx, INDEX dimy, FFTW_COMPLEX *mat)
{
	INDEX i, j;
	for (j = 0; j < dimy; ++j)
	{
		for (i = 0; i < dimx-1; ++i)
		{
			std::cout << mat[j*dimx+i][0] << " + i " << mat[j*dimx+i][1] << "   ";
		}
		std::cout << mat[j*dimx+dimx - 1][0] << " + i " << mat[j*dimx+dimx - 1][1] << std::endl;
	}
}

void printcpxmat(INDEX dimx, INDEX dimy, CUFFT_COMPLEX* mat)
{
	INDEX i, j;
	for (j = 0; j < dimy; ++j)
	{
		for (i = 0; i < dimx - 1; ++i)
		{
			std::cout << mat[j * dimx + i].x << " + i " << mat[j * dimx + i].y << "   ";
		}
		std::cout << mat[j * dimx + dimx - 1].x << " + i " << mat[j * dimx + dimx - 1].y << std::endl;
	}
}





int ddd_main(int argc, char* argv[])
{
	INDEX i, i1col;
	INDEX dimx1, dimy1, dimx2, dimy2;
	FFTW_COMPLEX* mat1, * mat2;
	FFTW_REAL* vet;
	FFTW_REAL* vet2;
	INDEX dim = 32;
	INDEX dimchuck = 8;

	CUFFT_COMPLEX* matg1, * matg2, * matg1_temp, *matg2_temp;


	dimx1 = 4;
	dimy1 = 5;
	dimx2 = 5;
	dimy2 = 3;

	mat1 = (FFTW_COMPLEX*)( fftw_malloc(dimx1 * dimy1 * sizeof(FFTW_COMPLEX) )) ;
	mat2 = (FFTW_COMPLEX*)(fftw_malloc(dimx2 * dimy2 * sizeof(FFTW_COMPLEX)));

	matg1_temp = (CUFFT_COMPLEX*)(fftw_malloc(dimx1 * dimy1 * sizeof(CUFFT_COMPLEX)));
	matg2_temp = (CUFFT_COMPLEX*)(fftw_malloc(dimx2 * dimy2 * sizeof(CUFFT_COMPLEX)));

	
	cuda_init(true);
	cuda_exec mycuda_dim;
	cuda_exec mycuda_dim_i;
	calc_cuda_exec(dimy1, deviceProp.maxThreadsPerBlock, &mycuda_dim);
	calc_cuda_exec(dimx2, deviceProp.maxThreadsPerBlock, &mycuda_dim_i);

	cudaMalloc(&matg1, dimx1 * dimy1 * sizeof(CUFFT_COMPLEX));
	cudaMalloc(&matg2, dimx2 * dimy2 * sizeof(CUFFT_COMPLEX));


	for (i = 0; i < dimx1 * dimy1; ++i)
	{
		mat1[i][0] = 2.0;
		mat1[i][1] = (FFTW_REAL)(i);

		matg1_temp[i].x = 2.0;
		matg1_temp[i].y = (FFTW_REAL)(i);

	}
	for (i = 0; i < dimx2 * dimy2; ++i)
	{
		mat2[i][0] = 0.0;
		mat2[i][1] = 0.0;
	}


	i1col = 1;
	for (i = 0; i < dimx1; ++i)
	{
		mat1[i1col * dimx1 + i][0] = (FFTW_REAL)(i);
	}

	printcpxmat(dimx1, dimy1, mat1);
	std::cout << std::endl;
	printcpxmat(dimx2, dimy2, mat2);
	std::cout << std::endl;
	std::cout << " ========================================================= " << std::endl;
	printcpxmat(dimx1, dimy1, matg1_temp);
	std::cout << std::endl;
	std::cout << " ========================================================= " << std::endl;
	std::cout << cudaMemcpy(matg1, matg1_temp, dimx1 * dimy1 * sizeof(CUFFT_COMPLEX), cudaMemcpyHostToDevice) << std::endl;


	cpx_col2row_gain_cpu(dimy1, dimx1, 2, mat1, (FFTW_REAL)(2.0), dimx2, 1, mat2);

	cpx_col2row_gain__gpu <<<mycuda_dim.nbk, mycuda_dim.nth >>>  (dimy1, dimx1, 2, matg1, (FFTW_REAL)(2.0), dimx2, 1, matg2);
	cudaDeviceSynchronize();
	std::cout << cudaMemcpy(matg2_temp, matg2, dimx2 * dimy2 * sizeof(CUFFT_COMPLEX), cudaMemcpyDeviceToHost) << std::endl;

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));


	std::cout << " ========================================================= "<<std::endl;
	printcpxmat(dimx1, dimy1, mat1);
	std::cout << std::endl;
	printcpxmat(dimx2, dimy2, mat2);
	std::cout << std::endl;

	std::cout << " ========================================================= " << std::endl;
	printcpxmat(dimx2, dimy2, matg2_temp);
	std::cout << std::endl;
	std::cout << " ========================================================= " << std::endl;


	cpx_row2col_gain_cpu(dimx2, dimx2, 1, mat2, (FFTW_REAL)(2.0), dimx1, 2, mat1);

	cpx_row2col_gain__gpu << <mycuda_dim_i.nbk, mycuda_dim_i.nth >> > (dimx2, dimx2, 1, matg2, (FFTW_REAL)(2.0), dimx1, 2, matg1);
	cudaDeviceSynchronize();
	std::cout << cudaMemcpy(matg1_temp, matg1, dimx1 * dimy1 * sizeof(CUFFT_COMPLEX), cudaMemcpyDeviceToHost) << std::endl;


	printcpxmat(dimx1, dimy1, mat1);
	std::cout << std::endl;
	printcpxmat(dimx2, dimy2, mat2);
	std::cout << std::endl;
	
	std::cout << " ========================================================= " << std::endl;
	printcpxmat(dimx1, dimy1, matg1_temp);
	std::cout << std::endl;
	std::cout << " ========================================================= " << std::endl;

	vet = new FFTW_REAL[dim];
	vet2 = new FFTW_REAL[dim];
	for (i = 0; i < dim; ++i)
	{
		vet[i] = (FFTW_REAL)(i);
	}

	useri.power_spectra_filename = "prova";
	for (i = 0; i < dim / dimchuck; ++i)
	{
		writeappend_partial_lutpw(i, 0, dimchuck, &(vet[i*dimchuck]));
	}
	read_mergedpartial_lutpw(0, dim, vet2);

	for (i = 0; i < dim; ++i)
	{
		std::cout << "original[" << i << "] = " << vet[i] << "\t";
		std::cout << "fromfile[" << i << "] = " << vet2[i]<<std::endl;
	}

	fftw_free(mat1);
	fftw_free(mat2);
	delete[] vet, vet2;

	return 0;
}
