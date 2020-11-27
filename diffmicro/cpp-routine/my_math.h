
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

#ifndef _MY_MATH_H_
#define _MY_MATH_H_

#include <iostream>
#include <vector>

#include <cmath>

#include "prep_vet.h"
#include "circular_index.h"
#include "contour.h"
#include "spherical_hyperbolical_projection.h"
#include "correlation_cpu.h"

//! 50 decimal digits greek_pi
#define GREEK_PI 3.14159265358979323846264338327950288419716939937510
//! for comparisons among floating point numbers.
#define EPS_INV_MAT	0.00000000001

/*! powers for integer numbers*/
template <typename INT, typename INT2>
inline INT pow_int(INT arg, INT2 exp)
{
 unsigned int i;
 INT arg1 = arg;
 for(i=1; i<exp; ++i) arg = arg * arg1;
 return arg;
}

template<typename INT, typename FLOAT>
inline void gain_a(INT dim, FLOAT v[], FLOAT gain)
{
	INT i;
	for(i=0;i<dim;++i)v[i]*=gain;
}

/*!
Implement the scalar product bewteen two vectors. Sum over the poducts beteween the corresponding elements of the vectors.
Attention, no check to condition v1.size == v2.size
*/
inline double scalprod_vv(std::vector<double> &v1, std::vector<double> &v2)
{
 unsigned int dimvet = (unsigned int)(v1.size());
 unsigned int i;
 double sum=0.;
 
 for(i=0; i<dimvet; ++i) sum = sum + v1[i] * v2[i];

 return sum;
}

/*!
Implement the scalar product bewteen two vectors. Sum over the poducts beteween the corresponding elements of the vector and the array.
Attention, no check to condition dimvet == v2.size
*/
inline double scalprod_av(unsigned int &dimvet, double *v1, std::vector<double> &v2)
{
 unsigned int i;
 double sum=0.;
 
 for(i=0; i<dimvet; ++i) sum = sum + v1[i] * v2[i];
 
 return sum;
}

/*!
Implement the scalar product bewteen two vectors. Sum over the poducts beteween the corresponding elements of the arrays
*/
template<typename FLOAT1, typename FLOAT2>
inline double scalprod_aa(unsigned int dimvet, FLOAT1 *v1, FLOAT2 *v2)
{
 unsigned int i;
 double sum=0.;
 
 for(i=0; i<dimvet; ++i) sum += v1[i] * v2[i];
 
 return sum;
}

/*!
Makes the difference between two arrays element by element
*/
inline void dif_aa(unsigned int &dimvet, double *min, double *sot, double *out)
{
 unsigned int i;
 for(i=0; i<dimvet; ++i) out[i] = min[i] - sot[i];
}

/*!
It makes the sum of two arrays element by element
*/
inline void sum_aa(unsigned int &dimvet, double *v1, double *v2, double *out)
{
 unsigned int i;
 for(i=0; i<dimvet; ++i) out[i] = v1[i] + v2[i];
}

/*!
It multiply two arrays element by element
*/
inline void prod_aa(unsigned int &dimvet, double *v1, double *v2, double *out)
{
 unsigned int i;
 for(i=0; i<dimvet; ++i) out[i] = v1[i] * v2[i];
}

/*!
It divides two arrays element by element
*/
inline void div_aa(unsigned int &dimvet, double *divid, double *divis, double *out)
{
 unsigned int i;
 for(i=0; i<dimvet; ++i) out[i] = divid[i] / divis[i];
}

/*!
It sums all the elements of the array vet
*/
template<typename T>
inline T integ_a(unsigned int &dimvet, T *vet)
{
 unsigned int i;
 T sum = 0;
 for(i=0; i<dimvet; ++i) sum = sum + *(vet + i);
 return sum;
}

/*! This function adds a bias of all elements of vet*/
template<typename FLOAT, typename INT>
inline void bias_a(FLOAT bias, INT dimvet, FLOAT *vet)
{
 for(INT i=0; i<dimvet; ++i) *(vet + i) += bias;
}

template<typename FLOAT, typename FLOAT1>
inline FLOAT mean(unsigned int dim, FLOAT1 *vet)
{
	unsigned int i;
	FLOAT mean = 0.;
	FLOAT coef1, coef2;

	i = 0;
	while(i<dim)
		{
			coef1 = (FLOAT)(i);
			++i;
			coef2 = 1 / ((FLOAT)(i));
			coef1 *= coef2;
			mean = mean * coef1 + vet[i] * coef2 ;
		}
	return mean;
}

template<typename UINT, typename FLOAT, typename FLOAT1>
inline void mean_std(UINT dim, FLOAT1 *vet, FLOAT &mean, FLOAT &std)
{
	UINT i;
	FLOAT coef1, coef2;

	mean = (FLOAT)(0.0);
	std = (FLOAT)(0.0);
	
	i = 0;
	while(i<dim)
		{
			coef1 = (FLOAT)(i);
			++i;
			coef2 = 1 / ((FLOAT)(i));
			coef1 *= coef2;
			mean = mean * coef1 + vet[i] * coef2 ;
			std = std * coef1 + vet[i] *vet[i]* coef2 ;
		}
	std = sqrt(std - mean*mean);
}

template<typename FLOAT, typename FLOAT1, typename FLOAT2>
inline FLOAT mean_w(unsigned int dim, FLOAT1 *vet, FLOAT2 *w)
{
	unsigned int i;
	FLOAT mean = 0.;
	FLOAT coef1, coef2, sumw;

	i = 0;
	sumw = 0;
	while(w[i] == 0) ++i;
	for( ; i<dim; ++i)
		{
			coef1 = sumw;
			coef2 = w[i];
			sumw += coef2;
			coef1 /= sumw;
			coef2 /= sumw;

			mean = mean * coef1 + vet[i] * coef2 ;
		}

	return mean;
}

/*!
vet wedge vet
*/
template <typename FLOAT1, typename FLOAT2, typename FLOAT3>
inline void wedge_vet(FLOAT1 *vet1, FLOAT2 *vet2, FLOAT3 *out)
{
	out[0] = vet1[1] * vet2[2] - vet1[2] * vet2[1];
	out[1] = vet1[2] * vet2[0] - vet1[0] * vet2[2];
	out[2] = vet1[0] * vet2[1] - vet1[1] * vet2[0];
}


template<typename FLOAT>
inline void cartesian_to_spherical(FLOAT c[], FLOAT s[])
{
	FLOAT r_xy;
	r_xy = (FLOAT)(sqrt(c[0]*c[0]+c[1]*c[1]));
	s[0] = (FLOAT)(sqrt(r_xy*r_xy+c[2]*c[2]));
	s[1] = (FLOAT)(atan2(c[2],r_xy));
	s[3] = (FLOAT)(atan2(c[1],c[0]));
}

template<typename FLOAT>
inline void spherical_to_cartesian(FLOAT s[], FLOAT c[])
{
	c[0]=s[0]*cos(s[1])*cos(s[2]);
	c[1]=s[0]*cos(s[1])*sin(s[2]);
	c[2]=s[0]*sin(s[1]);
}

/*!
normalize an array
*/
template <typename FLOAT1, typename FLOAT2>
inline FLOAT2 normalize_vet(unsigned int dim, FLOAT1 *in, FLOAT2 *out)
{
 unsigned int i;
 double norm = 0.;

	for(i = 0; i<dim; ++i) norm += in[i] * in[i];
 norm = (double)(1.0)/sqrt(norm);
	for(i=0; i<dim; ++i) out[i] = (FLOAT2)(in[i] * norm);
	return norm;
}

/*!
transpose a matrix
*/
template <typename FLOAT1, typename FLOAT2>
inline void mat_trasp(unsigned int dim_row, unsigned int dim_col, FLOAT1 *in, FLOAT2 *out)
{
 unsigned int i,j;
 
 for(i=0; i<dim_row; ++i)
  {
   for(j=0; j<dim_col; ++j)
    {
		out[j * dim_row + i] = in[i * dim_col + j];
    }
  }
 
}

/*!
Multiplies a matrix by a vector rows by coloum
*/
template <typename FLOAT1, typename FLOAT2, typename FLOAT3>
inline void mat_by_vet(unsigned int dim_row, unsigned int dim_col, FLOAT1 *mat, FLOAT2 *in, FLOAT3 *out)
{
 unsigned int i,j;
 FLOAT1 *row;
 double sum;
 
 for(i=0; i<dim_row; ++i)
  {
   row = &(mat[i * dim_col]);
   sum = 0;
//	std::cerr <<"row = "<<i<<"\t\tdim_row * i = "<<(i * dim_row)<<std::endl;
   for(j=0; j<dim_col; ++j)
    {
//	  std::cerr <<"(sum = "<<sum<<" ) + ( (row["<<j<<"]="<<*(row + j)<<") * (in["<<j<<"]="
//	            <<*(in + j)<<") = "<<(*(row + j) * *(in + j))<<")";
     sum += (double)(row[j] * in[j]);
//	  std::cerr <<"(sum = "<<sum<<" )"<<std::endl;
    }
   out[i] = sum;
  }
 
}

/*!
Multiplies a matrix by a vector rows by coloum
*/
template <typename INT, typename FLOAT1, typename FLOAT2,typename FLOAT3>
inline void mat_by_mat(unsigned int dim_row_1, unsigned int dim_col_1, FLOAT1 *mat1, unsigned int dim_col_2, FLOAT2 *mat2, FLOAT3 *mat_out)
{
 unsigned int i,j,k;
 FLOAT1 *row1;
 FLOAT2 *col2;
 FLOAT3 *row_out;
 double sum;
 
 for(j=0; j<dim_row_1; ++j)
  {
   row1 = &(mat1[j * dim_col_1]);
   row_out = &(mat_out[j * dim_col_2]);

   for(i=0; i<dim_col_2; ++i)
    {
		   col2 = mat2 + i;
					sum = 0.;
					for(k=0; k<dim_col_1; ++k) sum += (double)(row1[k] * col2[k * dim_col_2]);
					row_out[i] = sum;
    }
  }
 
}

/*!
Multiplies a matrix transposed by a vector rows by coloum
*/
template <typename INT, typename FLOAT1, typename FLOAT2, typename FLOAT3>
inline void mat_trasp_by_vet(INT dim_row, INT dim_col, FLOAT1 *mat, FLOAT2 *in, FLOAT3 *out)
{
 INT i,j;
 FLOAT1 *col;
 double sum;
 
 for(i=0; i<dim_col; ++i)
  {
   col = &(mat[i]);
   sum = 0;
//	std::cerr <<"row = "<<i<<"\t\tdim_row * i = "<<(i * dim_row)<<std::endl;
   for(j=0; j<dim_row; ++j)
    {
//	  std::cerr <<"(sum = "<<sum<<" ) + ( (row["<<j<<"]="<<*(row + j)<<") * (in["<<j<<"]="
//	            <<*(in + j)<<") = "<<(*(row + j) * *(in + j))<<")";
     sum += (double)(col[j * dim_col] * in[j]);
//	  std::cerr <<"(sum = "<<sum<<" )"<<std::endl;
    }
   out[i] = sum;
  }
 
}

/*!
Multiplies a matrix tranposed by a matrix rows by coloums
*/
template <typename INT, typename FLOAT1, typename FLOAT2,typename FLOAT3>
inline void mat_trasp_by_mat(INT dim_row_1, INT dim_col_1, FLOAT1 *mat1, INT dim_col_2, FLOAT2 *mat2, FLOAT3 *mat_out)
{
 INT i,j,k;
 FLOAT1 *col1;
 FLOAT2 *col2;
 FLOAT3 *row_out;
 double sum;
 
 for(j=0; j<dim_col_1; ++j)
  {
   col1 = &(mat1[j]);
   row_out = &(mat_out[j * dim_col_2]);
   
   for(i=0; i<dim_col_2; ++i)
    {
		col2 = &(mat2[i]);
		sum = 0.;
		for(k=0; k<dim_row_1; ++k) sum += (double)(col1[k * dim_col_1] * col2[k * dim_col_2]);
		row_out[i] = sum;
    }
  }
 
}

/*!
Multiplies a matrix by a matrix transposed rows by coloums
*/
template <typename INT, typename FLOAT1, typename FLOAT2,typename FLOAT3>
inline void mat_by_mat_trasp(INT dim_row_1, INT dim_col_1, FLOAT1 *mat1, INT dim_col_2, FLOAT2 *mat2, FLOAT3 *mat_out)
{
 INT i,j,k;
 FLOAT1 *row1;
 FLOAT2 *col2;
 FLOAT3 *row_out;
 double sum;
 
 for(j=0; j<dim_row_1; ++j)
  {
   row1 = &(mat1[j * dim_col_1]);
   row_out = &(mat_out[j * dim_col_2]);

   for(i=0; i<dim_col_2; ++i)
    {
 	   col2 = &(mat2[i * dim_col_2]);
					sum = 0.;
					for(k=0; k<dim_col_1; ++k) sum += (double)(row1[k] * col2[k]);
					row_out[i] = sum;
    }
  }
 
}

/*!
This function can be used to perform convolution of a big matrix with a small kernel
This avoid the complication and the delay that in such a situation are involved in
using the Fourier Transform.
The operation is not performed on the edges. In practice the operation is performed only in case
the kernel lays completely inside the bigger matrix. The old value is copied otherwise.
dimxk and dimyk must be odd values

warnings:
- INT must be a signed integer type
- out must have a different memory area than mat
*/
template<typename INT, typename FLOAT>
void real_space_convolve_mat(INT dimx, INT dimy, FLOAT mat[], INT dimxk, INT dimyk, FLOAT kernel[], FLOAT out[])
{
	int i,j;
	int ii,jj;
	int ibegin, iend;
	int jbegin,jend;
	FLOAT *mrow, *krow, *orow;
	FLOAT sum;
	
	if((0==dimxk%2)||(0==dimyk%2)) 
	{
		std::cerr <<"error in real_space_convolve_mat: kernel must have odd dimensions"<<std::endl;
		return;
	}
	
	
	ibegin = (dimxk-1)/2; iend = dimx - ibegin - 1;
	jbegin = (dimyk-1)/2; jend = dimy - jbegin - 1;
	
	
	for(j=jbegin; j<jend; ++j)
	{
		orow = &(out[j*dimx]);
		for(i=ibegin;i<iend;++i)
		{
			sum = 0;
			for(jj=-jbegin; jj<=jbegin; ++jj)
			{
				mrow = &(mat[(j+jj)*dimx]);
				krow = &(kernel[(jj+jbegin)*dimxk]);
				for(ii=-ibegin; ii<=ibegin; ++ii)
				{
					sum += krow[ii+ibegin]*mrow[ii+i];
				}
			}
			orow[i]=sum;
		}
	}
	//-------------------------------------------------------
	// copy of the edges
	// upper band
	for(j=0; j<jbegin; ++j)
	{
		orow = &(out[j*dimx]);
		mrow = &(mat[j*dimx]);
		for(i=0;i<dimx;++i) orow[i]=mrow[i];		
	}
// lower band
	for(j=jend; j<dimy; ++j)
	{
		orow = &(out[j*dimx]);
		mrow = &(mat[j*dimx]);
		for(i=0;i<dimx;++i) orow[i]=mrow[i];		
	}	
//lateral bands
	for(j=jbegin; j<jend; ++j)
	{
		orow = &(out[j*dimx]);
		mrow = &(mat[j*dimx]);
		for(i=0   ;i<ibegin;++i) orow[i]=mrow[i];
		for(i=iend;i<dimx  ;++i) orow[i]=mrow[i];		
	}		
	
}

/*!
Solving linear equation using reduction methods. return value is the determinant of the matrix mat
matrix mat must be square.
vet==out is allowed
*/
template<typename FLOAT, typename FLOAT1, typename FLOAT2, typename FLOAT3>
FLOAT equlin(unsigned int dim, FLOAT1 *_mat, FLOAT2 *vet, FLOAT3 *out)
{
	unsigned int i,j,k;
	unsigned int dimx = dim + 1;
	unsigned int dimy = dim;
	unsigned int dimy_m_1 = dimy - 1;
	
	unsigned int posmin,posmax;
	FLOAT max,min;
	FLOAT det;
	FLOAT diag, col_ele;
	FLOAT *mat = new FLOAT[dimx * dimy];
	FLOAT *reduction_row = new FLOAT[dimx];
	FLOAT *mat_row, *_mat_row;
	
	// copy of the enlarged matrix
	for(j=0; j<dim; ++j)
		{
			mat_row  = &(mat[j * dimx]);
			_mat_row = &(_mat[j * dim]);
			for(i=0; i<dim; ++i)
				{
					mat_row[i] = _mat_row[i];
				}
			mat_row[i] = vet[j];
		}
	
//		std::cerr <<std::endl;
//	cerr_mat(dimy, dimx, mat);
	
	// initial pivoting
	for(j=0; j<dimy; ++j)
		{
			mat_row = &(mat[j * dimx]);
			// ----pivoting---------------------------
			maxminary(dimx, mat_row, max, posmax, min, posmin);
			max = fabs(max); min = fabs(min);
			if(max<min) max = min;
			for(i=0; i<dimx; ++i) mat_row[i] /= max;
			// ----end-pivoting---------------------------
		}
	
//		std::cerr <<std::endl;
//		cerr_mat(dimy, dimx, mat);
//			std::cerr <<std::endl;
//				std::cerr <<std::endl;
	
	for(j=0; j<dimy_m_1; ++j)
		{
			mat_row = &(mat[j * dimx]);
			
			// avoid zero division and check for null determinanants
			k = j;
			do
				{
					diag = mat[k * dimx + k];
					k++;
				} while( (k<dimy) && (fabs(diag) < EPS_INV_MAT));
//			if(fabs(diag) < EPS_INV_MAT) return 0;
			k--;
			if( j != k)
				{
					for(i=0; i<dimx; ++i) mat_row[i] += mat[k * dimx + i];
					diag = mat_row[j];
					
//						std::cerr <<std::endl;
	//					cerr_mat(dimy, dimx, mat);
				}
			
			// reduction process
			for(i=0; i<dimx; ++i) reduction_row[i] = mat_row[i] / diag;
			for(k=j+1; k<dimy; ++k)
				{
					mat_row = &(mat[k * dimx]);
					col_ele = mat_row[j];
					for(i=0; i<dimx; ++i) mat_row[i] -= col_ele * reduction_row[i];
				}
				
//					std::cerr <<std::endl;
//					cerr_mat(dimy, dimx, mat);
		}
	
/*	if(fabs(*(mat + dimy_m_1 * dimx + dimy_m_1)) < EPS_INV_MAT)
		{
			for(i=0; i<dim; ++i) *(out + i) = 0;
		 return 0;
		}
*/	
	det = 1;
	for(j=0; j<dim; ++j) det *= mat[j * dimx + j];
	
	//ATT!!!  unsigned trick in the next line!!!! BE CAREFUL!!!!!-----------
	for(j=dimy_m_1; j<dimy; j--)
	//---------------------------
		{
			mat_row = mat + j * dimx;
			out[j] = mat_row[dimy];
			for(i=dimy_m_1; i>j; i--)
				out[j] -= mat_row[i] * out[i];
			out[j] /= mat_row[j];
		}
//	std::cerr <<std::endl;
//	cerr_ary("out[i] = ", dimy, out);
	
	delete[] mat;
	delete[] reduction_row;
	
	return det;
}

/*!

x[matrix] * par[colum vector] = y[coulum vector]
*/
template<typename FLOAT>
bool minsqrfit(unsigned int dimy, unsigned int dimpar, FLOAT *x, FLOAT *y, FLOAT *par, FLOAT *err_par)
{
	unsigned int i,j;
	FLOAT *x_trasp = new FLOAT[dimy *dimpar];
	FLOAT *row,*col;
	FLOAT *mat = new FLOAT[dimpar * dimpar];
	FLOAT *x_trasp_by_y = new FLOAT[dimpar];
	bool ret;
	
	for(j=0; j<dimy; ++j)
		{
			row = &(x[j * dimpar]);
			col = &(x_trasp[j]);
			for(i=0; i<dimpar; ++i) col[i * dimy] = row[i];
		}
	mat_by_mat<FLOAT>(dimpar, dimy, x_trasp, dimpar, x, mat);
	mat_by_vet<FLOAT>(dimpar, dimy, x_trasp, y, x_trasp_by_y);
	
//	print_mat(std::cerr, dimpar, dimpar, mat);

	if( 0.0000000000000001 > fabs(equlin<FLOAT>(dimpar, mat, x_trasp_by_y, par)))
		 ret = false;
	else
			ret = true;
	
	delete[] x_trasp;
	delete[] mat;
	delete[] x_trasp_by_y;
	
	return ret;
}


/*!
first float is the precision of the matrix inversion and general calculus
*/
template<typename FLOAT>
bool poly_fit(unsigned int dim, FLOAT *x, FLOAT *y, unsigned int dimpar, FLOAT *powers, FLOAT *par, FLOAT *err_par)
{
	unsigned int i,j;
	FLOAT scale, val;
	FLOAT _max, _min;
	unsigned int posmax, posmin;
	FLOAT *xx = new FLOAT[dim];
	FLOAT *mat = new FLOAT[dimpar * dim];
	FLOAT *row;
	bool ret;
	
	//maxminary(dim, x, _max, posmax, _min, posmin);
//	scale = 0.5 * (_max - _min);
	
//	for(i=0; i<dim; ++i) xx[i] = x[i]/scale;
	
	for(j=0; j<dim; ++j)
		{
			row = &(mat[j * dimpar]);
			for (i = 0; i < dimpar; ++i)
			{
				row[i] = std::pow(x[j], powers[i]);
	//			std::cerr <<row[i]<<"\t";
			}
	//		std::cerr << std::endl;
		}
	
	ret = minsqrfit<FLOAT>(dim, dimpar, mat, y, par, err_par);
	
//	for(i=0; i<dimpar; ++i)
//		{
//			val = pow(scale, powers[j]);
//			par[j] *= val;
//			err_par[j] *= val;
//		}

		
	delete[] xx;
	delete[] mat;
	
	return ret;
}

template<typename FLOAT>
inline bool poly_fit(unsigned int dim, FLOAT *x, FLOAT *y, int max_power, FLOAT *par, FLOAT *err_par)
{
	if (max_power == 0) return false;
	int i, j, step;
	int dimpar = abs(max_power) + 1;
	FLOAT *powers = new FLOAT[dimpar];
	bool ret;
	
	if(max_power > 0) step = 1;
	else              step = -1;
	
	for(i=0, j=0; i<dimpar; ++i, j+=step) powers[i] = (FLOAT)(j);


	//for (j = 0; j < dim; ++j)
	//{
	//	std::cerr << "[" << j << "] -> " << x[j] << "\t " << y[j] << std::endl;
	//}


	ret = poly_fit(dim, x, y, dimpar, powers, par, err_par); 
	
	delete[] powers;
	
	return ret;
}

template<typename FLOAT>
inline bool linear_fit(unsigned int dim, FLOAT *x, FLOAT *y, FLOAT *par, FLOAT *err_par)
{
	return (poly_fit<FLOAT>(dim, x, y, 1, par, err_par));
}

template<typename FLOAT, typename FLOAT1, typename FLOAT2, typename FLOAT3>
inline bool quadratic_fit(unsigned int dim, FLOAT1 *x, FLOAT2 *y, FLOAT3 *par, FLOAT3 *err_par)
{
	return (poly_fit<FLOAT>(dim, x, y, 2, par, err_par));
}

/*!
y = par[0] * exp(par[1] * x)
*/
template<typename FLOAT, typename FLOAT1, typename FLOAT2, typename FLOAT3>
inline bool exp_fit(unsigned int dim, FLOAT1 *x, FLOAT2 *y, FLOAT3 *par, FLOAT3 *err_par)
{
	unsigned int i;
	bool ret;
	FLOAT *log_y = new FLOAT[dim];
	
	for(i=0; i<dim; ++i) *(log_y + i) = log(*(y+i));
	
	ret = poly_fit<FLOAT>(dim, x, log_y, 1, par, err_par);
	
	if(*par <= 0)
		{
			ret = false;
		}
	else
		{
			*par = log(*par);
			*err_par = *err_par / (fabs(*par));
		}
	
	delete[] log_y;
	
	return ret;
}

/*!
This function find the best circles which approximate the input data
dimpoints - number of points contained in points
points - points in space that approximate the circle. Points are stored (x[0], y[0], x[1], y[1]....)
x0 - return coordinates of the middle point of the circle
r - radius of the circle
*/
template<typename FLOAT1, typename FLOAT2>
void circonference_fit(unsigned int dimpoints, FLOAT1 *points, FLOAT2 *x0, FLOAT2 &r)
{
 unsigned int i,j;
 unsigned int dimpar = 3;
 FLOAT2 *matx,*y;
 FLOAT2 *par, *errpar;
 FLOAT2 xx[2];
 FLOAT2 avgx[2], avgr;
 FLOAT2 coef1, coef2;
 FLOAT2 *pp;
 
 par = new FLOAT2[dimpar];
 errpar = new FLOAT2[dimpar];
 matx = new FLOAT2[dimpar*dimpoints];
 y = new FLOAT2[dimpoints];
 pp = new FLOAT2[dimpoints*2];
 
 //------------------------------------------------------------------
 // averaging and normalizing so to have the circumference centered and of unitary radius
	
	avgx[0] = 0.0; avgx[1] = 0.0;
 for(i=0; i<dimpoints; ++i)
 {
	coef2 = (FLOAT2)(1.0)/(FLOAT2)(i+1) ;
	coef1 = coef2*(FLOAT2)(i);
	for(j=0; j<2; ++j)
		avgx[j] = coef1 * avgx[j] + coef2 * (FLOAT2)(points[2*i+j]);
 }
 
 for(i=0; i<dimpoints; ++i)	for(j=0; j<2; ++j)	pp[2*i+j]=(FLOAT2)(points[2*i+j])-avgx[j];
 
	avgr = 0.0;
 for(i=0; i<dimpoints; ++i)
 {
	coef2 = (FLOAT2)(1.0)/(FLOAT2)(i+1) ;
	coef1 = coef2*(FLOAT2)(i);
	avgr = coef1 * avgr + coef2 * sqrt(pp[2*i]*pp[2*i]+pp[2*i+1]*pp[2*i+1]);
 } 

 if(0.00000000001>avgr) return;
 for(i=0; i<2*dimpoints; ++i) pp[i] /=avgr;
 
//	x0[0] = avgx[0]; x0[1] = avgx[1]; r = avgr;
// std::cout <<"avgx = "<<avgx[0]<<"\tavgy = "<<avgx[1]<<"\tavgr = "<<avgr<<std::endl;
 
//--------------------------------------------------
 for(i=0; i<dimpoints; ++i)
 {
	 matx[dimpar*i    ] = (FLOAT2)(1.0);
	 
	 for(j=0; j<2; ++j)
	 {
		xx[j] = pp[2*i+j];
		matx[dimpar*i + (j+1)] = (FLOAT2)(2.0)*xx[j];
	 }

	 y[i]=xx[0]*xx[0]+xx[1]*xx[1];
 }
 
 
 if(true==minsqrfit<FLOAT2>(dimpoints, dimpar, matx, y, par, errpar))
 {
	 for(j=0; j<2; ++j) x0[j]=par[j+1];
	 r = par[0] + x0[0]*x0[0] + x0[1]*x0[1];
	 
	 if(0<=r) r=sqrt(r);
	 
	 r *= avgr;
	 for(j=0; j<2; ++j) x0[j]=(x0[j]*avgr)+avgx[j];
 }
else
 {

	 for(j=0; j<2; ++j) x0[j]=(FLOAT2)(0.0);
	 r = (FLOAT2)(0.0);
 }
	
 delete[] par;
 delete[] errpar;
 delete[] matx;
 delete[] y;
	delete[] pp;
}



/*!It gives back the convolution between a vector vetin and square impulse of lenght dimpulse. Square impulse is
normalized to 1*/
template<typename INT, typename FLOAT>
inline void pulse_smooth(INT dimvet, INT dimpulse, FLOAT *vetin, FLOAT *vetout)
{
 lldiv_t d;
 INT i;
 INT dimpulse_d_2;
 INT dimpulse_p_1;
 circular_index<INT> i_front, i_back;
 FLOAT *vet = new FLOAT[dimvet];
 FLOAT mean;
 FLOAT one_div_dimpulse;
 
 d = div((INT)dimpulse,(INT)(2));
 if(d.rem == 0)
  {
   dimpulse_d_2 = dimpulse / 2;
   dimpulse_p_1 = dimpulse + 1;
  }
 else
  {
   dimpulse_d_2 = (dimpulse-1) / 2;
   dimpulse_p_1 = dimpulse;
  }
  

// std::cerr <<"dimpulse_d_2 = "<<dimpulse_d_2<<std::endl;
// std::cerr <<"dimpulse_p_1 = "<<dimpulse_p_1<<std::endl;
// std::cerr <<"dimvet = "<<dimvet<<std::endl;
 i_front.init(dimvet-1);
 i_back.init(dimvet-1);
 one_div_dimpulse = 1./((FLOAT)(dimpulse_p_1));
 for(i=0; i<dimvet; ++i)
  {
   vet[i] = vetin[i] * one_div_dimpulse;
//   std::cerr <<"(vet["<<i<<"] = "<<*(vet + i)<<") = (vetin["<<i<<"] = "<<*(vetin + i)<<") * (one_div_dimpulse = "<<one_div_dimpulse<<")"<<std::endl;
  }
 
 mean = 0; i_front = - (long)(dimpulse_d_2);
 for(i=0; i<dimpulse_p_1; ++i)
  {

   mean = mean + vet[i_front.i];
//   std::cerr <<"mean += (vet["<<i_front.i<<"] = "<<*(vet + i_front.i)<<") = "<<mean<<std::endl;
   ++i_front;
  }
  
 i_front = dimpulse_d_2 + 1; i_back = - (long)(dimpulse_d_2);
 for(i=0; i<dimvet; ++i)
  {
   vetout[i] = mean;
   mean = mean - vet[i_back.i] + vet[i_front.i];
   ++i_front; ++i_back;
  }
	delete[] vet;
}


/*!It gives back the convolution between a vector vetin and square impulse of lenght dimpulse. Square impulse is
normalized to 1*/
template<typename INT, typename FLOAT>
inline void kernel_smooth(INT dimvet, INT dimkernel, FLOAT *kernel, FLOAT *vetin, FLOAT *vetout)
{
	lldiv_t d;
	INT i,j;
	INT dimpulse_d_2;
	INT dimpulse_p_1;
	circular_index<INT> i_c;
	
	// std::cerr <<"dimpulse_d_2 = "<<dimpulse_d_2<<std::endl;
	// std::cerr <<"dimpulse_p_1 = "<<dimpulse_p_1<<std::endl;
	// std::cerr <<"dimvet = "<<dimvet<<std::endl;
	i_c.init(dimvet - 1);
	
	for (i = 0; i < dimvet; ++i)
	{
		vetout[i] = 0.0;
		for (j = 0; j < dimkernel; ++j)
		{
			i_c=j+i;
	//		k = i_c();
			vetout[i] += vetin[i_c()] * kernel[j];
		}
		
	}


}




/*!This function calculates a rotation matrix starting from a vector.
Vector direction will be rotation axis while modulus the rotation angle.
the function is thought in 3D
*/
template<typename FLOAT>
inline void rotation(FLOAT *vet, FLOAT *rot)
{
	int i, indmin;
	FLOAT *vx,*vy,*vz;
	FLOAT mat[9], mattmp[9];
	FLOAT co,so;
	FLOAT component,omega;

	// new basis where the rotation should be performed
	vx = &(mat[0]);	vy = &(mat[3]);	vz = &(mat[6]);

	// calculating versor of rotation
	omega = sqrt(vet[0]*vet[0]+vet[1]*vet[1]+vet[2]*vet[2]);
	if(fabs(omega)<0.00000001)
	{
		std::cerr <<"warning in rotation: rotation angle smaller than 10^-7"<<std::endl;
	}
	for(i=0; i<3; ++i) vz[i]=vet[i]/omega;
	
	// find the axis with least projection
	indmin = 0;
	component = vz[0];
	for(i=1; i<3; ++i)
	{
		if(component > vz[i])
		{
			indmin = i;
			component = vz[i];
		}
	}
	
	// building up other axes
	for(i=0; i<3; ++i) vy[i]=(FLOAT)(0.0);
	vy[indmin] = (FLOAT)(1.0);

	wedge_vet(vy, vz, vx);
	normalize_vet(3, vx, vx);
	wedge_vet(vz, vx, vy);

	//normalization

	co = cos(omega);
	so = sin(omega);
	rot[0] =     co      ; rot[1] =    -so      ; rot[2] = (FLOAT)(0.0);
	rot[3] =     so      ; rot[4] =     co      ; rot[5] = (FLOAT)(0.0);
	rot[6] = (FLOAT)(0.0); rot[7] = (FLOAT)(0.0); rot[8] = (FLOAT)(0.0);

	mat_by_mat<unsigned int, FLOAT>(3,3,rot,3,mat,mattmp);
	mat_trasp_by_mat<unsigned int, FLOAT>(3,3,mat,3,mattmp,rot);
}

/*!calculates rotation matrices for a new nord into the reference system*/
template<typename FLOAT>
void nord_to_rotation(FLOAT new_nord[], FLOAT mat_new_to_old[], FLOAT mat_old_to_new[])
{
	unsigned int i;
	FLOAT nord[3], new_nord_norm[3];
	FLOAT omega[3];
	FLOAT theta;
	FLOAT sum;
	
	for(i=0;i<3;++i) nord[i]=(FLOAT)(0.0);
	nord[2]=(FLOAT)(1.0);
	
	normalize_vet(3, new_nord, new_nord_norm);
	wedge_vet(nord, new_nord_norm, omega);
	theta = normalize_vet(3, omega, omega);
	if(fabs(theta)<0.0000001)
	{
		std::cerr <<"warning: rotation angle <10^-7"<<std::endl;
	}
	theta = asin(theta);
	for(i=0;i<3;++i) omega[i] *= theta;
	
	rotation(omega, mat_old_to_new);
	mat_trasp(3, 3, mat_old_to_new, mat_new_to_old);
}


/*!
Find intersection point between a plane and a line.
the function is thought in 3D
*/
template<typename FLOAT>
bool intersect_plane_line(FLOAT plane_pos[], FLOAT plane_normal[], FLOAT line_pos[], FLOAT line_direction[],  FLOAT pointout[], FLOAT matcollisionout[] = NULL)
{
	unsigned int i,j;
	FLOAT max,min;
	FLOAT matt[9];
	FLOAT *mat;
	FLOAT matsystem[9], vetsystem[3];
	FLOAT lined[3];
	FLOAT pnorm[3];
	FLOAT coef[3];
	double sum,val;
	
	if(NULL == matcollisionout) mat = matt;
	else                        mat = matcollisionout;
	
	normalize_vet(3,plane_normal,pnorm);
	normalize_vet(3,line_direction,lined);
	
	if(0.0000000001>scalprod_aa<FLOAT,FLOAT,FLOAT>(3,plane_normal,line_direction)) return false;
	
	sum=0;
	for(i=0; i<3; ++i)
	{
		val = pnorm[i]-lined[i];
		sum+=val*val;
	}		
	
	for(i=0; i<3; ++i) mat[6+i]=pnorm[i];
	if(0.000000000001<sum)
	{
		for(i=0; i<3; ++i) coef[i]=lined[i];
	}
	else
	{
		for(i=0; i<3; ++i) coef[i]=(FLOAT)(0.0);
		maxminary((unsigned int)(3),pnorm,max,i,min,j);
		coef[j]=(FLOAT)(1.0);			
	}
	
	wedge_vet(pnorm, coef, mat);
	normalize_vet(3, mat, mat);
	wedge_vet(&(mat[6]), mat, &(mat[3]));
	normalize_vet(3, &(mat[3]), &(mat[3]));
	
	for(i=0; i<3; ++i)
	{
		matsystem[    i*3] = mat[    i];
		matsystem[1 + i*3] = mat[3 + i];
		matsystem[2 + i*3] = -lined[i];
		vetsystem[i] = line_pos[i]-plane_pos[i];
	}
	
	equlin<double,double,double,double>((unsigned int)(3), matsystem, vetsystem, coef);
	for(i=0; i<3; ++i) pointout[i] = coef[3] * lined[i] + line_pos[i];
	
	return true;
}

template<typename UINT, typename FLOAT>
void gradient_numerical(void(*f)(FLOAT *x, FLOAT *y), UINT dimx, UINT dimy, FLOAT *x, FLOAT *xtmp, FLOAT *y, FLOAT *ytmp, FLOAT *dx, FLOAT *grad)
{
	UINT i,j;
	FLOAT *gradcol;

	f(x, y);
	for (i = 0; i < dimx; ++i) xtmp[i] = x[i];
	for (j = 0; j < dimx; ++j)
	{
		xtmp[j] = x[j] + dx[j];
		f(xtmp, ytmp);

		gradcol = &(grad[j]);
		for (i = 0; i < dimy; ++i)
		{
			gradcol[i*dimx] = (ytmp[i] - y[i]) / dx;
		}

		// restore previous state of xtmp
		xtmp[j] = x[j];
	}

	
}

#endif
