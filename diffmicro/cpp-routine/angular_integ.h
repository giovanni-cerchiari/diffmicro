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

#ifndef _ANGULAR_INTEGRATOR_H_
#define _ANGULAR_INTEGRATOR_H_

// choose your standar for index and size as you wish
#define INDEX size_t
#define GREEK_PI 3.1415926535897932384626433832795

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>

/*!
this functor calculates the angular integral of a square matrix from (0,0) corner.
The mean position of the (0,0) element of the matrix is intended to be in position (0,0)

The output of the integral is one dimensional vector of dimension m_dimr. The space
beetween each element of this output vector have to be intended equal to the distance
between to adiacent pixel in the orizontal and vertical directions.

The integral is evaluated as a weighted sum over all the pixels.
The two more distance points inside a pixel of side 1 (the two that stays on the diagonal)
have a separating distance of sqrt(2). This means that in the most comlicated case
a pixel can overlap at maximum three different distances in the output array.
We say that a pixel "falls" at a certein distance if it should be taken into
account in evaluating the azimuthal integral for a determinate output position.

First of all the three falling distances for each pixel are calculated.
In practice a matrix of index m_ind is prepared. The matrix stores for each pixel
the shortest falling index position in the output array.
Then a matrix of wheights is calculated. This weights are the sub-area of each pixel
that falls inside a determined annulus. For each pixel are store in sequence three
wheights and respctively the first wheight correspond to the index stored in
m_ind[pixel_index], the second in (m_ind[pixel_index]+1) and the third in (m_ind[pixel_index]+2)

The sub-areas are calculated in a not random motecarlo alghoritm. Each pixel are subdivided
in subpixels. The ratio beetween the number of sub-pixels that fall into a determined annular
region above the total number of sub-pixels is considered as the weight of the pixel for
the specific anular distance.

*/
template<typename FLOAT_OUT, typename FLOAT_IN>
class angular_integrator
{
public:
	angular_integrator(INDEX dim_side = 1, INDEX pixel_division = 8)
		{
			m_dimside = 0;
			m_size = 0;
			m_w = NULL;
			m_ind = NULL;

			m_px_div = 0;
	
	  m_dimr = 0;
	  m_wr = NULL;

			this->init(dim_side, pixel_division);
		}

	angular_integrator(angular_integrator &ang_int)
	{
		this->operator=(ang_int);
	}

	~angular_integrator()
		{
			this->clear();
		}
 /*! clear variables and pointers*/
	void clear()
		{
			m_dimside = 0;
			m_size = 0;
			if(m_w != NULL)
				{
					delete[] m_w;
					m_w = NULL;
				}
			if(m_ind != NULL)
				{
					delete[] m_ind;
					m_ind = NULL;
				}

			m_px_div = 0;

			m_dimr = 0;
			m_wr = NULL;
		}

	/*!
	initialization.
	- dim_side is the side in elements of the square matrix of which the angular integral should
	evaluated
	- pixel_division are the sub-division per side of a pixel that are used to evaluate the weights matrix m_w
	*/
	void init(INDEX dim_side, INDEX pixel_division = 8)
		{
			if( ((m_dimside == dim_side) && (m_px_div == pixel_division)) || (pixel_division == 0) || (dim_side == 0) ) return;
			
			INDEX i,j,jj,ii,k;

			this->clear();

			// do not exchange the two following statements!!!
			m_px_div = pixel_division;
			this->def_alloc(dim_side);
	
			// initialization of matrices

			for(j=0; j<m_dimside; ++j)
				{
					jj = j * m_dimside;
					for(i=j; i<m_dimside; ++i)
						{
							ii = i * m_dimside;

							rad_to_ind_w(i, j, m_ind[jj + i], &(m_w[3*(jj + i)]));

							m_ind[ii + j] = m_ind[jj + i];
							for(k=0; k<3; ++k) m_w[3*(ii + j) + k] = m_w[3*(jj + i) + k];
						}
				}

			// (0,0) recalculation to avoid numerical errors like negative indices
			m_ind[0] = 0;
			m_w[0] = (FLOAT_OUT)(GREEK_PI * 0.5 * 0.5);
			m_w[1] = (FLOAT_OUT)(1. - GREEK_PI * 0.5 * 0.5);
			m_w[2] = 0.;

			FLOAT_IN *mat_one = new FLOAT_IN[m_size];
			for(i=0; i<m_size; ++i) mat_one[i] = 1.;
			this->operator()(mat_one, m_wr);

			delete[] mat_one;

			/*
			for(i=0; i<m_size; ++i)
				{
					if(m_ind[i]+2 >= m_dimr)
						r=0;
				}
			*/
		}

	/*! copy operator*/
	void operator=(angular_integrator &ang_int)
	{
		if(m_dimside != ang_int.m_dimside)
			{
				this->clear();
				this->def_alloc(ang_int.m_dimside);
			}

		this->m_px_div = ang_int.m_px_div;

		std::memcpy(this->m_w, ang_int.m_w, 3 * m_size * sizeof(FLOAT_OUT));
		std::memcpy(this->m_wr, ang_int.m_wr, m_dimr * sizeof(FLOAT_OUT));
		std::memcpy(this->m_ind, ang_int.m_ind, m_size * sizeof(INDEX));

	}

	/*!
	This is the functor function. As you see the initialization should be done before calling the functor
	and the memory are of the integral should be also prepared in advance
	*/
	void operator()(FLOAT_IN mat[], FLOAT_OUT integ[])
	{
		INDEX i, j;
		FLOAT_OUT mat_val;

		memset(integ, 0, m_dimr * sizeof(FLOAT_OUT));
		for(i=0; i<m_size; ++i)
			{
				mat_val = (FLOAT_OUT)(mat[i]);
				for(j=0; j<3; ++j)
					integ[m_ind[i] + j] += m_w[3*i + j] * mat_val;
			}
	}

	INDEX dimside(){return m_dimside;}
	INDEX size(){return m_size;}
	FLOAT_OUT* w(){return m_w;}
	INDEX* ind(){return m_ind;}

	INDEX pixel_division() {return m_px_div;}

	INDEX dimr(){return m_dimr;}
	FLOAT_OUT* wr(){return m_wr;}

protected:

	//! square matrix side dimension
	INDEX m_dimside;
	//! = m_dimside * m_dimside
	INDEX m_size;
	//! matrix of weights to calculate the azimuthal integral
	FLOAT_OUT *m_w;
	//! matrix that stores for each pixel the shortest falling index position in the output array.
	INDEX *m_ind;

	//! pixel sub-division per side. Sub-division of pixel are used to evaluate the weights matrix m_w
	INDEX m_px_div;

	//! output array dimension
	INDEX m_dimr;
	/*!
	this array contains the azimuthal integral of a matrix full of ones. This means that this are the
	azimuthal weights integrate and srinked on a possible output array. The division of any output
	array with this one trasform the angular integral into an angular average.
*/
	FLOAT_OUT *m_wr;

	/*!
	allocation of the necessary space of the functor
	*/
	void def_alloc(INDEX dim_side)
	{
		INDEX x,y,dimr_m_3;
		FLOAT_OUT ww[3];
			//-----------------
			// sizes definitions
			m_dimside = dim_side;
			m_size = m_dimside * m_dimside;

			// dimr definition
		x = m_dimside - 1;
		y = m_dimside - 1;
		rad_to_ind_w(x, y, dimr_m_3, ww);



			//DOVREBBE ESSERE GIUSTO MA HO UN DUBBIO
			m_dimr = dimr_m_3 + 3;



			//--------------------
			// allocation
			m_w = new FLOAT_OUT[3*m_size];
			m_ind = new INDEX[m_size];
			m_wr = new FLOAT_OUT[m_dimr];
			//-------------------
	}

	/*!
	From pixel indices to its m_ind and weigh
	*/
	inline void rad_to_ind_w(INDEX &x, INDEX &y, INDEX &ind, FLOAT_OUT w[])
	{
		INDEX i,j;
		INDEX counter[3] = {0,0,0};
		FLOAT_OUT step;
		FLOAT_OUT xx,yy;
		FLOAT_OUT left, up;
		FLOAT_OUT r2;
		FLOAT_OUT rint2_1, rint2_2;

		step = (FLOAT_OUT)(1./(FLOAT_OUT)(m_px_div));
		left = (FLOAT_OUT)((FLOAT_OUT)(x) - 0.5 + step/2.);
		up = (FLOAT_OUT)((FLOAT_OUT)(y) - 0.5 + step/2.);

		xx = (FLOAT_OUT)(x*x);
		yy = (FLOAT_OUT)(y*y);
		r2 = sqrt(xx + yy);
		rint2_1 = std::floor(r2);
		rint2_2 = std::ceil(r2);

		if( (r2 - rint2_1) < (rint2_2 - r2))
			{
				ind = (INDEX)(rint2_1) - 1;
				rint2_2 = (FLOAT_OUT)(rint2_1 + 0.5);
				rint2_1 -= (FLOAT_OUT)(0.5);
			}
		else
			{
				ind = (INDEX)(rint2_2) - 1;
				rint2_1 = (FLOAT_OUT)(rint2_2 - 0.5);
				rint2_2 += (FLOAT_OUT)(0.5);
			}

		rint2_1 *= rint2_1;
		rint2_2 *= rint2_2;
		for(j=0; j<m_px_div; ++j)
			{
				yy = up + step * (FLOAT_OUT)(j);
				yy *= yy;
				for(i=0; i<m_px_div; ++i)
					{
						xx = left + step * (FLOAT_OUT)(j);
						r2 = yy + xx * xx;

						if(r2 < rint2_1)
							{
								++counter[0];
							}
						else
							{
								if(r2 < rint2_2)	++counter[1];
								else	            ++counter[2];
							}//if(r2 < rint2_1)

					}//for(i=0; i<m_px_div; ++i)
			}//for(j=0; j<m_px_div; ++j)

		xx = (FLOAT_OUT)(1./(FLOAT_OUT)(m_px_div * m_px_div));
		for(j=0; j<3; ++j) w[j] = (FLOAT_OUT)(counter[j]) * xx;
	}

	private:
};

/*!
To perform angular integral on all 4 quadrants folding of the quadrants is necessary.
We use here FFT convention for element placing and we assume even dimensions
dimxout = dimxin/2
dimyout = dimyin/2
*/
template<typename FLOAT>
void average_quadrants(INDEX dimxin, INDEX dimyin, FLOAT *in, FLOAT *out)
{
	if ((1 == dimxin % 2) || (1 == dimyin % 2))return;
	INDEX i, j, ii, jj;
	INDEX dimxout, dimyout;

	dimxout = dimxin / 2;
	dimyout = dimyin / 2;

	out[0] = in[0];

	for (i = 1; i < dimxout; ++i)
	{
		ii = dimxin - i;
		out[i] = 0.5*(in[i] + in[ii]);
	}

	for (j = 1; j < dimyout; ++j)
	{
		jj = dimyin - j;
		out[j*dimxout] = 0.5*(in[j*dimxin] + in[jj*dimxin]);
	}

	// all
	for (j = 1; j < dimyout; ++j)
	{
		jj = dimyin - j;
		for (i = 1; i < dimxout; ++i)
		{
			ii = dimxin - i;
			out[j*dimxout + i] = 0.25*(in[j*dimxin + i] + in[jj*dimxin + i] + in[j*dimxin + ii] + in[jj*dimxin + ii]);
		}
	}
	

}

template<typename FLOAT>
void angular_average(INDEX dimside, FLOAT *mat, INDEX &dimout, FLOAT *avg)
{
	angular_integrator<FLOAT, FLOAT> integrator;
	INDEX dim,i;
	FLOAT *quad;
	FLOAT *norm;

	memset(avg, 0, sizeof(FLOAT)*dimout);

	dim = dimside / 2;

	quad = new FLOAT[dim*dim];

	average_quadrants(dimside, dimside, mat, quad);

	integrator.init(dim);
	dimout = integrator.dimr();

	norm = new FLOAT[dimout];

	integrator(mat, avg);

	dim = dim*dim;
	for (i = 0; i < dim; ++i) mat[i] = (FLOAT)(1.0);
	integrator(mat, norm);

	for (i = 0; i < dimout; ++i) avg[i] /= norm[i];

	delete[] quad;
	delete[] norm;
}



#endif
