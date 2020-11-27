
/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 12/2015
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

#ifndef _histogram_H_
#define _histogram_H_

#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>

#include "tensor.h"
#include "my_math.h"
#include "prep_vet.h"

/*!
This class implements a multidimensional histogram.
The coordinates of each bin are intended as central coordinate of the bin
Step is the physical dimension of each bin.
Bins are intended as squared and equal in shape one to the other.
Data feeded into the histogram are snapped to the closet bin without extending their influence over other bins.
*/
template<typename TYPE, typename WEIGHT>
class histogram
{
	public:
	
	histogram()
	{
		start = NULL;
		step = NULL;
	}
	
	~histogram()
	{
		this->clear();
	}
	
	/*!
	Memory clear function
	*/
	void clear()
	{

		if(NULL != start)
		{
			delete[] start;
			start = NULL;
		}
		if(NULL != step)
		{
			delete[] step;
			step = NULL;
		}
		freq.clear();
	}

	/*!
	Memory initialization of the histogram. Frequency memory area is set to zero.
	*/
	bool init(INDEX dimensions, INDEX binnumber[], TYPE startbin[], TYPE binstep[])
	{
		INDEX i;

		this->clear();

		start = new TYPE[dimensions];
		step = new TYPE[dimensions];

		for(i=0; i<dimensions; ++i)
		{
			start[i] = startbin[i];
			step[i] = binstep[i];
		}

		freq.resize(dimensions, binnumber);

		zero();
		return true;
	}
	
	/*!
	This method put to zero each value of the histogram
	*/
	void zero()
	{
		memset(freq.ptr(), 0, sizeof(WEIGHT) * freq.size());
	}

	/*!
	Use this method to feed histogram with new values.
	This method can be called more than once to update histograms values.
	For each mathing the corresponding frequency value is incremented by the corresponding weight (w[]) 
	*/
 void update(INDEX dimvet, TYPE *x, WEIGHT *w)
	{
	
		INDEX space_dimension = freq.dimspace();
		INDEX i,j,k;
		INDEX histogram_size = freq.size();
		INDEX *jump = freq.jump();
		WEIGHT *hist_ptr = freq.ptr();
		TYPE *xx;
		
		for(i=0; i<dimvet; ++i)
			{
				k = 0;
				xx = &(x[i * space_dimension]);
				for(j=0; j<space_dimension; ++j)
				{
					k += (INDEX)(floor( ( ( (TYPE)(xx[j]) - (TYPE)(start[j]) ) / step[j]) + 0.5 ) ) * jump[j];
				}
				// INDEX should be unsigned!!!
				if(k >= histogram_size) k = histogram_size - 1;
			 
			 hist_ptr[k] += w[i];
			}

	}
	/*!
	Use this method to feed histogram with new values.
	This method can be called more than once to update histograms values.
	For each mathing the corresponding frequency value is incremented by 1 
	*/
	void update(INDEX dimvet, TYPE *x)
	{
	
		INDEX space_dimension = freq.dimspace();
		INDEX i,j,k;
		INDEX histogram_size = freq.size();
		INDEX *jump = freq.jump();
		WEIGHT *hist_ptr = freq.ptr();
		TYPE *xx;
		
		for(i=0; i<dimvet; ++i)
			{
				k = 0;
				xx = &(x[i * space_dimension]);
				for(j=0; j<space_dimension; ++j)
				{
					k += (INDEX)(floor( ( ( (TYPE)(xx[j]) - (TYPE)(start[j]) ) / step[j]) + 0.5 ) ) * jump[j];
				}
				// INDEX should be unsigned!!!
				if(k >= histogram_size) k = histogram_size - 1;
			 
			 hist_ptr[k] += 1;
			}

	}

	/*!
	this method writes histogram to a binary file. Suggested extension .hst
	*/
	bool write_binaryfile(std::string filename)
	{
		FILE *fid;
		fid = fopen(filename.c_str(), "wb");
		if(NULL == fid)
		{
			std::cerr <<"error attempting writing histogram to "<<filename<<std::endl;
			return false;
		}
	 write_binary(fid);

		fclose(fid);
		return true;
	}

	/*!
	write histogram to an already opened binary file
	*/
	void write_binary(FILE *fid)
	{
		unsigned int i;
		INDEX *ddim;
		unsigned int dim,dimspace;
		ddim = freq.dim();

		dimspace = freq.dimspace();
		fwrite(&dimspace, sizeof(unsigned int), 1, fid);
		for(i=0; i<dimspace; ++i)
		{
			dim = (unsigned int)(ddim[i]);
			fwrite(&dim, sizeof(unsigned int), 1, fid);
		}
		dim = sizeof(TYPE);
		fwrite(&dim, sizeof(unsigned int), 1, fid);
		dim = sizeof(WEIGHT);
		fwrite(&dim, sizeof(unsigned int), 1, fid);
		fwrite(start, sizeof(TYPE), freq.dimspace(), fid);
		fwrite(step, sizeof(TYPE), freq.dimspace(), fid);
		fwrite(freq.ptr(), sizeof(WEIGHT), freq.size(), fid);
	}

	/*!
	this method reads histogram from a binary file according to write_binary
	*/
	bool read_binaryfile(std::string filename)
	{
		FILE *fid;
		fid = fopen(filename.c_str(), "rb");
		if(NULL == fid)
		{
			std::cerr <<"error attempting reading histogram from "<<filename<<std::endl;
			return false;
		}
	 if(false == read_binary(fid))
		{
			std::cerr <<"error attempting reading histogram from "<<filename<<std::endl;
			return false;
		}

		fclose(fid);
		return true;
	}

		/*!
	this method reads histogram from an already opened binary file according to write_binary
	*/
	bool read_binary(FILE *fid)
	{
		INDEX i;
		unsigned int ddim;
		unsigned int *uint_dim;
		INDEX *dim;
		TYPE *step;
		TYPE *start;
		unsigned int dim,dimspace;
		unsigned int sizeof_type, sizeof_weight;
		bool ret = true;

		fread(&dimspace, sizeof(unsigned int), 1, fid);

		uint_dim = new unsigned int[dimspace];
		dim = new INDEX[dimspace];
		start = new TYPE[dimspace];
		step = new TYPE[dimspace];

		fread(&ddim, sizeof(unsigned int), dimspace, fid);
		for(i=0; i<dimspace; ++i) dim[i] = (INDEX)(ddim[i]);

		fread(&sizeof_type, sizeof(unsigned int), 1, fid);
		fread(&sizeof_weight, sizeof(unsigned int), 1, fid);

		if((sizeof(TYPE)!=sizeof_type)||(sizeof(WEIGHT)!=sizeof_weight))
		{
			std::cerr <<"error reading histogram from bynary: size of variables does not match"<<std::endl;

			ret = false;
		}

		if(true == ret)
		{
			fread(start, sizeof(TYPE), dimspace, fid);
			fread(step, sizeof(TYPE), dimspace, fid);
			this->init(dimspace, dim, start, step);
			fread(freq.ptr(), sizeof(WEIGHT), freq.size(), fid);
		}

		delete[] uint_dim;
		delete[]	dim;
		delete[]	start;
		delete[]	step;

		return ret;
	}


	/*
	void normalize()
	{
		unsigned int i;
		long double sum_freq = 0;;
		for(i=0; i<dim; ++i) sum_freq += *(freq_rel + i);
		if(sum_freq <= 0)
			{
				std::cerr <<"error trying to normalize the hysogram"<<std::endl;
				return;
			}
		for(i=0; i<dim; ++i) *(freq_rel + i) /= sum_freq;
	}
	
	void assign(unsigned int _dim, TYPE *_x, unsigned int *freq)
	{
		if(false == resize(_dim)) return;
		
		unsigned int i;
		FLOAT one_over_N;
		
		N = 0;
		for(i=0; i<dim; ++i) N += *(freq + i);
		one_over_N = 1/((FLOAT)(one_over_N));
		for(i=0; i<dim; ++i) *(freq_rel + i) = *(freq + i) * one_over_N;
		for(i=0; i<dim; ++i) *(x + i) = *(_x + i);	
	}
	
	void assign(unsigned int _dim, TYPE *_x, FLOAT *_freq_rel, unsigned int _N)
	{
		if(false == resize(_dim)) return;
		
		unsigned int i;
		FLOAT one_over_N;
		
		N = _N;
		for(i=0; i<dim; ++i) *(freq_rel + i) = *(_freq_rel + i);
		for(i=0; i<dim; ++i) *(x + i) = *(_x + i);	
	}
	
	void mode_min(FLOAT &mode, TYPE &x_mode, unsigned int &index_mode,
														 FLOAT &min,  TYPE &x_min,  unsigned int &index_min)
	{
		maxminary(dim, freq_rel, mode, index_mode, min, index_min);
		mode *= (FLOAT)(N);
		min  *= (FLOAT)(N);
		x_mode = *(x + index_mode);
		x_min = *(x + index_min);
	}
	
	void mean(FLOAT &mean_x, FLOAT &rms_x,  FLOAT &mean_freq, FLOAT &rms_freq)
	{
		unsigned int i, start_i;
		FLOAT coef1, coef2;
		FLOAT sum_freq;
		FLOAT dif;
		
		// start_i is the first index with not null frequency
		start_i = 0;
		while(*(freq_rel + start_i) == 0) ++start_i;
		
		// frequency
		i = 0;
		while(i<=dim)
			{
				coef1 = (FLOAT)(i);
				++i;
				coef2 = 1/(FLOAT)(i);
				coef1 *= coef2;
				
				mean_freq = mean_freq * coef1 + *(freq_rel + i - 1) * coef2; 
			}
		
		// rms frequency
		i = 0;
		while(i<=dim)
			{
				coef1 = (FLOAT)(i);
				++i;
				coef2 = 1/(FLOAT)(i);
				coef1 *= coef2;
				
				dif = *(freq_rel + i - 1) - mean_freq;
				rms_freq = rms_freq * coef1 + dif * dif * coef2; 
			}
		rms_freq = sqrt(rms_freq);

		// x
		sum_freq = 0;
		mean_x = 0.; // not delete this statement!!!
		for(i=start_i; i<dim; ++i)
			{
				if(*(freq_rel + i) > 0)
					{
						coef1 = sum_freq;
						sum_freq += *(freq_rel + i);
						coef1 /= sum_freq;
						coef2 = *(freq_rel + i)/sum_freq;
				
						mean_x = mean_x * coef1 + *(x + i) * coef2; 
					}
			}
		
		// rms x
		sum_freq = 0;
		for(i=start_i; i<dim; ++i)
			{
				coef1 = sum_freq;
				sum_freq += *(freq_rel + i);
				coef1 /= sum_freq;
				coef2 = *(freq_rel + i)/sum_freq;

				dif = *(x + i) - mean_x;
				rms_x = rms_x * coef1 + dif * dif * coef2; 
			}
		rms_x = sqrt(rms_x);

	}
	*/

	// central position of the first bin
	TYPE *start;
	// bin size
	TYPE *step;
	// counts
	tensor<WEIGHT> freq;


};


#endif
