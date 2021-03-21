
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


/*!
This file contains global variable and macro that are used inside the program
*/

#ifndef _GLOBAL_DEFINE_H_
#define _GLOBAL_DEFINE_H_

#include <cufft.h>
#include "fftw3.h"
#include "wtypes.h"

#define MY_DEBUG 1

#define GREEK_PI 3.14159265358979323846264338327950288419716939937510

//#define INDEX size_t
#define INDEX unsigned __int64
#define SIGNED_INDEX __int64

#define MY_REAL double

#define N_MAX_THREADS 8

#define STORE_REAL double

#define	STORE_TYPE_FLOAT		1
#define	STORE_TYPE_DOUBLE		2
#define STORE_TYPE STORE_TYPE_DOUBLE

#define HARDWARE_CPU	0
#define HARDWARE_GPU	1

//---------------------------------------------------------
// CUDA types
// list of possible types
#define	CUFFT_TYPE_FLOAT		1
#define	CUFFT_TYPE_DOUBLE		2
#define	CUFFT_TYPE	CUFFT_TYPE_DOUBLE

// configurable type selection
//#define	CUFFT_REAL	cufftReal
//#define	CUFFT_COMPLEX	cufftComplex
//#define CUFFT_TYPE	CUFFT_TYPE_FLOAT

#define	CUFFT_REAL	cufftDoubleReal
#define	CUFFT_COMPLEX	cufftDoubleComplex
//#define	CUFFT_REAL	double
//#define	CUFFT_COMPLEX	fftw_complex
#define CUFFT_TYPE	CUFFT_TYPE_DOUBLE

// FFTW types
// list of possible types
#define	FFTW_TYPE_FLOAT		1
#define	FFTW_TYPE_DOUBLE	2
#define	FFTW_TYPE		FFTW_TYPE_DOUBLE

#define FFTW_REAL double
#define FFTW_COMPLEX fftw_complex

//#define FFTW_REAL float
//#define FFTW_COMPLEX fftwf_complex

// other macros, with explicit comparisons!
//#if (CUFFT_TYPE == CUFFT_TYPE_FLOAT)
//#elif (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
//#else
//	#error Unknown CUDA type selected
//#endif

/*! This function could be used to delete any generic pointer, but it does not work always.*/
template<typename TYPE>
inline void my_delete(TYPE *ptr)
{
	if(NULL != ptr)
	{
		delete[] ptr;
		ptr = NULL;
	}
}

/*! This function gets the horizontal and vertical screen sizes in pixel*/
template <typename IND>
void GetDesktopResolution(IND& horizontal, IND& vertical)
{
	RECT desktop;
	// Get a handle to the desktop window
	const HWND hDesktop = GetDesktopWindow();
	// Get the size of screen to the variable desktop
	GetWindowRect(hDesktop, &desktop);
	// The top left corner will have coordinates (0,0)
	// and the bottom right corner will have coordinates
	// (horizontal, vertical)
	horizontal = desktop.right;
	vertical = desktop.bottom;
}

#endif