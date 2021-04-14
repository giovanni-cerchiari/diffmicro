/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 8/2011

*/

/*
Copyright: Mojtaba Norouzisadeh, Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
e-mail: norouzi.mojtaba.sade@gmail.com

update: 05/2020 - 09/2020
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
This functions and classes are written for diffmicro.exe application.
*/


#include "stdafx.h"
#include "power_spectra.h"
#include "diffmicro_display.h"
#include "radial_storage_lut.h"
#include<Engine.h>

fifo_min::fifo_min()
{
	size = 0;
	current_size = 0;
	file_index = NULL;
	flg_valid_image = NULL;
	ind_fifo = 0;
	ind_file = 0;
	dist_bias = 0;
	dist_map = NULL;
	m_im = NULL;
}
fifo_min::~fifo_min()
{
	this->clear();
}

void fifo_min::clear()
{
	size = 0;
	current_size = 0;
	ind_fifo = 0;
	ind_file = 0;
	dist_bias = 0;

	if(file_index != NULL)
	{
		delete[] file_index;
		file_index = NULL;
	}
	if(dist_map != NULL)
	{
		delete[] dist_map;
		dist_map = NULL;
	}
	if(m_im != NULL)
	{
		delete[] m_im;
		m_im = NULL;
	}
	if (flg_valid_image != NULL)
	{
		delete[] flg_valid_image;
		flg_valid_image = NULL;
	}
}

void fifo_min::init(INDEX _size_fifo, INDEX image_size)
{
	this->clear();
	size = _size_fifo;
	file_index = new INDEX[size];
	flg_valid_image = new bool[size];
	dist_map = new INDEX[size];
	m_im = new unsigned short[image_size];
}

void fifo_min::reset(INDEX _dist_bias)
{
	current_size = 0;
	ind_fifo = 0;
	ind_file = 0;
	dist_bias = _dist_bias;
}

INDEX fifo_min::load_next(INDEX ind_sot, std::vector<bool> &flg_valid_image, STORE_REAL image_mean[])
{
	INDEX i, dimx, dimy;
	INDEX ret;
	// association ind_fifo <-> ind_file
	file_index[ind_fifo] = ind_file;
	// recording if it is a valid image
	flg_valid_image[ind_fifo] = flg_valid_image[ind_file];
	// filling FIFO
	if (current_size != size) ++current_size;

	//calculating the association ind_fifo <-> temporal delay
	for (i = 0; i<current_size; ++i)
		dist_map[i] = ind_sot - file_index[i] - dist_bias;

	if (true == flg_valid_image[ind_file])
	{
		// load image into FIFO
		time_reading_from_disk.start();
		//load_binary_image(useri.file_list[ind_file], dimy, dimx, true, m_im, flg_display_read);
		load_image(useri.file_list[ind_file], dimy, dimx, true, m_im, flg_display_read);
		time_reading_from_disk.stop();
		if (1 == image_to_dev(ind_fifo, image_mean[ind_file], m_im, false))
		{
			std::cerr << "warning an image (index = " << ind_file << " appears to be null" << std::endl
				<< "\t=> average has been put equal to one to avoid bad divisions" << std::endl;
		}
	}
	else
	{
		image_mean[ind_file] = (STORE_REAL)(0.0);
	}

	if (true == flg_valid_image[ind_sot])
	{
		// load image to be subtracted to all images store into the FIFO
		time_reading_from_disk.start();
		//load_binary_image(useri.file_list[ind_sot], dimy, dimx, true, m_im, flg_display_read);
		load_image(useri.file_list[ind_sot], dimy, dimx, true, m_im, flg_display_read);
		time_reading_from_disk.stop();
		if (1 == image_to_dev(-1, image_mean[ind_sot], m_im, false))
		{
			std::cerr << "warning an image (index = " << ind_sot << " appears to be null" << std::endl
				<< "\t=> average has been put equal to one to avoid bad divisions" << std::endl;
		}
	}
	else
	{
		image_mean[ind_sot] = (STORE_REAL)(0.0);
	}

	//refreshing FIFO indices
	++ind_fifo;
	if (ind_fifo == size) ind_fifo = 0;

	ret = ind_file;
	++ind_file;
	return ret;
}

/*!
This function calculates a group of average of power spectra. See bool calc_power_spectra(INDEX dimy, INDEX dimx) for further details
*/
void calc_diagonal(int k,INDEX starting_index, unsigned int power_spectra_avg_counter[], fifo_min &fifo, INDEX nimages, STORE_REAL image_mean[], bool flg_debug)
{
	
	int flg_continue_averages = 1;
	// initializing array

	fifo.reset(starting_index);
	INDEX i = starting_index;
	while ((i<nimages) && (1 == flg_continue_averages))
	{
		fifo.load_next(i, useri.flg_valid_image, image_mean);
		if (true == useri.flg_valid_image[i])
		{
			time_differences.start();
			flg_continue_averages = diff_autocorr(useri.file_list.size(),fifo.current_size, power_spectra_avg_counter, i, fifo.file_index, fifo.dist_map, useri.flg_valid_image, useri.n_pw_averages);
			time_differences.stop();
			n_computed_diff += fifo.current_size;
		}

		++i;
	}
	

}

/*!
This function unload th averaged power spectra from the video card and start the parallel
evaluation of the azimuthal average with the CPU.
*/
void pw_save_and_azth_avg(unsigned int *ram_radial_lut, INDEX starting_index,
	                         INDEX npw, INDEX dimr, MY_REAL azh_avgs[], STORE_REAL ram_power_spectra[] )
{
	if(true == useri.flg_write_azimuthal_avg)
		{
			power_spectra_from_dev(npw, ram_radial_lut, ram_power_spectra, starting_index, false);

			// writing power spectra to file
			if(true == useri.flg_write_power_spectra)
				{
					time_writing_on_disk.start();
					write_power_spectra(starting_index, npw, s_load_image.dimy/2, s_load_image.dimx, ram_power_spectra);
					time_writing_on_disk.stop();
				}

			power_spectra_to_azhavg(npw, ram_power_spectra, &(azh_avgs[starting_index * dimr]));

			//--------------------------------------------------------------------
			// Hi,yes, always me, welcome Mojtaba!
			if (useri.flg_graph_mode)
			{
				for (int kk = 0; kk < npw; ++kk)
				{
					display_update_azhavg(dimr, &(azh_avgs[(kk + starting_index) * dimr]));
					waitkeyboard(10);
				}
			}
			//-------------------------------------------------------------------
		}
	else
		{
			// loading power spectra to ram and eventually write them to file
			power_spectra_from_dev(npw, ram_radial_lut ,ram_power_spectra, starting_index, false);
			if(true == useri.flg_write_power_spectra)
				{
					time_writing_on_disk.start();
					write_power_spectra(starting_index, npw, s_load_image.dimy / 2, s_load_image.dimx, ram_power_spectra);
					time_writing_on_disk.stop();
				}
		}
}

void pw_azth_avg(unsigned int* lut,
	INDEX npw, INDEX dimr, MY_REAL azh_avgs[], STORE_REAL ram_power_spectra[]) {

	for (int i = 0; i < npw; i++) {

		memset(ram_power_spectra, 0, sizeof(STORE_REAL) * s_load_image.dim / 2);

		for (int j = 0; j < s_power_spectra.dim; j++) {
			ram_power_spectra[lut[j]] = (STORE_REAL)dev_images_cpu[j+i* s_power_spectra.dim][0];
		}

		power_spectra_to_azhavg(1.0, ram_power_spectra, &(azh_avgs[i * dimr]));

	}

}


int diff_autocorr(INDEX dim_file_list, INDEX dim_fifo, unsigned int* counter_avg, INDEX ind_sot, INDEX* file_index, INDEX* dist_map, std::vector<bool>& flg_valid_image_fifo, INDEX n_max_avg)
{
	INDEX j, ind_dist;
	CUFFT_REAL coef1, coef2;
	int ret = 0;
	unsigned char* exe_map_row = &(execution_map[ind_sot * dim_file_list]);

	for (j = 0; j < dim_fifo; ++j)
	{
		//----------------------------------------------------------------------------
		// half power spectrum with average
		ind_dist = dist_map[j];

		// if number maximum number of averages have been reached for this power spectrum
		// no difference and refreshing of the power spectrum is made
		if ((n_max_avg != counter_avg[ind_dist]) && (true == flg_valid_image_fifo[j]))
		{
			coef1 = (CUFFT_REAL)(counter_avg[ind_dist]);
			++counter_avg[ind_dist];
			coef2 = (CUFFT_REAL)(1. / (CUFFT_REAL)(counter_avg[ind_dist]));
			coef1 *= coef2;

			diff_power_spectrum_to_avg(coef1, coef2, j, ind_dist);

			exe_map_row[file_index[j]] += (unsigned char)(1);

			ret = 1;
		}

	}

	return ret;
}

bool calc_power_spectra(INDEX dimy, INDEX dimx)
{
		//------------------------------------------------------------------
	// declaration

	// allocation
	INDEX j, jj;
	INDEX nimages = (INDEX)(useri.file_list.size());
	
	int percentexe = 0;
	bool flg_continue = true;
	STORE_REAL *ram_power_spectra(NULL);
	STORE_REAL *image_mean = new STORE_REAL[nimages];
	memset(image_mean, 0, nimages * sizeof(STORE_REAL));
	unsigned int* power_spectra_avg_counter(NULL);
	INDEX useri_dist_max;

//	unsigned int *ram_radial_lut;

	// thread init
	INDEX dimr;
	MY_REAL *azh_avgs = NULL;

	std::string filename;

	//--------------------------------------------------------------------
	// radial look up table (lut) preparation
	radial_lut_dimr_from_max_freq(dimy/2, dimx, (float)(useri.frequency_max), dim_radial_lut);
	ram_radial_lut = new unsigned int[dim_radial_lut];
	radial_lut_length = dim_radial_lut;
	radial_storage_lut(dimy/2, dimx, dim_radial_lut, ram_radial_lut);

	//--------------------------------------------------------------------
	if (useri.flg_graph_mode)
	{
		display_radial_lut(dimy, dimx, dim_radial_lut, ram_radial_lut);
		waitkeyboard(10);
	}
	//--------------------------------------------------------------------


	//---------------------------------------------------------------------
	// cuda allocation
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	if(0 != diffmicro_allocation(useri.execution_mode, nimages, dimy, dimx, dim_radial_lut, ram_radial_lut))
		{
			std::cerr <<"error in allocating device memory"<<std::endl;
			return false;
		}

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	//----------------------------------------------------------------
	// Azimuthal averages initialization
	if((dimx == dimy) && (true == useri.flg_write_azimuthal_avg))
		{
			power_spectra_to_azhavg_init(useri.nthread, dimx/2, dimr);
			azh_avgs = new MY_REAL[dimr * nimages];

			//first distance is never calculated and, so, assigned to zero
			memset(azh_avgs, 0, sizeof(STORE_REAL) * dimr);
		}
	else
		{
			useri.flg_write_azimuthal_avg = false;
		}

	//-------------------------------------------------------------------
	// printing execution main parameters
	std::cout <<"number of images : "<<nimages<<std::endl;
	std::cout <<"image size : "<<dimx<<" x "<<dimy<<std::endl;
	std::cout <<"number of storable FFTs : "<<s_fft_images.numerosity<<std::endl<<std::endl;

	std::cout <<"calculating power spectra ..."<<std::endl;
	//---------------------------------------------------------------------
	// this we need later to show the azimuthal averages
	if (useri.flg_graph_mode)
		display_open_azhavg(dimr);
	switch (useri.execution_mode)
	{
	case DIFFMICRO_MODE_FIFO:

		//-------------------------------------------------------------------
		// group averages initialization
		power_spectra_avg_counter = new unsigned int[useri.file_list.size()];
		ram_power_spectra = new STORE_REAL[s_power_spectra.numerosity * dimx * dimy / 2];

		calc_power_spectra_ALL(nimages, dimx, dimy, dimr,ram_power_spectra, azh_avgs);

		//calc_power_spectra_autocorr2(nimages, dimx, dimy, dimr, ram_power_spectra, azh_avgs);


		memset(power_spectra_avg_counter, 0, useri.file_list.size() * sizeof(unsigned int));
		//calc_power_spectra_fifo(nimages, useri_dist_max, image_mean, dimr,
			         //           power_spectra_avg_counter, ram_power_spectra, azh_avgs);
		if (useri.flg_graph_mode)
			display_average(useri.file_list.size(), power_spectra_avg_counter, &average_counter_window, &average_counter_scatter, &x_avg_counter, &y_avg_counter, "avg counter");

		break;
	case DIFFMICRO_MODE_TIMECORRELATION:
		power_spectra_avg_counter = new unsigned int[useri.file_list.size()];
		ram_power_spectra = new STORE_REAL[s_power_spectra.numerosity * dimx * dimy / 2];

		//calc_power_spectra_autocorr2(nimages, dimx, dimy, dimr, ram_power_spectra, azh_avgs);
		calc_power_spectra_autocorr(dimy, dimx, nimages, image_mean, dimr, power_spectra_avg_counter,
				ram_power_spectra, azh_avgs);
		for (INDEX i = 0; i < nimages; i++) 

		{

			power_spectra_avg_counter[i] = nimages - i;

		}
		break;
	default:
		break;
	}
	if (useri.flg_graph_mode)
	{
		display_average(useri.file_list.size(), image_mean, &average_window, &average_scatter, &x_avg, &y_avg, "average intensity");
		waitkeyboard(100);
	}

	std::cout << "... power spectra calculated" << std::endl;

	//--------------------------------------------------------------------------
	// write to file section
	time_writing_on_disk.start();
	if(true == useri.flg_write_images_avg)
			write_vet_to_file(useri.images_avg_filename, nimages, image_mean);

	if(true == useri.flg_write_azimuthal_avg)
			write_mat_to_file(useri.azimuthal_avg_filename, nimages, dimr, azh_avgs);
	time_writing_on_disk.stop();

	useri.write_time_sequence_to_file();

 //--------------------------------------------------------------------------
	// memory free operations
	diffmicro_deallocation();
	power_spectra_to_azhavg_free();
	if(NULL != power_spectra_avg_counter)
		delete[] power_spectra_avg_counter;
	if (NULL != ram_power_spectra)
		delete[] ram_power_spectra;
	delete[] image_mean;
	delete[] ram_radial_lut;
	if(true == useri.flg_write_azimuthal_avg)
		{
			delete[] azh_avgs;
		}

	return true;
}

void plot_dynamics(INDEX dimx){

	Engine* m_Engine = engOpen(NULL);
	int d = (int)dimx;
	int fr = (int)useri.frequency_max;
	std::string pp = useri.azimuthal_avg_filename;

	//mxArray* path_ = mxCreateDoubleMatrix(100, 1, mxSTRING);
	mxArray* dimx_crop = mxCreateDoubleMatrix(1, 1, mxREAL);
	mxArray* fr_max = mxCreateDoubleMatrix(1, 1, mxREAL);
	mxArray* path_dinamics;
	//memcpy((void*)mxGetPr(dimx_crop), (void*)d, sizeof(int) );
	//engPutVariable(m_Engine, "dimx_cro", dimx_crop);
	//char* pt = mxGetPr(dimx_crop);

	double* d_crop = mxGetPr(dimx_crop);
	d_crop[0] = (double)d;
	engPutVariable(m_Engine, "dimx_cr", dimx_crop);

	double* f = mxGetPr(fr_max);
	f[0] = (double)fr;
	engPutVariable(m_Engine, "frequency_max", fr_max);

	path_dinamics = mxCreateString(useri.azimuthal_avg_filename.c_str());
	engPutVariable(m_Engine, "p_dinamics", path_dinamics);

	engEvalString(m_Engine, "fid = fopen(p_dinamics, 'rb');");
	engEvalString(m_Engine, "size = fread(fid, 1, 'uint16');");
	engEvalString(m_Engine, "dimy = fread(fid, 1, 'uint16');");
	engEvalString(m_Engine, "dimx = fread(fid, 1, 'uint16');");
	engEvalString(m_Engine, "azhavgs = fread(fid, [dimx, dimy], 'double');");
	engEvalString(m_Engine, "dimazh = [dimx dimy];");
	engEvalString(m_Engine, "figure(1);");
	engEvalString(m_Engine, "left = 2;");
	engEvalString(m_Engine, "right = min(dimx_cr(1) / 2, frequency_max);");
	engEvalString(m_Engine, "xx = (left:right);");
	engEvalString(m_Engine, "loglog(xx, (azhavgs(left:right, 1)), 'k');");
	engEvalString(m_Engine, "hold on;");
	engEvalString(m_Engine, "n = 2;");
	engEvalString(m_Engine, "while (n <= dimazh(2) / 4) loglog(xx, (azhavgs(left:right, n)), 'k');loglog(xx, (azhavgs(left:right, n + 1)), 'r');loglog(xx, (azhavgs(left:right, n + 2)), 'g');loglog(xx, (azhavgs(left:right, n + 3)), 'b');n = n + 4;end");

}
void calc_power_spectra_ALL(INDEX nimages, INDEX dimy, INDEX dimx, INDEX& dimr, STORE_REAL* ram_power_spectra, MY_REAL* azh_avgs) {
	
	//cufftHandle plan_;
	//memory = dev_im_gpu_, s_load_image.memory_tot + dev_fft_gpu_, s_fft.memory_tot + CUFFT_COMPLEX *dev_images_gpu(NULL)
	CUFFT_COMPLEX* dev_fft_gpu_(NULL);
	unsigned short* dev_im_gpu_(NULL);
	unsigned short* m_im_;
	int alloc_status_li_, alloc_status_fft_;
	alloc_status_li_ = cudaMalloc(&dev_im_gpu_, s_load_image.memory_tot);
	if (cudaSuccess != alloc_status_li_) {

		std::cout << "ERROR: cudaMalloc " << std::endl;
	}

	alloc_status_fft_ = cudaMalloc(&dev_fft_gpu_, s_fft.memory_tot);
	if (cudaSuccess != alloc_status_fft_) {

		std::cout << "ERROR: cudaMalloc dev_fft_gpu_" << std::endl;
	}
	/*alloc_status_fft_ = cudaMalloc(&dev_fft_gpu_, s_fft.memory_tot);
	if (cudaSuccess != alloc_status_fft_) {

		std::cout << "ERROR: cudaMalloc " << std::endl;
	}*/

/*#if (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
	cufftExecZ2Z(plan_, dev_fft_gpu_, dev_fft_gpu_, CUFFT_FORWARD);
#else
#error Unknown CUDA type selected
#endif*/
	m_im_ = new unsigned short[dimx * dimy];
	for (int i = 0; i < nimages; i++) {

		time_reading_from_disk.start();
		load_image(useri.file_list[i], dimy, dimx, true, m_im_, flg_display_read);
		time_reading_from_disk.stop();

		time_from_host_to_device.start();
		cudaMemcpy(dev_im_gpu_, m_im_, s_load_image.memory_one, cudaMemcpyHostToDevice);
		time_from_host_to_device.stop();

		// from image to complex matrix
		Image_to_complex_matrix(dev_im_gpu_, dev_fft_gpu_,i);

	}

	/*CUFFT_COMPLEX* tmp_display_cpx_(NULL);

	tmp_display_cpx_ = new CUFFT_COMPLEX[nimages*s_fft_images.dim];

	cudaMemcpy(tmp_display_cpx_, dev_images_gpu, nimages * s_fft_images.memory_one, cudaMemcpyDeviceToHost);
	for (int ii = 0; ii < nimages * s_fft_images.dim; ++ii)
		std::cout << tmp_display_cpx_[ii].x << "  + i " << tmp_display_cpx_[ii].y << std::endl;*/

	//memory = dev_ALLfft_diff, s_power_spectra.memory_one * (nimages - 1) + STORE_REAL *dev_power_spectra_gpu(NULL);

	/*
	int accessible = 0;
	cudaDeviceCanAccessPeer(&accessible, 0, 0);
	*/

	int device_count = 0;
	cudaGetDeviceCount(&device_count);

	std::vector<std::thread> threads;
	for (int i = 0; i < device_count; i++) {

		threads.push_back(std::thread([&, i]() {

			int cudaStatus = cudaSetDevice(i);

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
				// goto Error;
			}

			Calc_structure_function(nimages,i, device_count);

			}));

	}
	for (auto& thread : threads)
		thread.join();

	pw_save_and_azth_avg(ram_radial_lut, 0, nimages, dimr, azh_avgs, ram_power_spectra);
	
}
void calc_power_spectra_fifo(INDEX nimages, INDEX &useri_dist_max, STORE_REAL* image_mean, INDEX &dimr, unsigned int* power_spectra_avg_counter, STORE_REAL* ram_power_spectra, MY_REAL* azh_avgs)
{
	INDEX j, jj;
	lldiv_t group;
	INDEX n_group, group_rem;
	fifo_min fifo;
	INDEX npw;

	if (nimages <= useri.dist_max)
	{
		useri_dist_max = nimages - 1;
	}
	else
	{
		useri_dist_max = useri.dist_max;
	}

	group = std::div((long long)(useri_dist_max), (long long)(s_power_spectra.numerosity));
	n_group = (INDEX)(group.quot);
	n_groups_reload = n_group + 1;
	group_rem = (INDEX)(group.rem);
	//----------------------------------------------------------------------
	if (useri.flg_graph_mode)
		display_execution_map(useri.file_list.size(), execution_map);
	//---------------------------------------------------------------------
	fifo.init(s_power_spectra.numerosity, s_load_image.dim);

	std::vector<std::thread> threads;
	for (j = 0; j < n_group; ++j)
	{
		//threads.push_back(std::thread([&, j]() {
			jj = 1 + j * s_power_spectra.numerosity;
			// calculating a group of power spectra
			int k = 0;
			calc_diagonal(k,jj, &(power_spectra_avg_counter[jj]), fifo, nimages, image_mean);
			// npw is the number of power spectra to be transfered from the video card to the ram
			npw = s_power_spectra.numerosity;
			pw_save_and_azth_avg( ram_radial_lut, jj, npw, dimr, azh_avgs, ram_power_spectra);
			//}));
	}
	//for (auto& thread : threads)
		//thread.join();
	//----------------------------------------------------------------------
	// out of group elements
	if (0 != group_rem)
	{
		fifo.init(group_rem, s_load_image.dim);
		jj = 1 + j * s_power_spectra.numerosity;

		//for (int k = 0; k < 1; k++) {

			//threads.push_back(std::thread([&, k]() {
		int k = 0;
				calc_diagonal(k,jj, &(power_spectra_avg_counter[jj]), fifo, nimages, image_mean);

				//}));
		//}
		
		for (auto& thread : threads)
			thread.join();
		npw = group_rem;
		pw_save_and_azth_avg(ram_radial_lut, jj, npw, dimr, azh_avgs, ram_power_spectra); //use this for version 3
	}
	//-------------------------------------------------------------------------
	update_execution_map(useri.file_list.size(), execution_map);
}

void calc_power_spectra_autocorr(INDEX dimy, INDEX dimx, INDEX nimages, STORE_REAL* image_mean, INDEX& dimr, unsigned int* power_spectra_avg_counter, STORE_REAL* ram_power_spectra, MY_REAL* azh_avgs)
{
	INDEX j,i;
	lldiv_t group;
	INDEX n_group, group_rem;
	fifo_min fifo;
	INDEX npw;
	unsigned short* m_im = new unsigned short[dimy*dimx];

	group = std::div((long long)(s_fft_images.dim), (long long)(s_time_series.numerosity));
	n_group = (INDEX)(group.quot);
	n_groups_reload = n_group + 1;
	group_rem = (INDEX)(group.rem);
	//-----------------------------------------
	std::cout<< n_group<<std::endl;
	for (i = 0; i < n_group; ++i)
	{
		// 1) load all images to dev and move the fft values into the storage
		load_memory_for_time_correlation(dimx, dimy, nimages, i* s_time_series.numerosity, s_time_series.numerosity, m_im, image_mean);
		// 2) analyze the time series 
		time_series_analysis();
		// 3) unrol time series and save partial results
		save_partial_timeseries(nimages, i, s_time_series.numerosity, ram_power_spectra);
	}
	if (0 != group_rem)
	{
		// 1) load all images to dev and move the fft values into the storage
		load_memory_for_time_correlation(dimx, dimy, nimages, i* s_time_series.numerosity, group_rem, m_im, image_mean);

		// 2) analyze the time series 
		time_series_analysis();

		// 3) unrol time series and save partial results
		save_partial_timeseries(nimages, i, group_rem, ram_power_spectra);
	}
	//-------------------------------------------------
	// reload merged partial results 
	read_memory_after_time_correlation(nimages, dimr, azh_avgs, ram_power_spectra);
	//--------------------------------------------------
	delete[] m_im;
}

void calc_power_spectra_autocorr2(INDEX nimages, INDEX dimy, INDEX dimx, INDEX& dimr, STORE_REAL* ram_power_spectra, MY_REAL* azh_avgs) {

	
	CUFFT_COMPLEX* dev_fft_gpu_(NULL);
	unsigned short* dev_im_gpu_(NULL);
	unsigned short* m_im_;
	int alloc_status_li_, alloc_status_fft_;
	alloc_status_li_ = cudaMalloc(&dev_im_gpu_, s_load_image.memory_tot);
	if (cudaSuccess != alloc_status_li_) {

		std::cout << "ERROR: cudaMalloc " << std::endl;
	}

	alloc_status_fft_ = cudaMalloc(&dev_fft_gpu_, s_fft.memory_tot);
	if (cudaSuccess != alloc_status_fft_) {

		std::cout << "ERROR: cudaMalloc dev_fft_gpu_" << std::endl;
	}
	/*alloc_status_fft_ = cudaMalloc(&dev_fft_gpu_, s_fft.memory_tot);
	if (cudaSuccess != alloc_status_fft_) {

		std::cout << "ERROR: cudaMalloc " << std::endl;
	}*/

	/*#if (CUFFT_TYPE == CUFFT_TYPE_DOUBLE)
		cufftExecZ2Z(plan_, dev_fft_gpu_, dev_fft_gpu_, CUFFT_FORWARD);
	#else
	#error Unknown CUDA type selected
	#endif*/
	m_im_ = new unsigned short[dimx * dimy];
	for (int i = 0; i < nimages; i++) {

		time_reading_from_disk.start();
		load_image(useri.file_list[i], dimy, dimx, true, m_im_, flg_display_read);
		time_reading_from_disk.stop();

		time_from_host_to_device.start();
		cudaMemcpy(dev_im_gpu_, m_im_, s_load_image.memory_one, cudaMemcpyHostToDevice);
		time_from_host_to_device.stop();

		// from image to complex matrix
		Image_to_complex_matrix2(dev_im_gpu_, dev_fft_gpu_, i, nimages);

	}
	cudaFree(dev_fft_gpu_);
	dev_fft_gpu_ = NULL;

	cudaFree(dev_im_gpu_);
	dev_im_gpu_ = NULL;

	delete[] m_im_;
	m_im_ = NULL;

	/*CUFFT_COMPLEX* tmp_display_cpx_(NULL);

	tmp_display_cpx_ = new CUFFT_COMPLEX[s_fft_images.dim * nimages];

	cudaMemcpy(tmp_display_cpx_, dev_images_gpu, nimages * s_fft_images.memory_one, cudaMemcpyDeviceToHost);
	for (int ii = 0; ii < s_fft_images.dim * nimages; ++ii)
		std::cout << tmp_display_cpx_[ii].x << "  + i " << tmp_display_cpx_[ii].y << std::endl;*/

	/*CUFFT_COMPLEX* tmp_display_cpx_(NULL);

	tmp_display_cpx_ = new CUFFT_COMPLEX[s_power_spectra.dim * nimages];

	cudaMemcpy(tmp_display_cpx_, dev_images_gpu, nimages * s_fft_images.memory_one, cudaMemcpyDeviceToHost);
	//for (int ii = 0; ii < s_fft_images.dim* nimages; ++ii)
		//std::cout << tmp_display_cpx_[ii].x << "  + i " << tmp_display_cpx_[ii].y << std::endl;

	FILE* version3;
	version3 = fopen("v4_timeSeries.txt", "w");
	for (int ii = 0; ii < nimages * s_power_spectra.dim; ++ii)
		//fprintf()
		fprintf(version3, "%f   %f\n",tmp_display_cpx_[ii].x, tmp_display_cpx_[ii].y);

	fclose(version3);*/

	Calc_StructureFunction_With_TimeCorrelation(nimages,dimx,dimy);

	pw_azth_avg(ram_radial_lut,nimages, dimr, azh_avgs, ram_power_spectra);

	//pw_save_and_azth_avg(ram_radial_lut, 0, nimages, dimr, azh_avgs, ram_power_spectra);

	

    //save_partial_timeseries(nimages, 0, s_time_series.numerosity, ram_power_spectra);

    //read_memory_after_time_correlation(nimages, dimr, azh_avgs, ram_power_spectra);

}

void load_memory_for_time_correlation(INDEX dimx, INDEX dimy, INDEX nimages, INDEX start_spacial_freq_in_lut, INDEX dimfreq, unsigned short *m_im, FFTW_REAL *image_mean)
{
	INDEX ifile;
	for (ifile = 0; ifile < nimages; ++ifile)
	{
		time_reading_from_disk.start();
		//load_binary_image(useri.file_list[ind_sot], dimy, dimx, true, m_im, flg_display_read);
		load_image(useri.file_list[ifile], dimy, dimx, true, m_im, flg_display_read);
		time_reading_from_disk.stop();
		if (1 == image_to_dev(-1, image_mean[ifile], m_im, false))
		{
			std::cerr << "warning an image (index = " << ifile << " appears to be null" << std::endl
				<< "\t=> average has been put equal to one to avoid bad divisions" << std::endl;
		}
		time_reshuffling_memory.start();
		lutfft_to_timeseries(dimfreq, (FFTW_REAL)(1.0), ifile, start_spacial_freq_in_lut);
		time_reshuffling_memory.stop();
	}

	/*CUFFT_COMPLEX* tmp_display_cpx_(NULL);

	tmp_display_cpx_ = new CUFFT_COMPLEX[s_power_spectra.dim * nimages];

	cudaMemcpy(tmp_display_cpx_, dev_images_gpu, nimages * s_fft_images.memory_one, cudaMemcpyDeviceToHost);
	//for (int ii = 0; ii < s_fft_images.dim* nimages; ++ii)
		//std::cout << tmp_display_cpx_[ii].x << "  + i " << tmp_display_cpx_[ii].y << std::endl;

	FILE* version3;
	version3 = fopen("v2_timeSeries.txt", "w");
	for (int ii = 0; ii < nimages * s_power_spectra.dim; ++ii)
		//fprintf()
		fprintf(version3, "%f   %f\n", tmp_display_cpx_[ii].x, tmp_display_cpx_[ii].y);

	fclose(version3);*/

	/**
	CUFFT_COMPLEX* v2(NULL);

	v2 = new CUFFT_COMPLEX[s_power_spectra.dim * nimages];
	CUFFT_COMPLEX* v4(NULL);

	v4 = new CUFFT_COMPLEX[s_power_spectra.dim * nimages];

	FILE* version2;
	version2 = fopen("C:\\Users\\mchraga\\source\\repos\\diffmicro\\diffmicro\\v2_timeSeries.txt", "r");
	FILE* version4;
	version4 = fopen("C:\\Users\\mchraga\\source\\repos\\diffmicro\\diffmicro\\v4_timeSeries.txt", "r");
	for (int ii = 0; ii < nimages * s_power_spectra.dim; ++ii) {
		//fprintf()
		fscanf(version2, "%lf   %lf", &v2[ii].x, &v2[ii].y);
		fscanf(version4, "%lf   %lf", &v4[ii].x, &v4[ii].y);
		if ((v2[ii].x != v4[ii].x) || (v2[ii].y != v4[ii].y)) {
			std::cout << "ERROR at " << ii <<"for timeSeries"<< std::endl;
			break;
		}
		//std::cout << v2[ii].x << "  + i " << v2[ii].y << std::endl;

	}
	std::cout << "test timeSeries OK" << std::endl;*/
}

void save_partial_timeseries(INDEX nimages, INDEX igroup, INDEX dimgroup, STORE_REAL *ram_power_spectra)
{
	// ram_power_spectra is used as temporary memory area becuase it is big enough
	INDEX i, ifile;
	for (ifile = 0; ifile < nimages; ++ifile)
	{
		// the result can be put at the beggining of the memory area start_freq = 0

		time_reshuffling_memory.start();
		timeseries_to_lutpw(dimgroup, (FFTW_REAL)(1.0), ifile, (INDEX)(0), ram_power_spectra);
		time_reshuffling_memory.stop();

		time_writing_on_disk.start();
		writeappend_partial_lutpw(igroup, ifile, dimgroup, ram_power_spectra);
		time_writing_on_disk.stop();
	}
}


void read_memory_after_time_correlation(INDEX nimages, INDEX dimr, MY_REAL *azh_avgs, FFTW_REAL *ram_power_spectra)
{
	INDEX i,ifile;
	for (ifile = 0; ifile < nimages; ++ifile)
	{
		// Warning for cpu execution:
		// We load the file to dev_power_spectra_cpu which is the variable read by the
		// function pw_save_and_azth_avg to simulate the download of data from gpu.

		time_reading_from_disk.start();
		read_mergedpartial_lutpw(ifile, s_fft_images.dim, dev_power_spectra_cpu);
		time_reading_from_disk.stop();

		pw_save_and_azth_avg(ram_radial_lut, ifile, (INDEX)(1), dimr, azh_avgs, ram_power_spectra);
	}
}

void time_series_analysis_cpu()
{
	time_time_correlation.start();
	timeseries_analysis_cpu(useri.nthread);
	time_time_correlation.stop();
}

void power_spectra_from_dev(INDEX n_pw, unsigned int* ram_radial_lut, STORE_REAL ram_power_spectra[], INDEX start_dist_display, bool flg_debug)
{
	INDEX i, dim_normal;
	STORE_REAL* power_spectrum_r;
	time_from_device_to_host.start();

	dim_normal = s_load_image.dim / 2;

	power_spectrum_r = new STORE_REAL[s_power_spectra.dim * s_power_spectra.numerosity];

	copy_power_spectra_from_dev(power_spectrum_r);

	/*for (int i = 0; i < s_power_spectra.dim * s_power_spectra.numerosity; i++) {

		std::cout << i <<"   "<<power_spectrum_r[i] << std::endl;
	}*/

	//---------------------------------------------------------------
	if (useri.flg_graph_mode)
	{
		for (int kk = 0; kk < n_pw; ++kk)
		{
			display_power_spectrum_r(kk + start_dist_display, &(power_spectrum_r[kk * s_power_spectra.dim]));
			waitkeyboard(10);
		}
	}
	//---------------------------------------------------------------

	for (i = 0; i < n_pw; ++i)
		radial_lut_rad_to_normal(dim_normal, s_power_spectra.dim, ram_radial_lut,
			&(power_spectrum_r[i * s_power_spectra.dim]), &(ram_power_spectra[i * dim_normal]));

	time_from_device_to_host.stop();

	delete[] power_spectrum_r;

	if (true == flg_debug && useri.flg_graph_mode)
	{
		for (i = 0; i < n_pw; ++i)
		{
			display_power_spectrum_normal(0, &(ram_power_spectra[i * dim_normal]));
		}
	}

}