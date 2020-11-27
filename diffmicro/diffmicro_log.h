/*
Copyright: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 8/2011

*/
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

/*
fuctions for diffmicro.exe application
*/

#ifndef _DIFFMICRO_TIME_H_
#define _DIFFMICRO_TIME_H_

#include <iostream>
#include <iomanip>
#include "stopwatch.h"
#include "diffmicro_io.h"
#include "correlation.h"


extern stopwatch time_reading_from_disk;
extern stopwatch time_writing_on_disk;
extern stopwatch time_from_host_to_device;
extern stopwatch time_from_device_to_host;
extern stopwatch time_fft_norm;
extern stopwatch time_differences;
extern stopwatch time_azh_avg;
extern stopwatch general_stw;
extern stopwatch time_time_correlation;
extern stopwatch time_reshuffling_memory;


extern unsigned long n_computed_fft;
extern unsigned long n_computed_diff;
extern unsigned long n_groups_reload;
extern unsigned long radial_lut_length;
extern unsigned long n_capacity;

extern unsigned long tot_memory_fft_images;
extern unsigned long tot_calculation_memory;

extern unsigned char *execution_map;

/*!This function must be called at the beginning of the program to initialize the log enviroment of diffmicro.*/
void init_log(INDEX dimtime);
/*This function prints the log of the execution on a output stream.*/
void print_log(std::ostream &out, INDEX &dimy, INDEX &dimx);
/*!This function must be called at the beginning of the program to free the memory of the log enviroment of diffmicro.*/
void close_log();

#endif
