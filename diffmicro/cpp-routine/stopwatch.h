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

#ifndef _STOPWATCH_H_
#define _STOPWATCH_H_

#include <ctime>
#include <iostream>
#include <fstream>

/*!
class stopwacth integrates the elapesed time between any call to functions start() and stop()
*/
class stopwatch
{
	public:
		stopwatch() {m_t = 0; m_sec_over_clock = 1./CLOCKS_PER_SEC;}
		~stopwatch() {}

		void start() {m_start = clock();}
		double stop() {m_end = clock(); m_t += (m_end - m_start) * m_sec_over_clock; return m_t;}
		void reset() {m_t=0;}
		double t() {return (m_t);}
		
		void operator=(stopwatch &stw) {this->m_t = stw.m_t;	}
		void operator=(double t) {this->m_t = t;}

		double operator+(stopwatch &stw) {return (this->m_t+stw.m_t);}
		double operator-(stopwatch &stw) {return (this->m_t-stw.m_t);}
		double operator*(stopwatch &stw) {return (this->m_t*stw.m_t);}
		double operator/(stopwatch &stw) {return (this->m_t/stw.m_t);}

		bool operator<(stopwatch &stw)  {return (this->m_t<stw.m_t);}
		bool operator>(stopwatch &stw)  {return (this->m_t>stw.m_t);}
		bool operator<=(stopwatch &stw) {return (this->m_t<=stw.m_t);}
		bool operator>=(stopwatch &stw) {return (this->m_t>=stw.m_t);}
		bool operator==(stopwatch &stw) {return (this->m_t==stw.m_t);}
		bool operator!=(stopwatch &stw) {return (this->m_t!=stw.m_t);}

		friend std::ostream& operator<<(std::ostream& stream, stopwatch& stw){ stream <<stw.m_t; return stream;}

	protected:

		clock_t m_start;
		clock_t m_end;
		double m_t;
		double m_sec_over_clock;
};


#endif
