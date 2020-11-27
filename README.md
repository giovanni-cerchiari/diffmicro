**Diffmicro**

This program implements the Differential Dynamic Algorithm.

Given sequence of <img src="https://render.githubusercontent.com/render/math?math=N> images $I_n$ equally spaced in time the program calculates the structure function $`d(m)`$ given by the following formula: 

$`d\left(m\right)=\frac{1}{N-m}\sum_{n=m}^{N-1}\lvert F_{xy} \left(I_{n-m}-I_{m}\right)\rvert^2`$ ,
where
- $`n`$ and $`m`$ are indices in the interval $`[0, N-1]`$,
- $`F_{xy}`$ indicates the bidimensional FFT over the pixels of the images,
- $`\lvert \ldots \rvert^2`$ indicates the modulus square of each amplitude of the bi-dimensional FFT.

The computation can be carried out by the program with two alghoritms either on CPU or GPU hardware for a total of four different modes.

The first alghortim is called "FIFO" and it is described in this article: https://aip.scitation.org/doi/10.1063/1.4755747.   

The second algorithm is called "time correlation" and it is described in this article: .  This article also contains a performance comparison between the "FIFO" and the "time correlation" alghoritms.

**External dependencies and installation**
This program is using the following external libraries.
* The Fast Fourier transfer package FFTW version 3.3.3 which can be found at this address: http://fftw.org/download.html.
* OpenCV - 3.0.0 for graphical interfaces purposes such as input panel or showing graphs, which can be found at this address https://opencv.org/releases/.
* CUDA toolkit version 10.2 (visit https://developer.nvidia.com/cuda-10.2-download-archive)
* file "dirent.h" (Copyright (C) 2006 Toni Ronkko) which is already included in the repository.
External dependencies dlls should be placed either in the working directory of the program or in "C:\Windows\System32".

**Build**
A build of the program can be found in folder "bin" . The program was built with Visual Studio 2013 tools (v120) for x64 hardware (https://visualstudio.microsoft.com/vs/older-downloads/).

**Input** 
The input image should be sequentially named (ex: image_0000, image_0001... image_0010, ...) and placed into the same folder with no other file. Accepted image types are TIFF, PNG, JPG. The maximum bit depth accepted is 16-bit. Above 16-bit depth the behaviour of the program is undefined.

**Output**
The program outputs the structure function $`d(m)`$ and their azimuthal average in a dedicated output folder specified by the user.
A log file is created after execution of the program which contains some informations on the run, for example execution time, etc ... .

**Usage**
The program needs a user_interface "option.txt" input file to initialize the starting parameters. The file should be sent to the program as argv. Do not use spaces in the path. If run without the file, a graphical user interface will help creating it. The GUI creates the file, which is then reloaded to start the computation. By using the argv option, the program can be run via matlab or a bash file. A sample of matlab code is uploaded in the repository.

**License**
Diffmicro is free software: it can be redistributed or modified under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
Diffmicro is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
A copy of the GNU General Public License is distributed along with Diffmicro, and can be seen at: <https://www.gnu.org/licenses/>.

**Cite**
If you wish to cite this program in your work the file "citation_diffmicro.bib" contains all the relevant citations in Bibtex format.
