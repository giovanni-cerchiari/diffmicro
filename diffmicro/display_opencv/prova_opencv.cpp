

/*
author: Giovanni Cerchiari
e-mail: giovanni.cerchiari@gmail.com
date: 12/2015
implemented with opencv v 3.0
*/

#include "stdafx.h"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include "figure_opencv.h"
#include "mouse_opencv.h"

cv::Mat image_read;
cv::Mat image;
cv::Mat image_display;

void gaussian(int dimx, int dimy, double sigma[], double *mat)
{
	int i, j;
	double xc, yc;
	double x, y2;
	double *row;

	xc = (double)(dimx) / 2.;
	yc = (double)(dimy) / 2.;

	for (j = 0; j<dimy; ++j)
	{
		y2 = (double)(j)-yc;
		
			y2 *= y2;
			row = &(mat[j*dimx]);
			for (i = 0; i < dimx; ++i)
			{
				x = (double)(i)-xc;
				if (x>0)	row[i] = 255 * exp(-0.5*(x*x / (sigma[0] * sigma[0]) + y2 / (sigma[1] * sigma[1])));
				else      row[i] = 0;
			}
	}

}




int main(int argc, char** argv)
{

	std::string filename = "F:\\foto tunisia\\vale\\vale118.jpg";
	//std::string filename = "E:\\gopro\\GOPR0008.JPG";
	std::string trackbarname_min = "min_trackbar";
	std::string trackbarname_max = "max_trackbar";
	int trackbarpos = 0;
	std::string window_name1 = "Universe1";
	std::string window_name2 = "Universe2";
	double *imaged;
	int i,j, dim;
	float r, g, b;

	image_read = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);   // Read the file

	if (!image_read.data)                              // Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	std::cout << "step[0] = " << image_read.step[0]/3 << std::endl;
	std::cout << "cols[0] = " << image_read.cols  << std::endl;
	std::cout << "rows[0] = " << image_read.rows << std::endl;
	image.create(image_read.rows, image_read.cols, CV_8U);
	image_display.create(image.rows, image.cols, CV_8U);
	dim = image.rows*image.cols;
	imaged = new double[dim];
	for (j = 0; j < image_read.rows; ++j)
	{
		for (i = 0; i < image_read.cols; ++i)
		{
			r = (unsigned __int8)(image_read.data)[j * image_read.step[0] + i * image_read.step[1]];
			g = (unsigned __int8)(image_read.data)[j * image_read.step[0] + i * image_read.step[1] + 1];
			b = (unsigned __int8)(image_read.data)[j * image_read.step[0] + i * image_read.step[1] + 2];
			//		(unsigned __int8)(image.data)[i] = (r + g + b) / 3.0;
			//		(unsigned __int8)(image_display.data)[i] = (r + g + b) / 3.0;
			imaged[j*image_read.cols+i] = (r + g + b) / 3.0;
		}
	}

	double sigma[2];
	sigma[0] = 100;
	sigma[1] = 2*sigma[0];
	//gaussian(image_read.cols, image_read.rows, sigma, imaged);

	new_figure(image.cols, image.rows, imaged);	
	new_figure(image.cols, image.rows, imaged);

	cv::waitKey(0);                                          // Wait for a keystroke in the window
	delete_figures();
	system("pause");
	//delete[] imaged;
	//imaged = NULL;
	return 0;
}

