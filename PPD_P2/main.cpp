#include <SDKDDKVer.h>

#include <stdio.h>
#include <tchar.h>

#include <string>
#include <iostream>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/types_c.h> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <cmath>
#include <chrono>  

using namespace cv;

extern "C" bool apply_Sobel(cv::Mat * imgTresh, cv::Mat * Grayscale);

int main()
{
	String pathToImage = "D:\\ppd\\P2_PPD\\PPD_P2\\yacc.jpg";

	Mat inputImg = imread(pathToImage, 0); 
	Mat filteredImg = inputImg.clone();
	apply_Sobel(&inputImg, &filteredImg);
	imshow("Edgy yacc", filteredImg);

	waitKey(0);
	return 0;
}
