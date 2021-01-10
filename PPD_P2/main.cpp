#include <string>
#include <iostream>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/types_c.h> 
#include <opencv2/imgproc/imgproc.hpp> 

using namespace cv;

extern "C" bool apply_Sobel(cv::Mat * inputImage, cv::Mat * outputImage);

int main()
{
	String pathToImage = "C:\\Users\\Andrei\\source\\repos\\P2_PPD\\PPD_P2\\yacc.jpg";

	Mat inputImg = imread(pathToImage, 0);
	Mat filteredImg = inputImg.clone();
	apply_Sobel(&inputImg, &filteredImg);
	imshow("Edgy yacc", filteredImg);

	waitKey(0);
	return 0;
}
