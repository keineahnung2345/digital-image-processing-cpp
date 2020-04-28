#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> //line
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h> //CV_8UC3
#include <iostream>
#include <map>
#include <cmath> //M_PI
#include "utility.h"

bool RGB2CMY(cv::Mat& img);
bool CMY2RGB(cv::Mat& img);
bool RGB2HSI(cv::Mat& img);
bool HSI2RGB(cv::Mat& img);
bool RGB2HSV(cv::Mat& img);
bool HSV2RGB(cv::Mat& img);
bool RGB2YUV(cv::Mat& img);
bool YUV2RGB(cv::Mat& img);
bool RGB2YIQ(cv::Mat& img);
bool YIQ2RGB(cv::Mat& img);
bool compensate(cv::Mat& img);
bool colorBalance(cv::Mat& img);
