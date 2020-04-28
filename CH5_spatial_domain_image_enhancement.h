#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> //line
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h> //CV_8UC3
#include <iostream>
#include <map>
#include <cmath> //M_PI
#include "utility.h"

using namespace std;

struct Kernel{
    string name;
    int kernelHeight;
    int kernelWidth;
    int kernelMiddleY;
    int kernelMiddleX;
    vector<float> arr;
    float coef;
    bool isGradient; //if its gradient, take abs() and then normalize the range to [0,255]

    Kernel(string n, vector<float> a, float c, bool g = false) : name(n), arr(a), coef(c), isGradient(g){
        kernelHeight = kernelWidth = (int)(sqrt(a.size()));
        kernelMiddleY = kernelMiddleX = kernelHeight/2;
    };
};

void FilterOp(cv::Mat& img, vector<Kernel*> kernels, bool padding = false, float adaptiveThreshold = 0.0,
    float mixRatio = 0.0);
void MedianFilterOp(cv::Mat& img, int kernelHeight, int kernelWidth, 
    int kernelMiddleY, int kernelMiddleX, bool padding = false, bool adaptive = false);
void addNoise(cv::Mat& img, string mode = "gaussian", double mean = 0.0, double stddev = 0.0);
