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

enum class FB{
    X = -1, //don't care
    B, //background
    F //foreground
};

struct BinaryKernel{
    string name;
    int kernelHeight;
    int kernelWidth;
    int kernelMiddleY;
    int kernelMiddleX;
    vector<FB> arr;

    //only support for square kernel currently!
    BinaryKernel(string n, vector<FB> a) : name(n), arr(a){
        kernelHeight = kernelWidth = (int)(sqrt(a.size()));
        kernelMiddleY = kernelMiddleX = kernelHeight/2;
    };
};

//defined in CH8.cpp
extern BinaryKernel crossKernel;

void initializeKernels();
void Erode(cv::Mat& img, BinaryKernel& kernel);
void Dilate(cv::Mat& img, BinaryKernel& kernel);
void Open(cv::Mat& img, BinaryKernel& kernel);
void Close(cv::Mat& img, BinaryKernel& kernel);
void ExtractBoundary(cv::Mat& img, BinaryKernel& kernel);
void CleanConnRgn(cv::Mat& img, int startRow, int startCol, int nConn = 8);
void TraceBoundary(cv::Mat& img, vector<vector<vector<int>>>& boundaries, bool traceAll = false);
void FillRgn(cv::Mat& img, int seedRow = -1, int seedCol = -1, BinaryKernel& kernel = crossKernel);
void LabelConnRgn(cv::Mat& img, int nConn = 8);
void Thining(cv::Mat& img);
int TestConnRgn(cv::Mat& img, vector<vector<bool>>& visited, vector<vector<int>>& ptVisited, 
    int& curConnRgnSize, int row, int col, int lowerThres, int upperThres);
void PixelImage(cv::Mat& img, int lowerThres = 1, int upperThres = 1000);
void Convex(cv::Mat& img, bool constrain = false);
void GrayDilate(cv::Mat& img, BinaryKernel& kernel);
void GrayErode(cv::Mat& img, BinaryKernel& kernel);
void GrayOpen(cv::Mat& img, BinaryKernel& kernel);
void GrayClose(cv::Mat& img, BinaryKernel& kernel);
void TopHat(cv::Mat& img, BinaryKernel& kernel);
