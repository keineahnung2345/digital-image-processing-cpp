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

struct Line{
    int dist;
    int angle;
    int count;

    Line(int d, int a, int c) : dist(d), angle(a), count(c){};
};

bool EdgePrewitt(cv::Mat& img, int thres = 0, int edgeType = 0, bool thinning = true, bool outputGradient = false);
bool EdgeSobel(cv::Mat& img, int thres = 0, int edgeType = 0, bool thinning = true, bool outputGradient = false);
bool EdgeLoG(cv::Mat& img, int thres = 0, bool thinning = true, bool outputGradient = false);
bool EdgeCanny(cv::Mat& img, int thresL = 0, int thresH = 0, bool thinning = true);
bool Hough(cv::Mat& img, vector<Line>& lines, int numLines);
int DetectThreshold(cv::Mat& img, int& diff, int maxIter = 100);
int AutoThreshold(cv::Mat& img);
bool RegionGrow(cv::Mat& img, int seedRow = 0, int seedCol = 0, int thres = 16);
void qtdecomp(cv::Mat& img, cv::Mat& res, int thres, int minDim, int maxDim, int startRow, int startCol, int length);
bool qtdecomp(cv::Mat& img, cv::Mat& res, int thres = 4, int minDim = 1, int maxDim = 512);
void qtgetblk(cv::Mat& img, cv::Mat& res, int length, vector<cv::Mat>& vals, vector<vector<int>>& rcs);
bool qtsetblk(cv::Mat& img, cv::Mat& res, int length, vector<cv::Mat>& vals, vector<vector<int>>& rcs);
