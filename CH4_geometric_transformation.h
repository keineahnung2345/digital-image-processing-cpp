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

bool ImMove(cv::Mat& img, int tcol, int trow);
void HorMirror(cv::Mat& img);
void VerMirror(cv::Mat& img);
void Scale(cv::Mat& img, double times);
void Rotate(cv::Mat& img, float angle);
void Rotate(cv::Mat& img, float angle, int center_col, int center_row);
int InterpBilinear(cv::Mat& img, double x, double y);
void InterpCubic(double x, double y);
// void cp2tform(vector<vector<double>>& input_points, vector<vector<double>>& base_points, vector<vector<double>>& transform, string transform_type = "affine");
void GetProjPara(vector<vector<double>>& basePoints, vector<vector<double>>& srcPoints, vector<double>& projectionPara);
void ProjTrans(vector<double>& srcPoint, vector<double>& projectionPara, vector<double>& dstPoint);
bool ImProjRestore(cv::Mat& img, vector<vector<double>>& basePoints, vector<vector<double>>& srcPoints, bool isInterp);
void fitBasePoints(cv::Mat& img, vector<vector<double>>& basePoints);
