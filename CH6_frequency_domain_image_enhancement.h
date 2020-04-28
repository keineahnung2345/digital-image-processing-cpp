#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> //line
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h> //CV_8UC3
#include <iostream>
#include <map>
#include <cmath> //M_PI
#include <complex>
#include "utility.h"

using namespace std;

inline long GetFreqWidth(cv::Mat& img, bool isExtending = true);
inline long GetFreqHeight(cv::Mat& img, bool isExtending = true);
void FFT(vector<complex<double>>& times, vector<complex<double>>& freqs, int r);
void IFFT(vector<complex<double>>& freqs, vector<complex<double>>& times, int r);

template <typename T>
double getNorm(vector<vector<T>>& vv, int row, int col){};

template <typename T>
double getNorm(vector<vector<T>>& vv, int row, int col);

template<>
inline double getNorm(vector<vector<double>>& vv, int row, int col);

template <>
inline double getNorm(vector<vector<complex<double>>>& vv, int row, int col);

template <typename T>
void vv2Mat(cv::Mat& img, vector<vector<T>>& vv, int h, int w, bool isForward);

void FFT2(cv::Mat& img, bool isExtending, vector<vector<complex<double>>>& pOutput, char fillColor = 0xFF);
void FFTShift(vector<vector<complex<double>>>& matrix);
void IFFTShift(vector<vector<complex<double>>>& matrix);
void IFFT2(cv::Mat& img, vector<vector<complex<double>>>& pInput, long height, long width, long outHeight = 0, long outWidth = 0);
void FreqFilt(cv::Mat& img, vector<vector<double>>& filter, char fillColor = 0xFF, bool isExtending = true);
void FreqIdealLPF(cv::Mat& img, vector<vector<double>>& filter, double nFreq, bool isExtending = true);
void FreqGaussLPF(cv::Mat& img, vector<vector<double>>& filter, double sigma, bool isExtending = true);
void FreqGaussHPF(cv::Mat& img, vector<vector<double>>& filter, int sigma, bool isExtending = true);
void FreqLaplace(cv::Mat& img, vector<vector<double>>& filter, bool isExtending = true);
void FreqGaussBRF(cv::Mat& img, vector<vector<double>>& filter, double blockFreq, double blockWidth, bool isExtending = true);
void addPeriodicNoise(cv::Mat& img, double amp = 20.0, double freq = 20.0);

#include "getNorm.tpp"
#include "vv2Mat.tpp"
