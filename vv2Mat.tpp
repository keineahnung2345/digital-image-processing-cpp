#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> //line
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h> //CV_8UC3
#include <vector>
#include <iostream>

using namespace std;

template <typename T>
void vv2Mat(cv::Mat& img, vector<vector<T>>& vv, int h, int w, bool isForward){
    /*
    convert vector<vector<complex<double>>> to cv::Mat
    used in FFT2 and IFFT2
    if it's forward fourier transform, vv is of frequency domain;
    if it's inverse fourier transform, vv is of time domain
    */

    //resize to padded (or cropped) size
    img = cv::Mat(cv::Size(w, h), CV_8UC1, cv::Scalar(0));

    //find max and min of amplitude
    double maxAmp = std::numeric_limits<double>::lowest(), minAmp = std::numeric_limits<double>::max();
    for(int row = 0; row < h; row++){
        for(int col = 0; col < w; col++){
            // double amp = sqrt(pow(vv[row][col].real(), 2) + pow(vv[row][col].imag(), 2));
            double amp = getNorm(vv, row, col);
            if(isForward){
                amp /= 100.0; //?
                amp = log2(1.0 + amp); //log transformation?
            }
            maxAmp = max(maxAmp, amp);
            minAmp = min(minAmp, amp);
        }
    }

    // cout << "amp range: " << "[" << minAmp << ", " << maxAmp << "]" << endl;
    for(int row = 0; row < h; row++){
        for(int col = 0; col < w; col++){
            // double amp = sqrt(pow(vv[row][col].real(), 2) + pow(vv[row][col].imag(), 2));
            double amp = getNorm(vv, row, col);
            if(isForward){
                amp /= 100.0; //?
                amp = log2(1.0 + amp); //log transformation?
            }
            
            //normalize to [0,255]
            amp = (amp - minAmp)/(maxAmp - minAmp) * 255;
            // cout << (int)amp << " ";

            //move original point from left-top corner to the center
            int targetRow, targetCol;
            if(isForward){
                targetRow = (row < h/2) ? (row + h/2) : (row - h/2);
                targetCol = (col < w/2) ? (col + w/2) : (col - w/2);
            }else{
                targetRow = row;
                targetCol = col;
            }
            img.at<uchar>(targetRow, targetCol) = (int)amp;
        }
    }
};