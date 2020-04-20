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

inline long GetFreqWidth(cv::Mat& img, bool isExtending = true){
    //p.209

    int orgW = img.cols;
    //c++'s log is nature log!
    int w = pow(2, floor(log2(orgW))); //cut width so that w is a power of 2
    // cout << "in GetFreqWidth: " << orgW << " -> " << w;
    if(w != orgW && isExtending){
        w *= 2;
    }
    // cout << " -> " << w << endl;
    //w will be equal to orgW if orgW is a power of 2 by itself
    return w;
};

inline long GetFreqHeight(cv::Mat& img, bool isExtending = true){
    //p.210
    int orgH = img.rows;
    int h = pow(2, floor(log2(orgH))); //cut height so that h is a power of 2
    if(h != orgH && isExtending){
        h *= 2;
    }
    return h;
};

void FFT(vector<complex<double>>& times, vector<complex<double>>& freqs, int r){
    //p.198
    long count = 1 << r;

    /*
    W has symmetry, so we can just calculate half of it
    W[i+N/2] = -W[i]
    */
    vector<complex<double>> W(count/2);
    vector<complex<double>> X1;
    vector<complex<double>> X2(count);

    for(int n = 0; n < count/2; n++){
        double angle = -n * 2 * M_PI / count;
        W[n] = complex<double>(cos(angle), sin(angle));
    }

    X1 = times;
    for(int k = 0; k < r; k++){
        for(int j = 0; j < (1 << k); j++){
            int bfsize = 1 << (r-k);
            for(int i = 0; i < bfsize/2; i++){
                int p = j * bfsize;
                X2[i+p] = X1[i+p] + X1[i+p+bfsize/2];
                X2[i+p+bfsize/2] = (X1[i+p] - X1[i+p+bfsize/2]) * W[i * (1 << k)];
            }
        }
        X1.swap(X2);
    }

    //reorder
    for(int j = 0; j < count; j++){
        int p = 0;
        for(int i = 0; i < r; i++){
            if(j & (1 << i)){
                p += (1 << r-i-1);
            }
        }
        freqs[j] = X1[p];
    }
};

void IFFT(vector<complex<double>>& freqs, vector<complex<double>>& times, int r){
    //p.199
    long count = (1 << r);
    vector<complex<double>> X = freqs;
    freqs.resize(count);

    //find the conjugate of X
    for(int i = 0; i < X.size(); i++){
        X[i] = complex<double>(X[i].real(), -X[i].imag());
    }

    //do FFT on conjugated X
    FFT(X, times, r);

    //conjugate and * (1/N)
    for(int i = 0; i < count; i++){
        times[i] = complex<double>(times[i].real()/count, -times[i].imag()/count);
    }
};

template <typename T>
double getNorm(vector<vector<T>>& vv, int row, int col){};

template <typename T>
double getNorm(vector<vector<T>>& vv, int row, int col);

template<>
inline double getNorm(vector<vector<double>>& vv, int row, int col) { return vv[row][col]; }

template <>
inline double getNorm(vector<vector<complex<double>>>& vv, int row, int col) { return sqrt(pow(vv[row][col].real(), 2) + pow(vv[row][col].imag(), 2)); }

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

void FFT2(cv::Mat& img, bool isExtending, vector<vector<complex<double>>>& pOutput, char fillColor = 0xFF){ //0xFF: 8 bits
    //p.200
    //it also does fftshift of Matlab
    //img is a grayscale image?
    /*
    output "img"'s frequency original point is at the center
    output "pOutput"'s frequency original point is at the left-top corner

    img is input and output at the same time
    */
    long w = GetFreqWidth(img, isExtending);
    long h = GetFreqHeight(img, isExtending);

    vector<vector<complex<double>>> TD(h, vector<complex<double>>(w, complex<double>(0, 0)));
    vector<vector<complex<double>>> FD(h, vector<complex<double>>(w, complex<double>(0, 0)));

    //fill TD(time domain) with (padded) image
    for(int row = 0; row < h; row++){
        for(int col = 0; col < w; col++){
            if(row < img.rows && col < img.cols){
                TD[row][col] = complex<double>((int)img.at<uchar>(row, col), 0);
            }else{
                TD[row][col] = complex<double>((int)fillColor, 0);
            }
        }
    }

    //do FFT on each row of TD and save it to FD
    for(int row = 0; row < h; row++){
        FFT(TD[row], FD[row], (int)log2(w));
    }

    //transpose FD and save it to TD
    TD = vector<vector<complex<double>>>(w, vector<complex<double>>(h, complex<double>(0, 0)));
    for(int row = 0; row < h; row++){
        for(int col = 0; col < w; col++){
            TD[col][row] = FD[row][col];
        }
    }

    //do FFT on each col of TD(, or each row of transposed TD) and save it to FD
    FD = vector<vector<complex<double>>>(w, vector<complex<double>>(h, complex<double>(0, 0)));
    for(int col = 0; col < w; col++){
        FFT(TD[col], FD[col], (int)log2(h));
    }

    //pOutput is the transposed FD
    //pOutput = vector<vector<complex<double>>>(h, vector<complex<double>>(w, complex<double>(0, 0)));
    for(int row = 0; row < h; row++){
        for(int col = 0; col < w; col++){
            pOutput[row][col] = FD[col][row];
        }
    }

    vv2Mat(img, pOutput, h, w, true); //the following commented part is replaced by this line
    /*
    //resize to padded (or cropped) size
    img = cv::Mat(cv::Size(w, h), CV_8UC1, cv::Scalar(0));

    //find max and min of amplitude
    double maxAmp = std::numeric_limits<double>::lowest(), minAmp = std::numeric_limits<double>::max();
    for(int row = 0; row < h; row++){
        for(int col = 0; col < w; col++){
            double amp = sqrt(pow(pOutput[row][col].real(), 2) + pow(pOutput[row][col].imag(), 2));
            amp /= 100.0; //?
            amp = log2(1.0 + amp); //log transformation?
            maxAmp = max(maxAmp, amp);
            minAmp = min(minAmp, amp);
        }
    }

    // cout << "amp range: " << "[" << minAmp << ", " << maxAmp << "]" << endl;
    for(int row = 0; row < h; row++){
        for(int col = 0; col < w; col++){
            double amp = sqrt(pow(pOutput[row][col].real(), 2) + pow(pOutput[row][col].imag(), 2));
            amp /= 100.0; //?
            amp = log2(1.0 + amp); //log transformation?
            
            //normalize to [0,255]
            amp = (amp - minAmp)/(maxAmp - minAmp) * 255;
            // cout << (int)amp << " ";

            //move original point from left-top corner to the center
            int targetRow = (row < h/2) ? (row + h/2) : (row - h/2);
            int targetCol = (col < w/2) ? (col + w/2) : (col - w/2);
            img.at<uchar>(targetRow, targetCol) = (int)amp;
        }
    }
    */
};

void FFTShift(vector<vector<complex<double>>>& matrix){
    int height = matrix.size(), width = matrix[0].size();

    vector<vector<complex<double>>> oMatrix(height, vector<complex<double>>(width, complex<double>(0, 0)));

    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            //move original point from left-top corner to the center
            int targetRow = (row < height/2) ? (row + height/2) : (row - height/2);
            int targetCol = (col < width/2) ? (col + width/2) : (col - width/2);
            oMatrix[targetRow][targetCol] = matrix[row][col];
        }
    }

    matrix = oMatrix;
};

void IFFTShift(vector<vector<complex<double>>>& matrix){
    FFTShift(matrix);
};

void IFFT2(cv::Mat& img, vector<vector<complex<double>>>& pInput, long height, long width, long outHeight = 0, long outWidth = 0){
    //p.204
    /*
    we will process the top-left "width" and "height" part of img
    "outWidth" and "outHeight" are default "width" and "height"

    img is input and output at the same time
    */

    //pInput's original point is at the left-top corner, so we don't need fftshift here

    if(outWidth == 0) outWidth = width;
    if(outHeight == 0) outHeight = height;

    //copy pInput to FD
    vector<vector<complex<double>>> FD(height, vector<complex<double>>(width, complex<double>(0, 0)));
    vector<vector<complex<double>>> TD(height, vector<complex<double>>(width, complex<double>(0, 0)));
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            FD[row][col] = pInput[row][col];
        }
    }

    //do IFFT row by row
    for(int row = 0; row < height; row++){
        IFFT(FD[row], TD[row], (int)log2(width));
    }

    //set FD as transposed TD
    FD = vector<vector<complex<double>>>(width, vector<complex<double>>(height, complex<double>(0, 0)));
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            FD[col][row] = TD[row][col];
        }
    }

    //do IFFT col by col(in the view of original matrix)
    TD = vector<vector<complex<double>>>(width, vector<complex<double>>(height, complex<double>(0, 0)));
    for(int col = 0; col < width; col++){
        IFFT(FD[col], TD[col], (int)log2(height));
    }

    //transpose TD
    vector<vector<complex<double>>> tmp(height, vector<complex<double>>(width, complex<double>(0, 0)));
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            tmp[row][col] = TD[col][row];
        }
    }
    TD = tmp;

    vv2Mat(img, TD, outHeight, outWidth, false); //the following commented part is replaced by this line
    /*
    img = cv::Mat(cv::Size(outWidth, outHeight), CV_8UC1, cv::Scalar(0));

    //find max and min of amplitude
    double maxAmp = std::numeric_limits<double>::lowest(), minAmp = std::numeric_limits<double>::max();
    for(int row = 0; row < outHeight; row++){
        for(int col = 0; col < outWidth; col++){
            double amp = sqrt(pow(TD[row][col].real(), 2) + pow(TD[row][col].imag(), 2));
            // amp /= 100.0; //?
            // amp = log2(1.0 + amp); //log transformation?
            maxAmp = max(maxAmp, amp);
            minAmp = min(minAmp, amp);
        }
    }

    for(int row = 0; row < outHeight; row++){
        for(int col = 0; col < outWidth; col++){
            double amp = sqrt(pow(TD[row][col].real(), 2) + pow(TD[row][col].imag(), 2));
            // amp /= 100.0; //?
            // amp = log2(1.0 + amp); //log transformation?

            //normalize to [0,255]
            amp = (amp - minAmp)/(maxAmp - minAmp) * 255;

            //pInput's original point is at the left-top corner, so we don't need fftshift here
            img.at<uchar>(row, col) = (int)amp;
        }
    }
    */
};

void FreqFilt(cv::Mat& img, vector<vector<double>>& filter, char fillColor = 0xFF, bool isExtending = true){
    //p.208
    long orgWidth = img.cols;
    long orgHeight = img.rows;
    long width = GetFreqWidth(img, isExtending);
    long height = GetFreqHeight(img, isExtending);
    // cout << "in　FreqFilt, org: " << orgHeight << " x " << orgWidth << ", extended or cropped: " << height << " x " << width << endl;

    // cout << "after GetFreqWidth and GetFreqHeight: " << height << " x " << width << endl;

    vector<vector<complex<double>>> FD(height, vector<complex<double>>(width, complex<double>(0, 0)));
    //FFT to transform from space domain "img" to frequency domain "FD"
    // cout << "FFT2: " << endl;
    FFT2(img, isExtending, FD, fillColor);
    // cout << FD.size() << " x " << FD[0].size() << endl;
    //now the original point is at the "left-top corner" of FD

    //do operation with filter
    // cout << "do operation: " << endl;
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            FD[row][col] *= filter[row][col];
            //just use for visualization
            filter[row][col] = getNorm(FD, row, col);
        }
    }

    //don't need to use IFFTShift because both FD and filter's original point are at left-top corner

    //we use img's original width and height so the padding part will be cut
    // cout << "IFFT2: " << endl;
    IFFT2(img, FD, img.rows, img.cols, orgHeight, orgWidth);
    // cout << img.rows << " x " << img.cols << endl;
};

void FreqIdealLPF(cv::Mat& img, vector<vector<double>>& filter, double nFreq, bool isExtending = true){
    //p.214
    //we only use the input argument "img" to get its width and height
    //the generated filter's center is at left-top corner
    //nFreq = 20, 40, 60, ...

    int width = GetFreqWidth(img, isExtending);
    int height = GetFreqHeight(img, isExtending);

    filter = vector<vector<double>>(height, vector<double>(width, 0.0));

    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            if(sqrt(pow(row-height/2.0, 2.0) + pow(col-width/2.0, 2.0)) < nFreq){
                //shift original point from center to left-top corner
                int targetRow = (row < height/2.0) ? (row + height/2.0) : (row - height/2.0);
                int targetCol = (col < width/2.0) ? (col + width/2.0) : (col - width/2.0);
                filter[targetRow][targetCol] = 1;
                // cout << "(" << targetRow << ", " << targetCol << ") ";
            }else{
                //shift original point from center to left-top corner
                int targetRow = (row < height/2.0) ? (row + height/2.0) : (row - height/2.0);
                int targetCol = (col < width/2.0) ? (col + width/2.0) : (col - width/2.0);
                filter[targetRow][targetCol] = 0;
            }
        }
    }
};

void FreqGaussLPF(cv::Mat& img, vector<vector<double>>& filter, double sigma, bool isExtending = true){
    //p.219
    //sigma = 20, 40, 60, ...
    int width = GetFreqWidth(img, isExtending);
    int height = GetFreqHeight(img, isExtending);

    filter = vector<vector<double>>(height, vector<double>(width, 0.0));

    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            //shift original point from center to left-top corner
            int targetRow = (row < height/2) ? (row + height/2) : (row - height/2);
            int targetCol = (col < width/2) ? (col + width/2) : (col - width/2);
            filter[targetRow][targetCol] = exp(-(pow(row-height/2.0, 2) + pow(col-width/2.0, 2))/(2*pow(sigma, 2.0)));
        }
    }
};

void FreqGaussHPF(cv::Mat& img, vector<vector<double>>& filter, int sigma, bool isExtending = true){
    //p.223
    //sigma = 20, 40, 60, ...
    int width = GetFreqWidth(img, isExtending);
    int height = GetFreqHeight(img, isExtending);

    filter = vector<vector<double>>(height, vector<double>(width, 0.0));

    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            //shift original point from center to left-top corner
            int targetRow = (row < height/2) ? (row + height/2) : (row - height/2);
            int targetCol = (col < width/2) ? (col + width/2) : (col - width/2);
            filter[targetRow][targetCol] = 1 - exp(-(pow(row-height/2.0, 2) + pow(col-width/2.0, 2))/(2*pow(sigma, 2.0)));
        }
    }
};

void FreqLaplace(cv::Mat& img, vector<vector<double>>& filter, bool isExtending = true){
    //p.226
    int width = GetFreqWidth(img, isExtending);
    int height = GetFreqHeight(img, isExtending);

    filter = vector<vector<double>>(height, vector<double>(width, 0.0));

    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            //shift original point from center to left-top corner
            int targetRow = (row < height/2) ? (row + height/2) : (row - height/2);
            int targetCol = (col < width/2) ? (col + width/2) : (col - width/2);
            filter[targetRow][targetCol] = -(pow(row-height/2.0, 2.0) + pow(col-width/2.0, 2.0));
        }
    }
};

void FreqGaussBRF(cv::Mat& img, vector<vector<double>>& filter, double blockFreq, double blockWidth, bool isExtending = true){
    //p.229
    //band rejection filter, height and width and calculated from img
    int width = GetFreqWidth(img, isExtending);
    int height = GetFreqHeight(img, isExtending);

    //support for overlay multiple gauss brf together
    if(filter.size() == 0){
        filter = vector<vector<double>>(height, vector<double>(width, std::numeric_limits<double>::max()));
    }

    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            //shift original point from center to left-top corner
            int targetRow = (row < height/2) ? (row + height/2) : (row - height/2);
            int targetCol = (col < width/2) ? (col + width/2) : (col - width/2);
            double Duv = sqrt(pow(row-height/2,2)+pow(col-width/2,2));
            //Duv in denominator is sqrt(pow(row,2)+pow(col,2))?
            //support for overlay multiple gauss brf together
            //use min to merge two gauss brand rejection filter
            filter[targetRow][targetCol] = min(filter[targetRow][targetCol], 
                1 - exp(-pow(((pow(Duv,2.0)-pow(blockFreq, 2.0))/(Duv*blockWidth)), 2.0)/2.0));
        }
    }
};

void addPeriodicNoise(cv::Mat& img, double amp = 20.0, double freq = 20.0){
    //p.229
    for(int r = 0; r < img.rows; r++){
        for(int c = 0; c < img.cols; c++){
            double val = amp * sin(freq*r) + amp * sin(freq*c);
            val += img.at<uchar>(r, c);
            val = min(max((int)val, 0), 255);
            img.at<uchar>(r, c) = (int)val;
        }
    }
};

int main(){
    // cv::Mat img_gray = cv::imread("images/Lenna.png", 0);
    cv::Mat img_gray = cv::imread("images/cat.jpeg", 0);
    cv::Mat work_gray = img_gray.clone();
    bool isSave = false;
    bool isExtending = true;

    cv::Mat idealLPFed = work_gray.clone();
    vector<vector<double>> idealLPF;
    //generate ideal low pass filter
    FreqIdealLPF(idealLPFed, idealLPF, 20.0, isExtending);
    //apply ideal low pass filter
    FreqFilt(idealLPFed, idealLPF, 0xFF, isExtending);
    vector<cv::Mat> idealLPFs = {work_gray, idealLPFed};
    ShowHorizontal(idealLPFs, "Ideal Low pass filter", isSave);

    cv::Mat gaussLPFed = work_gray.clone();
    vector<vector<double>> gaussLPF;
    //generate gauss low pass filter
    FreqGaussLPF(gaussLPFed, gaussLPF, 20.0, isExtending);
    //apply gauss low pass filter
    FreqFilt(gaussLPFed, gaussLPF, 0xFF, isExtending);
    vector<cv::Mat> gaussLPFs = {work_gray, gaussLPFed};
    ShowHorizontal(gaussLPFs, "Gauss Low pass filter", isSave);

    cv::Mat gaussHPFed = work_gray.clone();
    vector<vector<double>> gaussHPF;
    //generate gauss low pass filter
    FreqGaussHPF(gaussHPFed, gaussHPF, 20.0, isExtending);
    //apply gauss low pass filter
    FreqFilt(gaussHPFed, gaussHPF, 0xFF, isExtending);
    vector<cv::Mat> gaussHPFs = {work_gray, gaussHPFed};
    ShowHorizontal(gaussHPFs, "Gauss High pass filter", isSave);

    cv::Mat laplaceed = work_gray.clone();
    vector<vector<double>> laplaceF;
    //generate gauss low pass filter
    FreqLaplace(laplaceed, laplaceF, isExtending);
    //apply gauss low pass filter
    FreqFilt(laplaceed, laplaceF, 0xFF, isExtending);
    vector<cv::Mat> laplaceeds = {work_gray, laplaceed};
    ShowHorizontal(laplaceeds, "Laplace filter", isSave);

    cv::Mat periodicNoised = work_gray.clone();
    //p.229
    //add periodic noise
    addPeriodicNoise(periodicNoised);
    //p.230
    //show original and noised image in frequency domain
    cv::Mat Fimg = work_gray.clone();
    vector<vector<complex<double>>> dummy(GetFreqHeight(Fimg, isExtending), vector<complex<double>>(GetFreqWidth(Fimg, isExtending), complex<double>(0, 0)));
    FFT2(Fimg, isExtending, dummy, 0xFF);
    cv::Mat FperiodicNoised = periodicNoised.clone();
    FFT2(FperiodicNoised, isExtending, dummy, 0xFF);
    vector<cv::Mat> Fimgs = {Fimg, FperiodicNoised};
    ShowHorizontal(Fimgs, "Image and noised image in frequency domain", isSave);

    //p.231, 232
    //show gauss brf and image applied gauss brf in frequency domain
    //show image and image applied gauss brf in space domain
    vector<vector<double>> gaussBRF;
    cv::Mat FgaussBRF = periodicNoised.clone(); //image of gaussBRF, in frequency domain
    //generate gauss band rejection filter
    FreqGaussBRF(FgaussBRF, gaussBRF, 93, 20, isExtending);
    //overlay two gauss brf together
    FreqGaussBRF(FgaussBRF, gaussBRF, 187, 20, isExtending);
    //gaussBRF is in frequency domain, so isForward is true
    vv2Mat(FgaussBRF, gaussBRF, gaussBRF.size(), gaussBRF[0].size(), true);
    //apply gauss band rejection filter
    cv::Mat gaussBRFed = periodicNoised.clone();
    FreqFilt(gaussBRFed, gaussBRF, 0xFF, isExtending);
    cv::Mat FguassBRFed; //the image operated with gaussBRF, in frequency domain
    //gaussBRF is in frequency domain, so isForward is true
    vv2Mat(FguassBRFed, gaussBRF, gaussBRF.size(), gaussBRF[0].size(), true);
    vector<cv::Mat> FgaussBRFimgs = {FgaussBRF, FguassBRFed};
    ShowHorizontal(FgaussBRFimgs, "Filter and filterd image in frequency domain", isSave);
    vector<cv::Mat> gaussBRFeds = {periodicNoised, gaussBRFed};
    ShowHorizontal(gaussBRFeds, "Gauss Band Rejection filter", isSave);
}
