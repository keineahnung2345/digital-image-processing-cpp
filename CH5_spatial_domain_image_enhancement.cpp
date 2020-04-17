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
    float mixRatio = 0.0){
    //p.146
    //Template, a.k.a. kernel

    /*
    mixRatio: for enhance filter, dst = src * mixRatio + gradient_value
    */

    //assume all kernel in "kernels"'s following member are the same
    int kernelHeight = kernels[0]->kernelHeight;
    int kernelWidth = kernels[0]->kernelWidth;
    int kernelMiddleY = kernels[0]->kernelMiddleY;
    int kernelMiddleX = kernels[0]->kernelMiddleX;
    float coef = kernels[0]->coef;
    bool isGradient = kernels[0]->isGradient;

    vector<vector<float>> arrs;
    for(int i = 0; i < kernels.size(); i++){
        arrs.push_back(kernels[i]->arr);
    }

    //source image
    int height = img.rows, width = img.cols;
    //destination image
    int newHeight = height-kernelHeight+kernelMiddleY;
    int newWidth = width-kernelWidth+kernelMiddleX;

    // cout << "org size: " << height << " * " << width << endl;

    if(padding){
        //after filter op, height will be decreased by (height-newHeight)
        int padt = (height-newHeight)/2, padb = height-newHeight-padt;
        int padl = (width-newWidth)/2, padr = width-newWidth-padl;
        pad(img, padt, padb, padl, padr, cv::BORDER_REPLICATE);
        //desitnation_image_height = (newHeight+height-newHeight) = height
        newHeight = height;
        newWidth = width;
        //source image 
        height += (padt + padb);
        width += (padl + padr);
    }

    // cout << "new src size: " << height << " * " << width << endl;
    // cout << "new dst size: " << newHeight << " * " << newWidth << endl;

    float maxGradient = numeric_limits<float>::lowest(), minGradient = std::numeric_limits<float>::max();
    float maxEnhanced = numeric_limits<float>::lowest(), minEnhanced = std::numeric_limits<float>::max();

    cv::Mat target(cv::Size(newWidth, newHeight), CV_8UC1, cv::Scalar(0));
    //the matrix used for storing the middle result when doing gradient kernel operation
    cv::Mat gMiddle(cv::Size(newWidth, newHeight), CV_32FC1, cv::Scalar(0));
    /*
    kernelHeight - kernelMiddleY: the displacement from middle to bottom
    (y + kernelHeight - kernelMiddleY): the y coordinate on image coordinate to the bottom of kernel
    */
    for(int y = kernelMiddleY; (y + kernelHeight - kernelMiddleY) < height; y++){
        /*
        (y + kernelHeight - kernelMiddleY): the pixel from source image that will be 
        operated with the bottom of the kernel
        */
        for(int x = kernelMiddleX; (x + kernelWidth - kernelMiddleX) < width; x++){
            //y and i moves along same coordinate, x and j moves along another
            if(adaptiveThreshold > 0.0){ //only valid if it's larger than 0
                //p.156, algo 5.1
                /*
                only do operation on the pixels, whose neighborhood:
                1. max - min larger than T
                or 
                2. variance larger than T
                */
                //find the max and min in its neighborhood
                int rangeMin = INT_MAX, rangeMax = INT_MIN;
                for(int i = 0; i < kernelHeight; i++){
                    for(int j = 0; j < kernelWidth; j++){
                        rangeMin = min(rangeMin, (int)img.at<uchar>(y-kernelMiddleY+i, x-kernelMiddleX+j));
                        rangeMax = max(rangeMax, (int)img.at<uchar>(y-kernelMiddleY+i, x-kernelMiddleX+j));
                    }
                }
                if(rangeMax - rangeMin <= adaptiveThreshold)continue;
            }

            float res = 0.0f;
            if(arrs.size() == 1){
                for(int i = 0; i < kernelHeight; i++){
                    for(int j = 0; j < kernelWidth; j++){
                        res += arrs[0][i*kernelWidth+j] * img.at<uchar>(y-kernelMiddleY+i, x-kernelMiddleX+j);
                    }
                }
            }else{
                //support for overlaying two kernel operation result
                for(vector<float>& arr : arrs){
                    for(int i = 0; i < kernelHeight; i++){
                        for(int j = 0; j < kernelWidth; j++){
                            res += arr[i*kernelWidth+j] * img.at<uchar>(y-kernelMiddleY+i, x-kernelMiddleX+j);
                        }
                    }
                    //take abs() here!!, G = abs(Gx) + abs(Gy)!!
                    res = fabs(res);
                }
            }

            if(!isGradient){
                res = (float)res * coef;
                res = fabs(res); //res could be negative in sharpening
                res = round(res);
                res = min(max((int)res, 0), 255); 
                target.at<uchar>(y, x) = res;
            }else if(mixRatio == 0){
                //gradient
                // take abs() and then normalize the range to [0,255]
                //it seems coef is always 1 for graident kernel?
                res = (float)res * coef;
                res = fabs(res);
                //temporarily store it into "target" for now
                gMiddle.at<float>(y, x) = res;
                //record the range of graident and do normalization later
                maxGradient = max(maxGradient, res);
                minGradient = min(minGradient, res);
                // cout << "minG: " << minGradient << ", maxG: " << maxGradient << ", res: " << res << endl;
            }else{
                //gradient, enhance filter
                res = (float)res * coef;
                //Note that the lower bound is -255 here!
                res = min(max((int)res, -255), 255);
                //mix(enhance)
                res = (int)img.at<uchar>(y, x) * mixRatio + res;
                //temporarily store it into "target" for now
                gMiddle.at<float>(y, x) = res;
                //record the range of graident and do normalization later
                maxEnhanced = max(maxEnhanced, res);
                minEnhanced = min(minEnhanced, res);
            }
        }
    }

    // cout << "minG: " << minGradient << ", maxG: " << maxGradient << endl;

    if(isGradient){
        for(int y = kernelMiddleY; (y + kernelHeight - kernelMiddleY) < height; y++){
            for(int x = kernelMiddleX; (x + kernelWidth - kernelMiddleX) < width; x++){
                float res = gMiddle.at<float>(y, x);
                // cout << "res: " << res << " -> ";
                if(mixRatio == 0){
                    //normalize from [minGradient, maxGradient] to [0,255]
                    res = (res-minGradient)/(maxGradient-minGradient) * 255;
                }else if(maxEnhanced-minEnhanced > 0){
                    //enhance
                    //normalize from [minEnhanced, maxEnhanced] to [0,255]
                    res = (res-minEnhanced)/(maxEnhanced-minEnhanced) * 255;
                }
                // cout << res << " -> ";
                res = round(res);
                // cout << res << " -> ";
                res = min(max((int)res, 0), 255); 
                // cout << res << endl;
                target.at<uchar>(y, x) = res;
            }
        }
    }

    img = target;
};

void MedianFilterOp(cv::Mat& img, int kernelHeight, int kernelWidth, 
    int kernelMiddleY, int kernelMiddleX, bool padding = false, bool adaptive = false){
    //p.146
    //Template, a.k.a. kernel

    //source image
    int height = img.rows, width = img.cols;
    //destination image
    int newHeight = height-kernelHeight+kernelMiddleY;
    int newWidth = width-kernelWidth+kernelMiddleX;

    if(padding){
        //after filter op, height will be decreased by (height-newHeight)
        int padt = (height-newHeight)/2, padb = height-newHeight-padt;
        int padl = (width-newWidth)/2, padr = width-newWidth-padl;
        pad(img, padt, padb, padl, padr, cv::BORDER_REPLICATE);
        //desitnation_image_height = (newHeight+height-newHeight) = height
        newHeight = height;
        newWidth = width;
        //source image 
        height += (padt + padb);
        width += (padl + padr);
    }

    cv::Mat target(cv::Size(newWidth, newHeight), CV_8UC1, cv::Scalar(0));
    /*
    kernelHeight - kernelMiddleY: the displacement from middle to bottom
    (y + kernelHeight - kernelMiddleY): the y coordinate on image coordinate to the bottom of kernel
    */
    for(int y = kernelMiddleY; (y + kernelHeight - kernelMiddleY) < height; y++){
        /*
        (y + kernelHeight - kernelMiddleY): the pixel from source image that will be 
        operated with the bottom of the kernel
        */
        for(int x = kernelMiddleX; (x + kernelWidth - kernelMiddleX) < width; x++){
            float res;
            //y and i moves along same coordinate, x and j moves along another
            if(adaptive){
                //p.162
                /*
                only do operation on the pixels, which is max or min in its neighborhood
                */
                //find the max and min in its neighborhood
                int rangeMin = INT_MAX, rangeMax = INT_MIN;
                for(int i = 0; i < kernelHeight; i++){
                    for(int j = 0; j < kernelWidth; j++){
                        rangeMin = min(rangeMin, (int)img.at<uchar>(y-kernelMiddleY+i, x-kernelMiddleX+j));
                        rangeMax = max(rangeMax, (int)img.at<uchar>(y-kernelMiddleY+i, x-kernelMiddleX+j));
                    }
                }
                if((img.at<uchar>(y, x) != rangeMax) && (img.at<uchar>(y, x) != rangeMin))continue;
            }

            vector<int> nbhd(kernelHeight * kernelWidth, 0); //neighborhood
            for(int i = 0; i < kernelHeight; i++){
                for(int j = 0; j < kernelWidth; j++){
                    nbhd[i*kernelWidth+j] = img.at<uchar>(y-kernelMiddleY+i, x-kernelMiddleX+j);
                }
            }

            //ndhd's size is always odd, so we just access nbhd[nbhd.size()/2]
            nth_element(nbhd.begin(), nbhd.begin()+nbhd.size()/2, nbhd.end());
            target.at<uchar>(y, x) = nbhd[nbhd.size()/2];
        }
    }
    img = target;
};

// void LoGFilterOp(){
//     //p.176, CH9
// };

void addNoise(cv::Mat& img, string mode = "gaussian", double mean = 0.0, double stddev = 0.0){
    //p.157, Matlab imnoise
    if(mode == "gaussian"){
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(mean, stddev);

        for(int r = 0; r < img.rows; r++){
            for(int c = 0; c < img.cols; c++){
                int val = distribution(generator);
                val += img.at<uchar>(r, c);
                val = min(max(val, 0), 255);
                img.at<uchar>(r, c) = val;
            }
        }
    }else if(mode == "salt_pepper"){
        //todo
    }
};

int main(){
    cv::Mat img_color = cv::imread("images/Lenna.png");
    cv::Mat img_gray = cv::imread("images/Lenna.png", 0);
    cv::Mat work_color = img_color.clone();
    cv::Mat work_gray = img_gray.clone();
    bool isSave = false;

    map<string, Kernel*> kernels = {
        //p.147
        {"SmoothAvg", new Kernel("SmoothAvg", {1,1,1,1,1,1,1,1,1}, 1.0/9)}
        , {"SmoothAvg5", new Kernel("SmoothAvg5", {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 1.0/25)}
        , {"SmoothAvg7", new Kernel("SmoothAvg7", {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 1.0/49)}
        , {"SmoothGauss", new Kernel("SmoothGauss", {1,2,1,2,4,2,1,2,1}, 1.0/16)}
        // , {"SobelVertical", new Kernel("SobelVertical", {-1,0,1, -2,0,2, -1,0,1}, 1.0)} //coef?
        // , {"SobelHorizontal", new Kernel("SobelHorizontal", {-1,-2,-1, 0,0,0, 1,2,1}, 1.0)}
        , {"LogEdgeDetection", new Kernel("LogEdgeDetection", {0,0,-1,0,0, 0,-1,-2,-1,0, -1,-2,16,-2,-1, 0,-1,-2,-1,0, 0,0,-1,0,0}, 1.0)}
        // , {"LaplacianEdgeDetection45", new Kernel("LaplacianEdgeDetection45", {0,-1,0, -1,4,-1, 0,-1,0}, 1.0)}
        // , {"LaplacianEdgeDetection90", new Kernel("LaplacianEdgeDetection90", {-1,-1,-1, -1,8,-1, -1,-1,-1}, 1.0)}
    };

    for(auto it = kernels.begin(); it != kernels.end(); it++){
        Kernel* kernel = it->second;
        work_gray = img_gray.clone();
        FilterOp(work_gray, {kernel}, true);
        vector<cv::Mat> imgs = {img_gray, work_gray};
        ShowHorizontal(imgs, kernel->name, isSave);
    }
    
    cv::Mat noise = img_gray.clone();
    addNoise(noise, "gaussian", 0.0, 15.0);
    //average filter
    cv::Mat average = img_gray.clone();
    Kernel* avgKernel = kernels["SmoothAvg"];
    FilterOp(average, {avgKernel}, false);
    //gaussian filter
    cv::Mat gaussian = img_gray.clone();
    Kernel* gKernel = kernels["SmoothGauss"];
    FilterOp(gaussian, {gKernel}, false);
    //median filter
    cv::Mat median = img_gray.clone();
    MedianFilterOp(median, 3, 3, 1, 1, false, false);
    vector<cv::Mat> imgs = {noise, average, gaussian, median};
    ShowHorizontal(imgs, "Average vs Gaussian vs Median", isSave);

    //Sharpen, gradient kernels
    map<string, Kernel*> gradientKernels = {
        //p.163
        {"RobertP45G", new Kernel("RobertP45G", {-1,0, 0,1}, 1.0, true)} //robert positive 45
        , {"RobertN45G", new Kernel("RobertN45G", {0,-1, 1,0}, 1.0, true)} //robert negative 45
        //p.165
        , {"SobelVerticalG", new Kernel("SobelVerticalG", {-1,0,1, -2,0,2, -1,0,1}, 1.0, true)}
        , {"SobelHorizontalG", new Kernel("SobelHorizontalG", {-1,-2,-1, 0,0,0, 1,2,1}, 1.0, true)}
        //p.167
        , {"Laplacian90", new Kernel("Laplacian90", {0,-1,0, -1,4,-1, 0,-1,0}, 1.0, true)}
        , {"Laplacian45", new Kernel("Laplacian45", {-1,-1,-1, -1,8,-1, -1,-1,-1}, 1.0, true)}
        , {"LaplacianWeighted", new Kernel("LaplacianWeighted", {-1,-4,-1, -4,20,-4, -1,-4,-1}, 1.0, true)}
        //p.176
        //Laplacian of Gaussian
    };

    // for(auto it = gradientKernels.begin(); it != gradientKernels.end(); it++){
    //     Kernel* kernel = it->second;
    //     work_gray = img_gray.clone();
    //     FilterOp(work_gray, {kernel}, true, 0.0);
    //     vector<cv::Mat> imgs = {img_gray, work_gray};
    //     ShowHorizontal(imgs, kernel->name, isSave);
    //     // Show(work_gray, kernel->name);
    // }

    cv::Mat rp45g = img_gray.clone();
    FilterOp(rp45g, {gradientKernels["RobertP45G"]}, true, 0.0);
    cv::Mat rn45g = img_gray.clone();
    FilterOp(rn45g, {gradientKernels["RobertN45G"]}, true, 0.0);
    cv::Mat rpn45g = img_gray.clone();
    FilterOp(rpn45g, {gradientKernels["RobertP45G"], gradientKernels["RobertN45G"]}, true, 0.0);
    vector<cv::Mat> RobertImgs = {img_gray, rp45g, rn45g, rpn45g};
    ShowHorizontal(RobertImgs, "Robert P vs N vs P+N 45G", isSave);

    cv::Mat vsg = img_gray.clone();
    FilterOp(vsg, {gradientKernels["SobelVerticalG"]}, true, 0.0);
    cv::Mat hsg = img_gray.clone();
    FilterOp(hsg, {gradientKernels["SobelHorizontalG"]}, true, 0.0);
    cv::Mat vhsg = img_gray.clone();
    FilterOp(vhsg, {gradientKernels["SobelVerticalG"], gradientKernels["SobelHorizontalG"]}, true, 0.0);
    vector<cv::Mat> SobelImages = {img_gray, vsg, hsg, vhsg};
    ShowHorizontal(SobelImages, "Sobel V vs H vs V+H G", isSave);

    cv::Mat l90 = img_gray.clone();
    FilterOp(l90, {gradientKernels["Laplacian90"]}, true, 0.0);
    cv::Mat l45 = img_gray.clone();
    FilterOp(l45, {gradientKernels["Laplacian45"]}, true, 0.0);
    cv::Mat lw = img_gray.clone();
    FilterOp(lw, {gradientKernels["LaplacianWeighted"]}, true, 0.0);
    vector<cv::Mat> LaplacianImages = {img_gray, l90, l45, lw};
    ShowHorizontal(LaplacianImages, "Laplacian 90 vs 45 vs Weighted G", isSave);
    
    
    cv::Mat rp45gEnhanced = img_gray.clone();
    FilterOp(rp45gEnhanced, {gradientKernels["RobertP45G"]}, true, 0.0, 1.8);
    vector<cv::Mat> RobertEnhancedImages = {img_gray, rp45g, rp45gEnhanced};
    ShowHorizontal(RobertEnhancedImages, "RobertP45G vs Enhanced G", isSave);
    
    cv::Mat vsgEnhanced = img_gray.clone();
    FilterOp(vsgEnhanced, {gradientKernels["SobelVerticalG"]}, true, 0.0, 1.8);
    vector<cv::Mat> SobelVerticalEnhancedImages = {img_gray, vsg, vsgEnhanced};
    ShowHorizontal(SobelVerticalEnhancedImages, "SobelVerticalG vs Enhanced G", isSave);

    cv::Mat lwEnhanced = img_gray.clone();
    FilterOp(lwEnhanced, {gradientKernels["LaplacianWeighted"]}, true, 0.0, 1.8);
    vector<cv::Mat> LaplacianEnhancedImages = {img_gray, lw, lwEnhanced};
    ShowHorizontal(LaplacianEnhancedImages, "Laplacian Weighted vs Enhanced G", isSave);
}
