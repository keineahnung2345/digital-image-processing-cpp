#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> //line
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h> //CV_8UC3
#include <iostream>
#include <map>

using namespace std;

std::string type2str(int type) {
  std::string r;
  
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  
  switch ( depth ) { 
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }
  
  r += "C";
  r += (chans+'0');
  
  return r;
};

void stringReplace(string& base, string from, string to){
    //start search from last end to speed up
    size_t pos = 0;
    while((pos = base.find(from, pos)) != string::npos){
        base.replace(pos, from.length(), to);
    }
};

void ConcatHorizontal(vector<cv::Mat>& imgs, cv::Mat& target){
    target = imgs[0];
    //assume imgs[0]'s height is always largest
    int finalHeight = target.rows;
    for(int i = 1; i < imgs.size(); i++){
        if(imgs[i].rows < finalHeight){
            //padding for imgs[i]
            int padh = finalHeight - imgs[i].rows;
            cv::Mat tmp(cv::Size(finalHeight, imgs[i].cols), CV_8UC1, cv::Scalar(0));
            copyMakeBorder(imgs[i], tmp, 0, padh, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
            imgs[i] = tmp;
        }
        hconcat(target, imgs[i], target);
    }
};

void Show(cv::Mat& img, string title = "Display Window", bool save = false){
    if(save){
        stringReplace(title, " ", "_");
        cv::imwrite("images/result/" + title + ".png", img);
    }else{
        cv::namedWindow( title, cv::WINDOW_AUTOSIZE);// Create a window for display.
        cv::imshow(title, img);                   // Show our image inside it.
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
};

void ShowHorizontal(vector<cv::Mat>& imgs, string title = "Display Window", bool save = false){
    cv::Mat target;
    ConcatHorizontal(imgs, target);

    if(save){
        stringReplace(title, " ", "_");
        cv::imwrite("images/result/" + title + ".png", target);
    }else{
        cv::namedWindow( title, cv::WINDOW_AUTOSIZE);// Create a window for display.
        cv::imshow(title, target);                   // Show our image inside it.
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
};

void pad(cv::Mat& img, int padt, int padb, int padl, int padr, int mode = cv::BORDER_CONSTANT){
    //https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
    //mode could be cv::BORDER_CONSTANT, cv::BORDER_REPLICATE, ...
    cv::Mat tmp(cv::Size(img.rows+padt+padb, img.cols+padl+padr), CV_8UC1, cv::Scalar(0));
    if(mode == cv::BORDER_CONSTANT)
        copyMakeBorder(img, tmp, padt, padb, padl, padr, cv::BORDER_CONSTANT, cv::Scalar(0));
    else
        copyMakeBorder(img, tmp, padt, padb, padl, padr, mode);
    img = tmp;
};

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
};
