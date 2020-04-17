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

double eps = std::numeric_limits<double>::epsilon();
// double eps = 0.0;

bool RGB2CMY(cv::Mat& img){
    //p.239
    //it also serves as CMY2RGB
    //Note that even if there is RGB in function name, the image's channel order is BGR
    if(type2str(img.type()) != "8UC3") return false;

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            pixel[0] = 255 - pixel[0]; //B
            pixel[1] = 255 - pixel[1]; //G
            pixel[2] = 255 - pixel[2]; //R
            img.at<cv::Vec3b>(row, col) = pixel;
        }
    }

    return true;
};

bool CMY2RGB(cv::Mat& img){
    return RGB2CMY(img);
};

bool RGB2HSI(cv::Mat& img){ //?
    //p.243
    if(type2str(img.type()) != "8UC3") return false;

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            double B = pixel[0]/255.0;
            double G = pixel[1]/255.0;
            double R = pixel[2]/255.0;

            double minColor = min({B, G, R});
            double maxColor = max({B, G, R});

            double H;
            double S;
            double I = (R+G+B)/3.0; //[0,1]

            if(minColor == maxColor){
                H = 0;
                S = 0;
            }else{
                //its unit is radiance
                double cosVal = (R-G+R-B)/2.0 / sqrt((R-G)*(R-G) + (R-B)*(G-B) + eps);
                double radiance = (abs(cosVal-1) < eps) ? 0.0 : (abs(cosVal-(-1)) < eps) ? M_PI : acos(radiance);
                // double radiance = acos(radiance);

                H = (B <= G) ? radiance : 2.0*M_PI - radiance; //[0,2.0*M_PI]
                S = 1 - 3.0*minColor/(R+G+B + eps); //[0,1]
            }

            //normalize
            H = H/(2.0*M_PI) * 255.0;
            S = S * 255.0;
            I = I * 255.0;

            //put H in R channel, S in G channel, I in B channel
            img.at<cv::Vec3b>(row, col) = cv::Vec3b((int)I, (int)S, (int)H);
        }
    }

    return true;
};

bool HSI2RGB(cv::Mat& img){ //?
    //p.247
    if(type2str(img.type()) != "8UC3") return false;

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            double I = pixel[0]/255.0; //normalize to [0,1]
            double S = pixel[1]/255.0; //normalize to [0,1]
            double H = pixel[2]/255.0*2*M_PI; //normalize to [0,2*M_PI]

            double B, G, R;

            if(H >= 0 && H < M_PI*2.0/3){ //0 to 120 degrees
                //RG sector
                B = I * (1.0 - S);
                R = I * (1.0 + S*cos(H)/cos(M_PI/3.0 - H));
                G = 3.0*I - (R+B);
            }else if(H >= M_PI*2.0/3 && H < M_PI*4.0/3){ //120 to 240 degrees
                //GB sector
                H = H - M_PI*2.0/3;
                R = I * (1.0 - S);
                G = I * (1.0 + S*cos(H)/cos(M_PI/3.0 - H));
                B = 3.0*I - (R+G);
            }else if(H >= M_PI*4.0/3){ //larger than 240 degrees
                //BR sector
                H = H - M_PI*4.0/3;
                G = I * (1.0 - S);
                B = I * (1.0 + S*cos(H)/cos(M_PI/3.0 - H));
                R = 3.0*I - (G+B);
            }

            R *= 255.0;
            G *= 255.0;
            B *= 255.0;

            img.at<cv::Vec3b>(row, col) = cv::Vec3b((int)B, (int)G, (int)R);
        }
    }

    return true;
};

bool RGB2HSV(cv::Mat& img){
    //p.250
    if(type2str(img.type()) != "8UC3") return false;

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            double B = pixel[0]/255.0;
            double G = pixel[1]/255.0;
            double R = pixel[2]/255.0;

            double minColor = min({B, G, R});
            double maxColor = max({B, G, R});

            double H, S, V;

            V = maxColor;

            if(maxColor == 0){
                S = 0;
                H = 0; //undefined color
            }else{
                S = (maxColor - minColor)/maxColor;
                if(R == maxColor){
                    H = (G-B)/(maxColor-minColor)*M_PI/3.0;
                }else if(G == maxColor){
                    H = (2 + (B-R)/(maxColor-minColor))*M_PI/3.0;
                }else if(B == maxColor){
                    H = (4 + (R-G)/(maxColor-minColor))*M_PI/3.0;
                }
                if(H < 0){
                    H += 2*M_PI;
                }
            }

            //normalize
            H = H/(2.0*M_PI) * 255.0;
            S = S * 255.0;
            V = V * 255.0;

            //put H in R channel, S in G channel, V in B channel
            img.at<cv::Vec3b>(row, col) = cv::Vec3b((int)V, (int)S, (int)H);
        }
    }

    return true;
};

bool HSV2RGB(cv::Mat& img){
    //p.250
    if(type2str(img.type()) != "8UC3") return false;

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            double V = pixel[0]/255.0; //normalize to [0,1]
            double S = pixel[1]/255.0; //normalize to [0,1]
            double H = pixel[2]/255.0*2*M_PI; //normalize to [0,2*M_PI]

            double B, G, R;

            int i = H/(M_PI/3.0); //falls in which sector
            double f = H/(M_PI/3.0) - i;
            double p = V * (1-S);
            double q = V * (1-f*S);
            double t = V * (1 - (1-f) * S);

            switch(i){
                case 0:
                    R = V;
                    G = t;
                    B = p;
                    break;
                case 1:
                    R = q;
                    G = V;
                    B = p;
                    break;
                case 2:
                    R = q;
                    G = V;
                    B = t;
                    break;
                case 3:
                    R = p;
                    G = q;
                    B = V;
                    break;
                case 4:
                    R = t;
                    G = p;
                    B = V;
                    break;
                case 5:
                    R = V;
                    G = p;
                    B = q;
                    break;
            }

            R *= 255.0;
            G *= 255.0;
            B *= 255.0;

            img.at<cv::Vec3b>(row, col) = cv::Vec3b((int)B, (int)G, (int)R);
        }
    }

    return true;
};

bool RGB2YUV(cv::Mat& img){
    //p.256
    if(type2str(img.type()) != "8UC3") return false;

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            double B = pixel[0];
            double G = pixel[1];
            double R = pixel[2];

            double Y = 0.299*R + 0.587*G + 0.114*B;
            double U = 0.567*(B-Y);
            double V = 0.713*(R-Y);

            //truncate value exceed the range, it seems we can use a better normalization method?
            Y = min(max((int)Y, 0), 255); 
            U = min(max((int)U, 0), 255); 
            V = min(max((int)V, 0), 255);
            
            img.at<cv::Vec3b>(row, col) = cv::Vec3b((int)V, (int)U, (int)Y);
        }
    }

    return true;
};

bool YUV2RGB(cv::Mat& img){
    //p.259
    if(type2str(img.type()) != "8UC3") return false;

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            double V = pixel[0];
            double U = pixel[1];
            double Y = pixel[2];

            double R = Y + 1.402 * V;
            double G = Y - 0.344 * U - 0.714 * V;
            double B = Y + 1.772 * U;

            R = min(max((int)R, 0), 255);
            G = min(max((int)G, 0), 255);
            B = min(max((int)B, 0), 255);

            img.at<cv::Vec3b>(row, col) = cv::Vec3b((int)B, (int)G, (int)R);
        }
    }

    return true;
};

bool RGB2YIQ(cv::Mat& img){
    //p.261
    if(type2str(img.type()) != "8UC3") return false;

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            double B = pixel[0];
            double G = pixel[1];
            double R = pixel[2];

            double Y = 0.299*R + 0.587*G + 0.114*B;
            double I = 0.596*R - 0.274*G - 0.322*B;
            double Q = 0.211*R - 0.523*G + 0.312*B;

            //truncate value exceed the range, it seems we can use a better normalization method?
            Y = min(max((int)Y, 0), 255); 
            I = min(max((int)I, 0), 255); 
            Q = min(max((int)Q, 0), 255);
            
            img.at<cv::Vec3b>(row, col) = cv::Vec3b((int)Q, (int)I, (int)Y);
        }
    }

    return true;
};

bool YIQ2RGB(cv::Mat& img){
    //p.263
    if(type2str(img.type()) != "8UC3") return false;

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            double Q = pixel[0];
            double I = pixel[1];
            double Y = pixel[2];

            double R = Y + 0.956*I + 0.114*Q;
            double G = Y - 0.272*I - 0.647*Q;
            double B = Y - 1.106*I + 1.703*Q;

            R = min(max((int)R, 0), 255);
            G = min(max((int)G, 0), 255);
            B = min(max((int)B, 0), 255);

            img.at<cv::Vec3b>(row, col) = cv::Vec3b((int)B, (int)G, (int)R);
        }
    }

    return true;
};

void compensate(cv::Mat& img){
    //p.266

};

void colorBalance(cv::Mat& img){
    //p.268
};

int main(){
    cv::Mat img_color = cv::imread("images/Lenna.png");
    cv::Mat work_color = img_color.clone();
    bool isSave = false;

    cv::Mat cmy = work_color.clone();
    RGB2CMY(cmy);
    cv::Mat cmy_back_color = cmy.clone();
    CMY2RGB(cmy_back_color);
    vector<cv::Mat> cmys = {work_color, cmy, cmy_back_color};
    ShowHorizontal(cmys, "CMY vs CMY BACK RGB", isSave);

    cv::Mat hsi = work_color.clone();
    RGB2HSI(hsi);
    cv::Mat hsi_back_color = hsi.clone();
    HSI2RGB(hsi_back_color);
    vector<cv::Mat> hsis = {work_color, hsi, hsi_back_color};
    ShowHorizontal(hsis, "HSI vs HSI BACK RGB", isSave);

    cv::Mat hsv = work_color.clone();
    RGB2HSV(hsv);
    cv::Mat hsv_back_color = hsv.clone();
    HSV2RGB(hsv_back_color);
    vector<cv::Mat> hsvs = {work_color, hsv, hsv_back_color};
    ShowHorizontal(hsvs, "HSV vs HSV BACK RGB", isSave);

    cv::Mat yuv = work_color.clone();
    RGB2YUV(yuv);
    cv::Mat yuv_back_color = yuv.clone();
    YUV2RGB(yuv_back_color);
    vector<cv::Mat> yuvs = {work_color, yuv, yuv_back_color};
    ShowHorizontal(yuvs, "YUV vs YUV BACK RGB", isSave);

    cv::Mat yiq = work_color.clone();
    RGB2YIQ(yiq);
    cv::Mat yiq_back_color = yiq.clone();
    YIQ2RGB(yiq_back_color);
    vector<cv::Mat> yiqs = {work_color, yiq, yiq_back_color};
    ShowHorizontal(yiqs, "YIQ vs YIQ BACK RGB", isSave);

}
