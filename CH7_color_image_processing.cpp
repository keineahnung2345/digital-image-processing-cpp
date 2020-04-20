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

            // given in textbook??
            // if(I < 0.078431){
            //     S = 0;
            // }else if(I > 0.92){
            //     S = 0;
            // }else{
            //     S = 1 - 3*minColor/(R+G+B);
            // }

            if(minColor == maxColor){
                H = 0;
                S = 0;
            }else{
                //its unit is radiance
                double cosVal = (R-G+R-B)/2.0 / sqrt((R-G)*(R-G) + (R-B)*(G-B) + eps);
                double radiance = (abs(cosVal-1) < eps) ? 0.0 : (abs(cosVal-(-1)) < eps) ? M_PI : acos(cosVal);
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

bool compensate(cv::Mat& img){
    //p.266
    /*
    select most red, green, blue points int the image and 
    learn a linear transformation to convert them to true r,g,b
    */
    if(type2str(img.type()) != "8UC3") return false;

    int rMaxR = INT_MIN;
    int cMaxR = INT_MIN;
    int rMaxG = INT_MIN;
    int cMaxG = INT_MIN;
    int rMaxB = INT_MIN;
    int cMaxB = INT_MIN;
    double maxR = std::numeric_limits<double>::lowest();
    double maxG = std::numeric_limits<double>::lowest();
    double maxB = std::numeric_limits<double>::lowest();

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            double B = pixel[0];
            double G = pixel[1];
            double R = pixel[2];
            
            if(R > maxR){
                rMaxR = row;
                cMaxR = col;
                maxR = R;
            }
            
            if(G > maxG){
                rMaxG = row;
                cMaxG = col;
                maxG = G;
            }
            
            if(B > maxB){
                rMaxB = row;
                cMaxB = col;
                maxB = B;
            }
        }
    }

    //old three point's rgb value
    double r1 = maxR, g1 = img.at<cv::Vec3b>(rMaxR, cMaxR)[1], b1 = img.at<cv::Vec3b>(rMaxR, cMaxR)[0];
    double r2 = img.at<cv::Vec3b>(rMaxG, cMaxG)[2], g2 = maxG, b2 = img.at<cv::Vec3b>(rMaxG, cMaxG)[0];
    double r3 = img.at<cv::Vec3b>(rMaxB, cMaxB)[2], g3 = img.at<cv::Vec3b>(rMaxB, cMaxB)[1], b3 = maxB;

    vector<vector<double>> A1 = {
        {r1, r2, r3},
        {g1, g2, g3},
        {b1, b2, b3}
    };

    //which does the best?
    //given by textbook, brightness will be lower?
    double R = 0.3*r1 + 0.59*g1 + 0.11*b1;
    double G = 0.3*r2 + 0.59*g2 + 0.11*b2;
    double B = 0.3*r3 + 0.59*g3 + 0.11*b3;

    //*3 to restore its brightness?
    // double R = (0.3*r1 + 0.59*g1 + 0.11*b1)*3;
    // double G = (0.3*r2 + 0.59*g2 + 0.11*b2)*3;
    // double B = (0.3*r3 + 0.59*g3 + 0.11*b3)*3;

    //assume each channel contribute same brightness?
    // double R = (r1+b1+g1)/3.0;
    // double G = (r2+b2+g2)/3.0;
    // double B = (r3+b3+g3)/3.0;

    //*3 to restore its brightness?
    // double R = (r1+b1+g1);
    // double G = (r2+b2+g2);
    // double B = (r3+b3+g3);

    vector<vector<double>> A2 = {
        {R, 0, 0},
        {0, G, 0},
        {0, 0, B}
    };

    vector<vector<double>> invA1 = A1;
    InvMat(invA1);
    vector<vector<double>> invA2 = A2;
    InvMat(invA2);
    vector<vector<double>> C;
    ProdMat(A1, invA2, C);
    vector<vector<double>> invC = C;
    InvMat(invC);

    // cout << "A1" << endl;
    // PrintMatrix(A1);

    // cout << "A2" << endl;
    // PrintMatrix(A2);

    // cout << "invA2" << endl;
    // PrintMatrix(invA2);

    // cout << "C" << endl;
    // PrintMatrix(C);

    // cout << "invC" << endl;
    // PrintMatrix(invC);

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            double B = pixel[0];
            double G = pixel[1];
            double R = pixel[2];
            
            vector<vector<double>> oldColor = {{R}, {G}, {B}}; // 3 x 1 matrix
            vector<vector<double>> newColor; //also a 3 x 1 matrix
            // ProdMat(invC, oldColor, newColor);
            ProdMat(invA1, oldColor, newColor);
            oldColor = newColor;
            ProdMat(A2, oldColor, newColor);

            for(int i = 0; i < 3; i++){
                newColor[i][0] = min(max((int)newColor[i][0], 0), 255); ;
            }

            // if(row == 0 && col % (img.cols/10) == 0){
            //     cout << "old" << endl;
            //     PrintMatrix(oldColor);
            //     cout << "new" << endl;
            //     PrintMatrix(newColor);
            // }

            img.at<cv::Vec3b>(row, col) = cv::Vec3b((int)newColor[2][0], (int)newColor[1][0], (int)newColor[0][0]);
        }
    }

    return true;
};

bool colorBalance(cv::Mat& img){
    //p.268
    //fix green color, do linear transform on red and blue to make the two chosen points "gray"
    if(type2str(img.type()) != "8UC3") return false;

    cv::Vec3b F1 = img.at<cv::Vec3b>(0, 0);
    //select farther point to avoid "Floating point exception"
    cv::Vec3b F2 = img.at<cv::Vec3b>(50, 50);
    cv::Vec3b newF1(F1[1], F1[1], F1[1]);
    cv::Vec3b newF2(F2[1], F2[1], F2[1]);
    //red
    double K1 = (double)(newF1[2]-newF2[2])/(double)(F1[2]-F2[2]);
    double K2 = newF1[2] - K1 * F1[2];
    //blue
    double L1 = (double)(newF1[0]-newF2[0])/(double)(F1[0]-F2[0]);
    double L2 = newF1[0] - L1 * F1[0];

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            double B = pixel[0];
            double G = pixel[1];
            double R = pixel[2];

            double newR = min(max((int)(K1 * R + K2), 0), 255); 
            double newG = G;
            double newB = min(max((int)(L1 * B + K2), 0), 255); 
            
            img.at<cv::Vec3b>(row, col) = cv::Vec3b((int)newB, (int)newG, (int)newR);
        }
    }

    return true;
};

int main(){
    cv::Mat img_color = cv::imread("images/Lenna.png");
    // cv::Mat img_color = cv::imread("images/rgb.png");
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

    cv::Mat compensated = work_color.clone();
    compensate(compensated);
    vector<cv::Mat> compensateds = {work_color, compensated};
    ShowHorizontal(compensateds, "Before and after compensating", isSave);

    cv::Mat balanced = work_color.clone();
    colorBalance(balanced);
    vector<cv::Mat> balanceds = {work_color, balanced};
    ShowHorizontal(balanceds, "Before and after color balancing", isSave);
}
