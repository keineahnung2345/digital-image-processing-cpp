#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> //line
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h> //CV_8UC3
#include <iostream>
#include <map>
#include <cmath> //M_PI
#include <iomanip> //setw
#include "utility.h"

using namespace std;

bool ImMove(cv::Mat& img, int tcol, int trow){
    //p.105
    //img could be color or grayscale
    //tcol: the translate distance across columns
    //trow: the translate distance across rows
    if(abs(tcol) >= img.cols || abs(trow) >= img.rows){
        cout << "The absolute value of x and y should not exceed the image's width and height!" << endl;
        return false;
    }

    int channels = img.channels();
    cv::Mat target;
    int maxr = INT_MIN, maxc = INT_MIN;
    if(channels == 1){
#ifdef MoveResize
        target = cv::Mat(cv::Size(img.cols*2, img.rows*2), CV_8UC1, cv::Scalar(0));
        for(int r = 0; r < img.rows*2; r++){
            for(int c = 0; c < img.cols*2; c++){
#else
        target = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC1, cv::Scalar(0));
        for(int r = 0; r < img.rows; r++){
            for(int c = 0; c < img.cols; c++){
#endif
                if(r - trow >= 0 && r - trow < img.rows && c - tcol >= 0 && c - tcol < img.cols){
                    //(r - trow, c - tcol): the corresponding coordinate on source image
                    target.at<uchar>(r, c) = img.at<uchar>(r - trow, c - tcol);
                    maxr = max(r, maxr);
                    maxc = max(r, maxc);
                }else{
                    target.at<uchar>(r, c) = 255; //or 0?
                }
            }
        }
    }else if(channels == 3){
#ifdef MoveResize
        target = cv::Mat(cv::Size(img.cols*2, img.rows*2), CV_8UC3, cv::Vec3b(0, 0, 0));
        for(int r = 0; r < img.rows*2; r++){
            for(int c = 0; c < img.cols*2; c++){
#else
        target = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3, cv::Vec3b(0, 0, 0));
        for(int r = 0; r < img.rows; r++){
            for(int c = 0; c < img.cols; c++){
#endif
                if(r - trow >= 0 && r - trow < img.rows && c - tcol >= 0 && c - tcol < img.cols){
                    //(r - trow, c - tcol): the corresponding coordinate on source image
                    target.at<cv::Vec3b>(r, c) = img.at<cv::Vec3b>(r - trow, c - tcol);
                    maxr = max(r, maxr);
                    maxc = max(r, maxc);
                }else{
                    target.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255); //or 0?
                }
            }
        }
    }

#ifdef MoveResize
    cout << "After rotating, crop to: " << maxc << " " << maxr << endl;
    target = target(cv::Rect(0, 0, maxc, maxr));
#endif

    img = target;
}

void HorMirror(cv::Mat& img){
    //p.108
    int channels = img.channels();
    cv::Mat target;
    if(channels == 1){
        target = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC1, cv::Scalar(0));
        for(int r = 0; r < img.rows; r++){
            for(int c = 0; c < img.cols; c++){
                target.at<uchar>(r, c) = img.at<uchar>(r, img.cols - c);
            }
        }
    }else if(channels == 3){
        target = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3, cv::Vec3b(0, 0, 0));
        for(int r = 0; r < img.rows; r++){
            for(int c = 0; c < img.cols; c++){
                target.at<cv::Vec3b>(r, c) = img.at<cv::Vec3b>(r, img.cols - c);
            }
        }
    }
    img = target;
}

void VerMirror(cv::Mat& img){
    //p.109
    int channels = img.channels();
    cv::Mat target;
    if(channels == 1){
        target = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC1, cv::Scalar(0));
        for(int r = 0; r < img.rows; r++){
            for(int c = 0; c < img.cols; c++){
                target.at<uchar>(r, c) = img.at<uchar>(img.rows - r, c);
            }
        }
    }else if(channels == 3){
        target = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3, cv::Vec3b(0, 0, 0));
        for(int r = 0; r < img.rows; r++){
            for(int c = 0; c < img.cols; c++){
                target.at<cv::Vec3b>(r, c) = img.at<cv::Vec3b>(img.rows - r, c);
            }
        }
    }
    img = target;
}

void Scale(cv::Mat& img, double times){
    //p.113
    int channels = img.channels();
    cv::Mat target;
    if(channels == 1){
        target = cv::Mat(cv::Size((int)(img.cols*times), (int)(img.rows*times)), CV_8UC1, cv::Scalar(0));
        // cout << target.rows << " * " << target.cols << endl;
        for(int r = 0; r < target.rows; r++){
            for(int c = 0; c < target.cols; c++){
                int srcr = ceil(r/times), srcc = ceil(c/times);
                // cout << srcr << " " << srcc << " | ";
                if(srcr >= 0 && srcr < img.rows && srcc >= 0 && srcc < img.cols){
                    target.at<uchar>(r, c) = img.at<uchar>(srcr, srcc);
                }else{
                    target.at<uchar>(r, c) = 255;
                }
                
            }
        }
    }else if(channels == 3){
        target = cv::Mat(cv::Size((int)(img.cols*times), (int)(img.rows*times)), CV_8UC3, cv::Vec3b(0, 0, 0));
        // cout << target.rows << " * " << target.cols << endl;
        for(int r = 0; r < target.rows; r++){
            for(int c = 0; c < target.cols; c++){
                int srcr = ceil(r/times), srcc = ceil(c/times);
                // cout << srcr << " " << srcc << " | ";
                if(srcr >= 0 && srcr < img.rows && srcc >= 0 && srcc < img.cols){
                    target.at<cv::Vec3b>(r, c) = img.at<cv::Vec3b>(srcr, srcc);
                }else{
                    target.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
                    // cout << srcr << " " << srcc << " | ";
                }
                // cout << (int)target.at<uchar>(r, c) << " ";
            }
        }
    }
    img = target;
}

void Rotate(cv::Mat& img, float angle){
    //left top corner is the original point
    //p.117
    /*
    transform
    (x', y') = (xcos-ysin, sinx+cosy)
    inverse transform
    (x, y) = (x'cos+y'sin, -sinx'+cosy')
    */
    int channels = img.channels();
    cv::Mat target;
    if(channels == 1){
        target = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC1, cv::Scalar(0));
    }else if(channels == 3){
        target = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3, cv::Vec3b(0, 0, 0));
    }

    for(int r = 0; r < img.rows; r++){
        for(int c = 0; c < img.cols; c++){
            int src_r = r*cos(angle*M_PI/180.0)+c*sin(angle*M_PI/180.0);
            int src_c = r*(-sin(angle*M_PI/180.0))+c*cos(angle*M_PI/180.0);
            if(channels == 1){
                if(src_r >= 0 && src_r < img.rows && src_c >= 0 && src_c < img.cols){
                    target.at<uchar>(r, c) = img.at<uchar>(src_r, src_c);
                }else{
                    target.at<uchar>(r, c) = 255;
                }
            }else if(channels == 3){
                if(src_r >= 0 && src_r < img.rows && src_c >= 0 && src_c < img.cols){
                    target.at<cv::Vec3b>(r, c) = img.at<cv::Vec3b>(src_r, src_c);
                }else{
                    target.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
                }
            }
        }
    }
    img = target;
}

void Rotate(cv::Mat& img, float angle, int center_col, int center_row){
    //still not work
    //(center_i, center_j) is the original point
    //p.115
    /*
    transform
    (x', y') = (xcos-ysin, sinx+cosy)
    inverse transform
    (x, y) = (x'cos+y'sin, -sinx'+cosy')
    */
    int channels = img.channels();
    if(channels == 1){
        //cv::Point(w, h): first argument is in left-right direction
        cv::circle(img, cv::Point(center_col, center_row), 10, cv::Scalar(0), cv::FILLED);
    }else if(channels == 3){
        cv::circle(img, cv::Point(center_col, center_row), 10, cv::Scalar(0, 0, 255), cv::FILLED);
    }

    ImMove(img, -center_col, -center_row);

    cv::Mat target;
    if(channels == 1){
        target = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC1, cv::Scalar(0));
    }else if(channels == 3){
        target = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3, cv::Vec3b(0, 0, 0));
    }

    for(int r = 0; r < img.rows; r++){
        for(int c = 0; c < img.cols; c++){
            int src_r = r*cos(angle*M_PI/180.0)+c*sin(angle*M_PI/180.0);
            int src_c = r*(-sin(angle*M_PI/180.0))+c*cos(angle*M_PI/180.0);
            if(channels == 1){
                if(src_r >= 0 && src_r < img.rows && src_c >= 0 && src_c < img.cols){
                    target.at<uchar>(r, c) = img.at<uchar>(src_r, src_c);
                }else{
                    target.at<uchar>(r, c) = 255;
                }
            }else if(channels == 3){
                if(src_r >= 0 && src_r < img.rows && src_c >= 0 && src_c < img.cols){
                    target.at<cv::Vec3b>(r, c) = img.at<cv::Vec3b>(src_r, src_c);
                }else{
                    target.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
                }
            }
        }
    }

    ImMove(target, center_col, center_row);

    img = target;
}

int InterpBilinear(cv::Mat& img, double x, double y){
    //p.120
    //return the interpolation value for (x, y)
    int width = img.cols, height = img.rows;
    if(x < 0 || x >= width || y < 0 || y >= height){
        return -1;
    }
    int x1 = x; //round down
    int x2 = x1+1; //round up
    int y1= y;
    int y2 = y1+1;

    /*
    Note that (x, y) resides in the pixel (x1, y1), 
    and x2 or y2 may exceed the range of the image
    */

    if(x2 >= width){
        //on the right edge
        if(y2 >= height){
            //on the bottom-right corner
            // cout << "bottom-right corner" << endl;
            return img.at<uchar>(x1, y1);
        }else{
            //on right edge, not at corner, interpolate once
            int ft = img.at<uchar>(x1, y1);
            int fb = img.at<uchar>(x1, y2);
            cout << "on right edge, top: " << ft << ", bottom: " << fb << ", dist: " << y-y1 << endl;
            return (int)(ft + (fb - ft) * (y - y1));
        }
    }else if(y2 >= height){
        //on the bottom edge, interpolate once
        //y2 is invalid
        int fl = img.at<uchar>(x1, y1);
        int fr = img.at<uchar>(x2, y1);
        cout << "on bottom edge, left: " << fl << ", right: " << fr << ", dist: " << x - x1 << endl;
        return (int)(fl + (fr - fl) * (x - x1));
    }else{
        //not at edge or corner, interpolate twice
        int flt = img.at<uchar>(x1, y1);
        int flb = img.at<uchar>(x1, y2);
        int frt = img.at<uchar>(x2, y1);
        int frb = img.at<uchar>(x2, y2);

        int fl = flt + (flb - flt) * (y - y1);
        int fr = frt + (frb - frt) * (y - y1);

        cout << "not on edge or corner, lt: " << flt << ", lb: " << flb << ", rt: " << frt << ", rb: " << frb << endl;
        cout << "interpolate value on left: " << fl << ", on right: " << fr << ", dist: " << x - x1 << endl;
        return (int)(fl + (fr - fl) * (x - x1));
    }
}

void InterpCubic(double x, double y){
    //p.122, Matlab
}

void PrintVector(vector<double>& row){
    for(int i = 0; i < row.size(); i++){
        cout << setw(7) << fixed << setprecision(2) << row[i];
    }
    cout << endl;
};

void PrintMatrix(vector<vector<double>>& matrix){
    for(int i = 0; i < matrix.size(); i++){
        PrintVector(matrix[i]);
    }
    cout << endl;
}

void SwapRow(vector<double>& row1, vector<double>& row2){
    int n = row1.size();
    for(int i = 0; i < n; i++){
        swap(row1[i], row2[i]);
    }
}

void MultiplyRow(vector<double>& row, double ratio){
    for(int i = 0; i < row.size(); i++){
        row[i] *= ratio;
    }
}

void MultiplySubstractRow(vector<double>& row1, vector<double>& row2, double ratio){
    //row1 = row1 - row2 * ratio
    for(int i = 0; i < row1.size(); i++){
        row1[i] -= row2[i] * ratio;
    }
}

bool InvMat(vector<vector<double>>& matrix){
    //https://ccjou.wordpress.com/2013/02/20/%E9%AB%98%E6%96%AF%E6%B6%88%E5%8E%BB%E6%B3%95/
    //https://www.geeksforgeeks.org/gaussian-elimination/
    //https://www.geeksforgeeks.org/program-for-gauss-jordan-elimination-method/
    //https://www.geeksforgeeks.org/finding-inverse-of-a-matrix-using-gauss-jordan-method/
    int n = matrix.size();

    //create augmented matrix
    vector<vector<double>> augMatrix = vector<vector<double>>(n, vector<double>(2 * n, 0.0));
    for(int i = 0; i < n; i++){
        //copy from input matrix
        for(int j = 0; j < n; j++){
            augMatrix[i][j] = matrix[i][j];
        }
        //the right part is identity matrix
        for(int j = n; j < 2*n; j++){
            //elements on diagonal are 1, others are 0
            augMatrix[i][j] = (int)(j-n == i);
        }
    }

    for(int row = 0; row < n; row++){
        //ensure the diagonal of augmented matrix's left part contains no 0
        //?
        if(augMatrix[row][row] == 0){
            int row2;
            for(row2 = row+1; row2 < n; row2++){
                if(augMatrix[row2][row] != 0){
                    //swap row i and row j
                    SwapRow(augMatrix[row], augMatrix[row2]);
                    break;
                }
            }
            //the column is all 0
            if(row2 == n) return false;
        }
        
        //R[row2] = R[row2] - ratio * R[row]
        for(int row2 = 0; row2 < n; row2++){
            //don't operate with itself!
            if(row == row2)continue;
            //make the column [?][row] all zero besides [row][row]
            double ratio = augMatrix[row2][row]/augMatrix[row][row];
            MultiplySubstractRow(augMatrix[row2], augMatrix[row], ratio);
        }
        
        //make diagonal all 1
        if(augMatrix[row][row] != 1){
            double ratio = 1.0/augMatrix[row][row];
            MultiplyRow(augMatrix[row], ratio);
        }
    }

    // //R[j] = R[j] - k * R[i]
    // for(int row1 = 0; row1 < n; row1++){
    //     for(int row2 = 0; row2 < n; row2++){
    //         //don't operate with itself!
    //         if(row1 == row2)continue;
    //         //make column row1 all zero besides [row1][row1]
    //         double ratio = augMatrix[row2][row1]/augMatrix[row1][row1];
    //         // augMatrix[row2] = augMatrix[row2] - ratio * augMatrix[row1];
    //         for(int col = 0; col < 2*n; col++){
    //             augMatrix[row2][col] -= ratio * augMatrix[row1][k];
    //         }
    //     }
    // }

    // //make diagonal all 1
    // for(int row = 0; row < n; row++){
    //     if(augMatrix[row][row] != 1){
    //         double ratio = 1.0/augMatrix[row][row];
    //         for(int col = 0; col < 2*n; col++){
    //             augMatrix[row][col] *= ratio;
    //         }
    //     }
    // }

    //copy back to input matrix
    for(int row = 0; row < n; row++){
        for(int col = 0; col < n; col++){
            matrix[row][col] = augMatrix[row][col+n];
        }
    }

    return true;
}

bool ProdMat(vector<vector<double>>& mat1, vector<vector<double>>& mat2, vector<vector<double>>& res){
    int m = mat1.size(), n = mat1[0].size(), p = mat2[0].size();
    if(n != mat2.size()) return false;

    res = vector<vector<double>>(m, vector<double>(p, 0.0));

    for(int i = 0; i < m; i++){
        for(int j = 0; j < p; j++){
            double val = 0.0;
            for(int k = 0; k < n; k++){
                val += mat1[i][k] * mat2[k][j];
            }
            res[i][j] = val;
        }
    }

    return true;
};

// void cp2tform(vector<vector<double>>& input_points, vector<vector<double>>& base_points, vector<vector<double>>& transform, string transform_type = "affine"){
//     //p.126, Matlab
//     //fit a transform from input_points to base_points
// }

void GetProjPara(vector<vector<double>>& basePoints, vector<vector<double>>& srcPoints, vector<double>& projectionPara){
    /*
    We want to project srcPoints onto basePoints,
    here we calculate the "INVERSE" transformation.
    It maps basePoints to srcPoints, later we will iterating target image,
    and use this transformation to find out which source point do we need for interpolation. 

    We use bilinear equation, so we need at least 4 pairs of base and source points
    There will be 4 * 2 = 8 parameters in the result "projectionPara"
    */

    vector<vector<double>> coefMatrix(4, vector<double>(4, 0.0));
    for(int i = 0; i < 4; i++){
        coefMatrix[i][0] = basePoints[i][0]; //x coordinate
        coefMatrix[i][1] = basePoints[i][1]; //y coordinate
        coefMatrix[i][2] = basePoints[i][0] * basePoints[i][1]; //x * y
        coefMatrix[i][3] = 1;
    }

    //calculate its inverse matrix in-place
    InvMat(coefMatrix);

    vector<vector<double>> srcXs(4, vector<double>(1, 0.0));
    for(int i = 0; i < 4; i++){
        srcXs[i][0] = srcPoints[i][0];
    }

    vector<vector<double>> c1234;
    //coefMatrix: 4x4, srcXs: 4x1, c1234: 4x1
    ProdMat(coefMatrix, srcXs, c1234);

    vector<vector<double>> srcYs(4, vector<double>(1, 0.0));
    for(int i = 0; i < 4; i++){
        srcYs[i][0] = srcPoints[i][1];
    }

    vector<vector<double>> c5678;
    //coefMatrix: 4x4, srcYs: 4x1, c5678: 4x1
    ProdMat(coefMatrix, srcYs, c5678);

    //copy the result to projectionPara
    for(int i = 0; i < 4; i++){
        projectionPara.push_back(c1234[i][0]);
    }
    for(int i = 0; i < 4; i++){
        projectionPara.push_back(c5678[i][0]);
    }
};

void ProjTrans(vector<double>& srcPoint, vector<double>& projectionPara, vector<double>& dstPoint){
    //p.135
    //transform source points by the give projection parameters to get destination point
    dstPoint = vector<double>(2, 0.0);

    dstPoint[0] = srcPoint[0] *               projectionPara[0] +
                  srcPoint[1] *               projectionPara[1] +
                  srcPoint[0] * srcPoint[1] * projectionPara[2] +
                  1 *                         projectionPara[3];

    dstPoint[1] = srcPoint[0] *               projectionPara[4] +
                  srcPoint[1] *               projectionPara[5] +
                  srcPoint[0] * srcPoint[1] * projectionPara[6] +
                  1 *                         projectionPara[7];
};

bool ImProjRestore(cv::Mat& img, vector<vector<double>>& basePoints, vector<vector<double>>& srcPoints, bool isInterp){
    vector<double> projectionPara;
    GetProjPara(basePoints, srcPoints, projectionPara);
    
    int height = img.rows, width = img.cols;

    cv::Mat dst(cv::Size(width, height), CV_8UC1, cv::Scalar(0));

    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            vector<double> dstPoint = {(double)row, (double)col};
            vector<double> srcPoint;
            ProjTrans(dstPoint, projectionPara, srcPoint);
            if(isInterp){
                //interpolation
                int gray = InterpBilinear(img, srcPoint[0], srcPoint[1]);
                if(gray >= 0){
                    //it's valid
                    dst.at<uchar>(row, col) = gray;
                }else{
                    //it's invalid
                    dst.at<uchar>(row, col) = 255;
                }
            }else{
                //nearest interpolation
                int srcRow = round(srcPoint[0]);
                int srcCol = round(srcPoint[1]);
                if(srcRow >= 0 && srcRow < height && srcCol >= 0 && srcCol < width){
                    dst.at<uchar>(row, col) = img.at<uchar>(srcRow, srcCol);
                }else{
                    dst.at<uchar>(row, col) = 255;
                }
            }
        }
    }

    img = dst;
};


void fitBasePoints(cv::Mat& img, vector<vector<double>>& basePoints){
    //start from left-top corner, clock-wise
    // basePoints = {
    //     {0,0},
    //     {0,31*9},
    //     {10*9,31*9},
    //     {10*9,0}
    // };
    int w = img.cols, h = img.rows;
    //x's center, image's center
    double ratio = min(w/310.0, h/100.0);
    double xl = ratio * (w/2 - 310.0/2);
    double xr = ratio * (w/2 + 310.0/2);
    double yt = ratio * (h/2 - 100.0/2);
    double yb = ratio * (h/2 + 100.0/2);
    basePoints = {
        {yt, xl},
        {yt, xr},
        {yb, xr},
        {yb, xl}
    };
};

int main(){
    cv::Mat img_color = cv::imread("images/Lenna.png");
    cv::Mat img_gray = cv::imread("images/Lenna.png", 0);
    cv::Mat work_color = img_color.clone();
    cv::Mat work_gray = img_gray.clone();
    bool isSave = false;

    //Move
    cout << "Please input the x and y to move the image..." << endl;
    int mx, my;
    cin >> mx >> my;
    string moveTitle = string("Move") + " " + to_string(mx) + " " + to_string(my);
    work_color = img_color.clone();
    ImMove(work_color, mx, my);
    Show(work_color, moveTitle + " color", isSave);
    work_gray = img_gray.clone();
    ImMove(work_gray, mx, my);
    Show(work_gray, moveTitle + " gray", isSave);

    //Horizontal Mirror
    cout << "Horizontal Mirror" << endl;
    work_color = img_color.clone();
    HorMirror(work_color);
    Show(work_color, "Horizontal Mirror color", isSave);
    work_gray = img_gray.clone();
    HorMirror(work_gray);
    Show(work_gray, "Horizontal Mirror gray", isSave);

    //Vertical Mirror
    cout << "Vertical Mirror" << endl;
    work_color = img_color.clone();
    VerMirror(work_color);
    Show(work_color, "Vertical Mirror color", isSave);
    work_gray = img_gray.clone();
    VerMirror(work_gray);
    Show(work_gray, "Vertical Mirror gray", isSave);

    //Scale
    cout << "Please input the ratio for scaling..." << endl;
    double ratio;
    cin >> ratio;
    string scaleTitle = string("Scale") + " " + to_string_with_precision(ratio, 2);
    cout << "Scale" << endl;
    work_color = img_color.clone();
    Scale(work_color, ratio);
    Show(work_color, scaleTitle + " color", isSave);
    work_gray = img_gray.clone();
    Scale(work_gray, ratio);
    Show(work_gray, scaleTitle + " gray", isSave);

    //Rotate
    cout << "Please input the angle for rotating..." << endl;
    double angle;
    cin >> angle;
    string rotateTitle = string("Rotate") + " " + to_string_with_precision(angle, 2);
    cout << "Rotate" << endl;
    work_color = img_color.clone();
    Rotate(work_color, angle);
    Show(work_color, rotateTitle + " color", isSave);
    work_gray = img_gray.clone();
    Rotate(work_gray, angle);
    Show(work_gray, rotateTitle + " gray", isSave);

    // //Rotate around any point
    // cout << "Please input the angle for rotating..." << endl;
    // double angle_any;
    // cin >> angle_any;
    // cout << "Please input the center for rotating..." << endl;
    // int center_i, center_j;
    // cin >> center_i >> center_j;
    // cout << "Rotate" << endl;
    // work_color = img_color.clone();
    // Rotate(work_color, angle_any, center_i, center_j);
    // Show(work_color, "Rotate around any point", isSave);
    // work_gray = img_gray.clone();
    // Rotate(work_gray, angle_any, center_i, center_j);
    // Show(work_gray, "Rotate around any point", isSave);
    
    // //Bilinear interpolation
    // cout << "Please input the (x, y) coordinate for calculating bilinear interpolation..." << endl;
    // double cx, cy;
    // cin >> cx >> cy;
    // work_gray = img_gray.clone();
    // double val = InterpBilinear(work_gray, cx, cy);
    // cout << "Final interpolated value: " << val << endl;

    // //Matrix Inverse
    // vector<vector<double>> mat = {
    //     {0,7,9},
    //     {4,0,8},
    //     {7,5,0}
    // };

    // cout << "Original matrix: " << endl;
    // PrintMatrix(mat);
    // vector<vector<double>> inv = mat;
    // InvMat(inv);
    // cout << "Inversed matrix: " << endl;
    // PrintMatrix(inv);

    // vector<vector<double>> prod;
    // ProdMat(mat, inv, prod);
    // cout << "Their product: " << endl;
    // PrintMatrix(prod);
    
    //Image Projection Restore
    // p.128-139
    for(int i = 1; i <= 2; i++){
        string picName = "images/license_plate" + to_string(i) + ".jfif";
        cv::Mat license_plate = cv::imread(picName, 0);

        //Calculate bilinear projection parameters
        vector<vector<double>> basePoints;
        fitBasePoints(license_plate, basePoints);

        vector<vector<double>> srcPoints;
        
        if(i == 1){
            srcPoints = {
                {51,94},
                {54,203},
                {121,195},
                {102,88}
            };
        }else if(i == 2){
            srcPoints = {
                {51,11},
                {5,237},
                {92,252},
                {173,51}
            };
        }

        vector<double> projectionPara;
        GetProjPara(basePoints, srcPoints, projectionPara);
        // PrintVector(projectionPara);

        for(int i = 0; i < basePoints.size(); i++){
            vector<double> dstPoint;
            //Note taht "projectionPara" map basePoint to srcPoint!
            ProjTrans(basePoints[i], projectionPara, dstPoint);
            // cout << srcPoints[i][0] << ", " << srcPoints[i][1] << " v.s. " << dstPoint[0] << ", " << dstPoint[1] << endl;
        }

        cv::Mat restored = license_plate.clone();
        ImProjRestore(restored, basePoints, srcPoints, false);
        vector<cv::Mat> RestoredImages = {license_plate, restored};
        // cout << license_plate.rows << " x " << license_plate.cols << endl;
        // cout << restored.rows << " x " << restored.cols << endl;
        ShowHorizontal(RestoredImages, string("Projection Restore") + " " + to_string(i), isSave);
    }


    return 0;
}
                
