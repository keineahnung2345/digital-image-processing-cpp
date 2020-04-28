#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> //line
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h> //CV_8UC3
#include <iostream>
#include <map>
#include <cmath> //M_PI
#include "CH3.h"
#include "CH5.h"
#include "CH8.h"
#include "CH9.h"

using namespace std;

map<string, Kernel*> kernels = {
    //horizontal edge
    {"PrewittH", new Kernel("PrewittH", {-1,-1,-1, 0,0,0, 1,1,1}, 1.0, true)}
    //vertical edge
    , {"PrewittV", new Kernel("PrewittV", {-1,0,1, -1,0,1, -1,0,1}, 1.0, true)}
    //45 degrees
    , {"PrewittCW", new Kernel("PrewittCW", {-1,-1,0, -1,0,1, 0,1,1}, 1.0, true)}
    //135 degrees
    , {"PrewittCCW", new Kernel("PrewittCCW", {0,1,1, -1,0,1, -1,-1,0}, 1.0, true)}

    //horizontal edge
    , {"SobelH", new Kernel("SobelH", {-1,-2,-1, 0,0,0, 1,2,1}, 1.0, true)}
    //vertical edge
    , {"SobelV", new Kernel("SobelV", {-1,0,1, -2,0,2, -1,0,1}, 1.0, true)}
    //45 degrees
    , {"SobelCW", new Kernel("SobelCW", {-2,-1,0, -1,0,1, 0,1,2}, 1.0, true)}
    //135 degrees
    , {"SobelCCW", new Kernel("SobelCCW", {0,1,2, -1,0,1, -2,-1,0}, 1.0, true)}

    //LoG
    , {"LogEdgeDetection", new Kernel("LogEdgeDetection", {0,0,-1,0,0, 0,-1,-2,-1,0, -1,-2,16,-2,-1, 0,-1,-2,-1,0, 0,0,-1,0,0}, 1.0)}
};

bool EdgePrewitt(cv::Mat& img, int thres, int edgeType, bool thinning, bool outputGradient){
    //used in EdgeCanny, but not given in textbook
    //different from CH8, black is bg, white is fg
    //it can accept grayscale image
    if(type2str(img.type()) != "8UC1") return false;

    int height = img.rows, width = img.cols;

    cv::Mat target = img;

    //used in case 0
    //cannot define variable under the block "case 0"?
    vector<Kernel*> values;
    values = {kernels["PrewittH"], kernels["PrewittV"], kernels["PrewittCW"], kernels["PrewittCCW"]};

    switch(edgeType){
        case 0:
            //not good for 45 degrees?
            FilterOp(target, values, true);
            break;
        case 1:
            FilterOp(target, {kernels["PrewittH"]}, true);
            break;
        case 2:
            FilterOp(target, {kernels["PrewittV"]}, true);
            break;
        case 3:
            //45 degrees
            FilterOp(target, {kernels["PrewittCW"]}, true);
            break;
        case 4:
            //135 degrees
            FilterOp(target, {kernels["PrewittCCW"]}, true);
            break;
        default:
            cout << "edgeType should be 0 to 4!" << endl;
            return false;
    }

    if(!outputGradient){
        //do post processing
        if(thres){
            Threshold(target, thres);
        }else{
            //auto threshold
            AutoThreshold(target);
        }

        if(thinning){
            //invert color
            LinTran(target, -1, 255);
            //"Thining" assumes black as fg, white as bg
            Thining(target);
            //invert color
            LinTran(target, -1, 255);
        }
    }

    img = target;
    return true;
};

bool EdgeSobel(cv::Mat& img, int thres, int edgeType, bool thinning, bool outputGradient){
    //p.344-345
    //different from CH8, black is bg, white is fg
    //it can accept grayscale image
    if(type2str(img.type()) != "8UC1") return false;

    int height = img.rows, width = img.cols;

    cv::Mat target = img;

    //used in case 0
    //cannot define variable under the block "case 0"?
    vector<Kernel*> values;
    values = {kernels["SobelH"], kernels["SobelV"], kernels["SobelCW"], kernels["SobelCCW"]};

    switch(edgeType){
        case 0:
            //not good for 45 degrees?
            FilterOp(target, values, true);
            break;
        case 1:
            FilterOp(target, {kernels["SobelH"]}, true);
            break;
        case 2:
            FilterOp(target, {kernels["SobelV"]}, true);
            break;
        case 3:
            //45 degrees
            FilterOp(target, {kernels["SobelCW"]}, true);
            break;
        case 4:
            //135 degrees
            FilterOp(target, {kernels["SobelCCW"]}, true);
            break;
        default:
            cout << "edgeType should be 0 to 4!" << endl;
            return false;
    }

    if(!outputGradient){
        //do post processing
        if(thres){
            Threshold(target, thres);
        }else{
            //auto threshold
            AutoThreshold(target);
        }

        if(thinning){
            //invert color
            LinTran(target, -1, 255);
            //"Thining" assumes black as fg, white as bg
            Thining(target);
            //invert color
            LinTran(target, -1, 255);
        }
    }

    img = target;
    return true;
};

bool EdgeLoG(cv::Mat& img, int thres, bool thinning, bool outputGradient){
    //p.346
    //different from CH8, black is bg, white is fg
    //it can accept grayscale image
    if(type2str(img.type()) != "8UC1") return false;

    int height = img.rows, width = img.cols;

    cv::Mat target = img;

    //used in case 0
    //cannot define variable under the block "case 0"?
    vector<Kernel*> values = {kernels["LogEdgeDetection"]};
    
    FilterOp(target, {kernels["LogEdgeDetection"]}, true);

    if(!outputGradient){
        //do post processing
        if(thres){
            Threshold(target, thres);
        }else{
            //auto threshold
            AutoThreshold(target);
        }

        if(thinning){
            //invert color
            LinTran(target, -1, 255);
            //"Thining" assumes black as fg, white as bg
            Thining(target);
            //invert color
            LinTran(target, -1, 255);
        }
    }

    img = target;
    return true;
};

enum class DIR{
    H = 0, //horizontal
    V, //vertical
    CW, //clockwise, 45 degrees
    CCW //counterclockwise, 135 degrees
};

bool EdgeCanny(cv::Mat& img, int thresL, int thresH, bool thinning){
    //p.347-350
    cv::Mat imgGH = img, imgGV= img, imgGCW = img, imgGCCW = img;
    EdgePrewitt(imgGH, 0, 1, false, true);
    EdgePrewitt(imgGV, 0, 2, false, true);
    EdgePrewitt(imgGCW, 0, 3, false, true);
    EdgePrewitt(imgGCCW, 0, 4, false, true);

    vector<cv::Mat*> imgGHVCWCCW = {&imgGH, &imgGV, &imgGCW, &imgGCCW};
    
    int height = img.rows, width = img.cols;
    //the result image storing max gradient from all 4 dirs
    cv::Mat imgMaxG(cv::Size(width, height), CV_8UC1, cv::Scalar(0));
    vector<vector<DIR>> directions(height, vector<DIR>(width));

    //find max gradient direction and set "directions" and "imgMaxG"
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            int g = 0;
            for(int i = 0; i < 4; i++){
                if(imgGHVCWCCW[i]->at<uchar>(row, col) > g){
                    imgMaxG.at<uchar>(row, col) = g = imgGHVCWCCW[i]->at<uchar>(row,col);
                    directions[row][col] = DIR(i);
                }
            }
        }
    }

    double minVal, maxVal;
    cv::minMaxLoc(imgMaxG, &minVal, &maxVal);
    cout << "imgMaxG's range: " << minVal << ", " << maxVal << endl;

    //Binarization
    if(thresL > thresH){
        cout << "thresL should be smaller than thresH" << endl;
        return false;
    }

    if(thresH == 0){
        //note that we will work on "imgMaxG"'s threshold here
        int minDiff = 100; //20; //why this constraint?
        /*
        this will be set through DetectThreshold, 
        meaning the difference of two part's mean grayscale value
        */
        int diff; 
        thresH = DetectThreshold(imgMaxG, diff) * 1.2;
        thresL = thresH * 0.4;
        if(diff > minDiff){
            cout << "found a threshold whose difference in dark part and light part's mean grayscale value is " << diff << " >= " << minDiff << endl;
            return false;
        }
    }else if(thresL == 0){
        thresL = thresH * 0.4;
    }

    cout << "low threshold: " << thresL << ", high threshold: " << thresH << endl;

    // cv::Mat imgThresL = imgMaxG, imgThresH = imgMaxG;
    cv::Mat imgThresL = imgGH, imgThresH = imgGV; //given in textbook
    Threshold(imgThresL, thresL);
    Threshold(imgThresH, thresH);

    cv::Mat target(cv::Size(width, height), CV_8UC1, cv::Scalar(0));
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            //skip if current pixel doesn't form an edge(with high threshold)
            if(imgThresH.at<uchar>(row, col) == 0) continue;
            target.at<uchar>(row, col) = 255;
            switch(directions[row][col]){
                case DIR::H:
                    //left
                    if(imgThresL.at<uchar>(row, col-1)){
                        //why not directly set target?
                        imgThresH.at<uchar>(row, col-1) = 255;
                    }
                    //right
                    if(imgThresL.at<uchar>(row, col+1)){
                        imgThresH.at<uchar>(row, col+1) = 255;
                    }
                    break;
                case DIR::V:
                    //top
                    if(imgThresL.at<uchar>(row-1, col)){
                        imgThresH.at<uchar>(row-1, col) = 255;
                    }
                    //down
                    if(imgThresL.at<uchar>(row+1, col)){
                        imgThresH.at<uchar>(row+1, col) = 255;
                    }
                    break;
                case DIR::CW: //45 degrees
                    //top-right
                    if(imgThresL.at<uchar>(row-1, col+1)){
                        imgThresH.at<uchar>(row-1, col+1) = 255;
                    }
                    //bottom-left
                    if(imgThresL.at<uchar>(row+1, col-1)){
                        imgThresH.at<uchar>(row+1, col-1) = 255;
                    }
                    break;
                case DIR::CCW: //135 degrees
                    //top-left
                    if(imgThresL.at<uchar>(row-1, col-1)){
                        imgThresH.at<uchar>(row-1, col-1) = 255;
                    }
                    //bottom-right
                    if(imgThresL.at<uchar>(row+1, col+1)){
                        imgThresH.at<uchar>(row+1, col+1) = 255;
                    }
                    break;
            }
        }
    }

    if(thinning){
        LinTran(imgThresH, -1, 255);
        Thining(imgThresH);
        LinTran(imgThresH, -1, 255);
    }

    target = imgThresH;
    img = target;

    return true;
};

bool Hough(cv::Mat& img, vector<Line>& lines, int numLines){
    //p.357-360
    //it only accept binary image!
    int height = img.rows, width = img.cols;

    int maxRho = sqrt(height*height + width*width);
    int maxTheta = 90;

    //key: (rho, theta), value: its count
    map<pair<int, int>, int> houghMatrix;

    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            //black is bg, white is fg
            if(img.at<uchar>(row, col) == 0) continue;
            for(int angle = 0; angle < maxTheta; angle++){
                double radiance = (double)angle/180.0*M_PI;
                //(col, row) maps to (x, y) in Cartesian coordinates
                int dist = col * cos(radiance) + row * sin(radiance);
                houghMatrix[make_pair(dist, angle)] += 1;
            }
                
        }
    }

    for(int lineId = 0; lineId < numLines; lineId++){
        std::map<pair<int, int>, int>::iterator p = std::max_element(houghMatrix.begin(), houghMatrix.end(), 
            [] (const std::pair<pair<int, int>, int> & p1, const std::pair<pair<int, int>, int> & p2) {
                return p1.second < p2.second;
            }
        );
        int dist = p->first.first;
        int angle = p->first.second;
        int count = p->second;

        if(count == 0){
            cout << "there are no more points on the same line!" << endl;
            return false;
        }

        Line line(dist, angle, count);
        lines.push_back(line);

        int maxDistAllow = 20;
        int maxAngleAllow = 5;

        for(int distOffset = -maxDistAllow; distOffset <= maxDistAllow; distOffset++){
            for(int angleOffset = -maxAngleAllow; angleOffset <= maxAngleAllow; angleOffset++){
                pair<int, int> searchKey = make_pair(dist+distOffset, angle+angleOffset);
                if(houghMatrix.find(searchKey) != houghMatrix.end()){
                    //it's in valid range
                    houghMatrix[searchKey] = 0;
                }
            }
        }
    }

    return true;
};

int DetectThreshold(cv::Mat& img, int& diff, int maxIter){
    //p.366-367
    /*
    return "diff" as the difference of mean gray value for light part and dark part
    */
    vector<double> hist;
    GenHist(img, hist, 256, false);

    double minGray, maxGray;
    cv::minMaxLoc(img, &minGray, &maxGray);

    if(minGray == maxGray){
        return minGray;
    }

    int thres = (maxGray + minGray)/2, lastThres;
    int pixelCount, graySum;
    //mean gray value for region 1 and region 2
    int meanGray1, meanGray2;

    do{
        lastThres = thres;

        //get pixelCount and graySum for region < thres
        pixelCount = 0;
        graySum = 0;
        for(int i = 0; i < thres; i++){
            pixelCount += hist[i];
            graySum += i * hist[i];
        }
        meanGray1 = graySum/pixelCount;

        //get pixelCount and graySum for region >= thres
        pixelCount = 0;
        graySum = 0;
        for(int i = thres; i < hist.size(); i++){
            pixelCount += hist[i];
            graySum += i * hist[i];
        }
        meanGray2 = graySum/pixelCount;

        thres = (meanGray1 + meanGray2)/2;
        diff = abs(meanGray1 - meanGray2);

    //break when converge or reach maxIter
    }while(maxIter-- > 0 && thres != lastThres);

    return thres;
};

//CH3
//bool Threshold(cv::Mat& img, int nThres);

int AutoThreshold(cv::Mat& img){
    //p.368

    int diff;
    int thres = DetectThreshold(img, diff, 100);
    cout << "Automatically selected threshold: " << thres << endl;
    Threshold(img, thres);
};

bool RegionGrow(cv::Mat& img, int seedRow, int seedCol, int thres){
    //p.371-372
    //it accepts grayscale image
    if(type2str(img.type()) != "8UC1") return false;

    int width = img.cols, height = img.rows;

    if(seedRow < 0 || seedRow >= height){
        cout << "invalid seedRow value!" << endl;
        return false;
    }

    if(seedCol < 0 || seedCol >= width){
        cout << "invalid seedCol value!" << endl;
        return false;
    }

    if(seedRow == 0 && seedCol == 0){
        seedRow = height/2;
        seedCol = width/2;
    }

    //black is bg, white is fg
    cv::Mat target(cv::Size(width, height), CV_8UC1, cv::Scalar(0));
    target.at<uchar>(seedRow, seedCol) = 255;

    //seed is originally in growed region
    int totalCount = 1;
    //count of added pixel in one iteration
    int curCount = 0;
    //pixel with gray value around "centerGray" can be added into the region
    int centerGray = img.at<uchar>(seedRow, seedCol);
    //growed region's gray value's sum
    int sumGray = centerGray;

    do{
        curCount = 0;
        for(int row = 1; row < height-1; row++){
            for(int col = 1; col < width-1; col++){
                //current pixel has been added into growed region
                if(target.at<uchar>(row, col) == 255){
                    //scan its 3*3 neighborhood
                    for(int i = 0; i < 3; i++){
                        for(int j = 0; j < 3; j++){
                            //it's already added into growed region
                            if(target.at<uchar>(row-1+i, col-1+j) == 255)continue;
                            if(abs(img.at<uchar>(row-1+i, col-1+j) - centerGray) <= thres){
                                target.at<uchar>(row-1+i, col-1+j) = 255;
                                curCount++;
                                sumGray += img.at<uchar>(row-1+i, col-1+j);
                            }
                        }
                    }
                }
            }
        }
        totalCount += curCount;
        centerGray = sumGray / totalCount;
    }while(curCount > 0);

    img = target;

    return true;
};

//Matlab: qtdecomp, qtgetblk, qtsetblk
void qtdecomp(cv::Mat& img, cv::Mat& res, int thres, int minDim, int maxDim, int startRow, int startCol, int length){
    //p.374-375
    if(res.rows == 0){
        res = cv::Mat(cv::Size(length, length), CV_8UC1, cv::Scalar(0));
    }
    
    res.at<uchar>(startRow, startCol) = length; //initial value

    double minVal, maxVal;
    cv::Mat areaConcerned = img(cv::Rect(startRow, startCol, length, length));
    cv::minMaxLoc(areaConcerned, &minVal, &maxVal);

    // cout << "cropped mat length: " << areaConcerned.rows << endl;
    // cout << "square length : " << length << ", (" << minVal << ", " << maxVal << ")" << endl;

    if((length > maxDim) || (maxVal - minVal > thres && length > minDim)){
        //divide it into four square
        qtdecomp(img, res, thres, minDim, maxDim, startRow, startCol, length/2);
        qtdecomp(img, res, thres, minDim, maxDim, startRow, startCol+length/2, length/2);
        qtdecomp(img, res, thres, minDim, maxDim, startRow+length/2, startCol, length/2);
        qtdecomp(img, res, thres, minDim, maxDim, startRow+length/2, startCol+length/2, length/2);
    }
    
};

bool qtdecomp(cv::Mat& img, cv::Mat& res, int thres, int minDim, int maxDim){
    //p.374-375
    //assume image is a square and it's length is a power of 2
    if(img.rows != img.cols){
        cout << "qtdecomp only accept square image currently!" << endl;
        return false;
    }
    if(log2(img.rows) != int(log2(img.rows))){
        cout << "its size must be a power of 2!" << endl;
        return false;
    }
    
    int length = img.rows;
    
    qtdecomp(img, res, thres, minDim, maxDim, 0, 0, length);

    return true;
}

void qtgetblk(cv::Mat& img, cv::Mat& res, int length, vector<cv::Mat>& vals, vector<vector<int>>& rcs){
    //p.375-376
    //vals, rc: return value
    int height = res.rows, width = res.cols;

    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            if(res.at<uchar>(row, col) == length){
                cv::Mat val = img(cv::Rect(row, col, row+length, col+length));
                vals.push_back(val);
                rcs.push_back({row, col});
            }
        }
    }
}

bool qtsetblk(cv::Mat& img, cv::Mat& res, int length, vector<cv::Mat>& vals, vector<vector<int>>& rcs){
    //p.377
    /*
    res: result from qtdecomp
    length: the length of square to be replaced
    vals: the corresponding square in img will be replaced with "vals"
    rcs: the position of square to be replaced
    */
    if(vals.size() != rcs.size()){
        cout << "size of vals and rcs must be the same!" << endl;
        return false;
    }

    if(vals[0].rows != length || vals[0].cols != length){
        cout << "the element in vals must be in size " << length << " x " << length << endl;
        return false;
    }

    for(int id = 0; id < vals.size(); id++){
        vector<int> rc = rcs[id];
        cv::Mat val = vals[id];

        int r = rc[0], c = rc[1];
        if(res.at<uchar>(r, c) != length){
            cout << "either length or rcs does not match result from qtdecomp!" << endl;
        }

        for(int i = 0; i < length; i++){
            for(int j = 0; j < length; j++){
                img.at<uchar>(r+i, c+j) = val.at<uchar>(i, j);
            }
        }
    }

    return true;
}

// should be compiled with utility.cpp  CH5.cpp CH3.cpp CH8.cpp

#ifdef CH9
int main(){
    cv::Mat img_lenna = cv::imread("images/Lenna.png", 0);
    cv::Mat img_binary = cv::imread("images/binary.png", 0);
    cv::Mat img_house = cv::imread("images/house.jpg", 0);
    cv::Mat img_cat = cv::imread("images/cat.jpeg", 0);
    cv::Mat img_light = cv::imread("images/light.jfif", 0);
    cv::Mat img_rice = cv::imread("images/rice.png", 0);
    bool isSave = false;

    // cv::Mat edgePrewittImg = img_house.clone();
    // // EdgePrewitt(edgePrewittImg, 0, 3);
    // EdgePrewitt(edgePrewittImg, 0, 3, false, true);
    // vector<cv::Mat> edgePrewittsImgs = {img_house, edgePrewittImg};
    // ShowHorizontal(edgePrewittsImgs, "edge Prewitt", isSave);

    // //p.346
    // cv::Mat edgeSobelImg = img_house.clone();
    // EdgeSobel(edgeSobelImg, 0, 3);
    // // EdgeSobel(edgeSobelImg, 100, 0);
    // vector<cv::Mat> edgeSobelImgs = {img_house, edgeSobelImg};
    // ShowHorizontal(edgeSobelImgs, "edge sobel", isSave);

    // //p.347
    // cv::Mat edgeLoGImg = img_house.clone();
    // EdgeLoG(edgeLoGImg, 0);
    // vector<cv::Mat> edgeLoGImgs = {img_house, edgeLoGImg};
    // ShowHorizontal(edgeLoGImgs, "edge LoG", isSave);

    // //p.350
    // cv::Mat edgeCannyImg = img_house.clone();
    // EdgeCanny(edgeCannyImg);
    // vector<cv::Mat> edgeCannyImgs = {img_house, edgeCannyImg};
    // ShowHorizontal(edgeCannyImgs, "edge Canny", isSave);

    // //p.360
    // cv::Mat houghImg = edgeCannyImg.clone();
    // vector<Line> lines;
    // int numLines = 10;
    // Hough(houghImg, lines, numLines);
    // //draw lines on image
    // int width = edgeCannyImg.cols, height = edgeCannyImg.rows;
    // for(Line& line : lines){
    //     /*
    //     // try use cv::line
    //     int dist = line.dsit, angle = line.angle;
    //     double radiance = (double)angle/180.0*M_PI;
    //     int col = dist * cos(radiance);
    //     int row = dist * sin(radiance);
    //     cv::Point p1 = cv::Point(0,0), p2 = cv::Point(50,50);
    //     cv::line(houghImg, p1, p2, cv::Scalar(255), 1);
    //     */
    //     for(int row = 0; row < height; row++){
    //         for(int col = 0; col < width; col++){
    //             double radiance = (double)line.angle/180.0*M_PI;
    //             int dist = col * cos(radiance) + row * sin(radiance);
    //             if(line.dist == dist){
    //                 //current point is on the line
    //                 houghImg.at<uchar>(row, col) = 255;
    //             }
    //         }
    //     }
    // }
    // vector<cv::Mat> houghImgs = {edgeCannyImg, houghImg};
    // ShowHorizontal(houghImgs, "hough", isSave);

    // //p.368
    // cv::Mat autoThresholdImg = img_rice.clone();
    // AutoThreshold(autoThresholdImg);
    // vector<cv::Mat> autoThresholdImgs = {img_rice, autoThresholdImg};
    // ShowHorizontal(autoThresholdImgs, "auto threshold", isSave);

    // //p.373
    // cv::Mat regionGrowedImg = img_light.clone();
    // RegionGrow(regionGrowedImg, 67, 147);
    // vector<cv::Mat> regionGrowedImgs = {img_light, regionGrowedImg};
    // ShowHorizontal(regionGrowedImgs, "region grow", isSave);

    //p.377-378
    cv::Mat img_cropped = img_rice; //(cv::Rect(0, 0, 32, 32));
    // cout << mat.rows << " x " << mat.cols << endl;
    cv::Mat img_decomposed;
    qtdecomp(img_cropped, img_decomposed, 256*0.2);
    // cout << "Matrix: " << endl;
    // cout << img_cropped << endl;
    // cout << "Decomposed: " << endl;
    // cout << img_decomposed << endl;
    
    //set all non-zero element to 255 to better visualization
    for(int row = 0; row < img_decomposed.rows; row++){
        for(int col = 0; col < img_decomposed.cols; col++){
            img_decomposed.at<uchar>(row, col) = (img_decomposed.at<uchar>(row, col) > 0) ? 255: 0;
        }
    }
    vector<cv::Mat> decomposedImgs = {img_cropped, img_decomposed};
    ShowHorizontal(decomposedImgs, "decompose", isSave);
}
#endif
