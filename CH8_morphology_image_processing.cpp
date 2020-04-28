#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> //line
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h> //CV_8UC3
#include <iostream>
#include <map>
#include <cmath> //M_PI
#include "CH8.h"

#define SHRINK

using namespace std;

//p.279
BinaryKernel full3Kernel("full3", vector<FB>(3*3, FB::F));
BinaryKernel full5Kernel("full5", vector<FB>(5*5, FB::F));
BinaryKernel full15Kernel("full15", vector<FB>(15*15, FB::F));
BinaryKernel crossKernel("cross", 
        {FB::X, FB::F, FB::X, 
         FB::F, FB::F, FB::F,
         FB::X, FB::F, FB::X});
BinaryKernel square50Kernel("square50", vector<FB>(53*53, FB::B));
BinaryKernel convex1Kernel("convex1", 
        {FB::F, FB::X, FB::X, 
         FB::F, FB::B, FB::X,
         FB::F, FB::X, FB::X});
BinaryKernel convex2Kernel("convex2", 
        {FB::F, FB::F, FB::F, 
         FB::X, FB::B, FB::X,
         FB::X, FB::X, FB::X});
BinaryKernel convex3Kernel("convex3", 
        {FB::X, FB::X, FB::F, 
         FB::X, FB::B, FB::F,
         FB::X, FB::X, FB::F});
BinaryKernel convex4Kernel("convex4", 
        {FB::X, FB::X, FB::X, 
         FB::X, FB::B, FB::X,
         FB::F, FB::F, FB::F});

void initializeKernels(){
    //p.279
    for(int i = 1; i < 51; i++){
        for(int j = 1; j < 51; j++){
            square50Kernel.arr[i*53+j] = FB::F;
        }
    }
};

void Erode(cv::Mat& img, BinaryKernel& kernel){
    //p.278
    //input img should be binary, containing only 0 or 255!
    int kernelHeight = kernel.kernelHeight;
    int kernelWidth = kernel.kernelWidth;
    int kernelMiddleY = kernel.kernelMiddleY;
    int kernelMiddleX = kernel.kernelMiddleX;
    vector<FB> arr = kernel.arr;

    int height = img.rows, width = img.cols;

    //initialize as all 255(white, meaning background)
    cv::Mat target(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
    
    //y + kernelHeight - kernelMiddleY - 1: the y coordinate cooresponding to the bottom of kernel
    for(int y = kernelMiddleY; (y + kernelHeight - kernelMiddleY - 1) < height; y++){
        for(int x = kernelMiddleX; (x + kernelWidth - kernelMiddleX - 1) < width; x++){
            //y and i moves along same coordinate, x and j moves along another
            //0 is foreground, 255 is background

            //current pixel on the image

            //ignore background
            //the center of kernel could be bg, don't skip at that case
            if(arr[kernelMiddleY*kernelWidth+kernelMiddleX] != FB::B && 
                (int)img.at<uchar>(y, x) == 255) continue;
            bool match = true;
            for(int i = 0; i < kernelHeight; i++){
                for(int j = 0; j < kernelWidth; j++){
                    // cout << ((int)img.at<uchar>(y-kernelMiddleY+i, x-kernelMiddleX+j) == 0) << " " << (int)arr[i*kernelWidth+j] << " | ";
                    switch(arr[i*kernelWidth+j]){
                        case FB::X:
                            //we don't care
                            // continue;
                            break;
                        case FB::F:
                            //it must be foreground
                            if((int)img.at<uchar>(y-kernelMiddleY+i, x-kernelMiddleX+j) != 0){
                                match = false;
                            }
                            break;
                        case FB::B:
                            //it must be background
                            if((int)img.at<uchar>(y-kernelMiddleY+i, x-kernelMiddleX+j) != 255){
                                match = false;
                            }
                            break;
                        default:
                            cout << "invalid value in erode structure element!" << endl;
                            break;
                    }
                    //eary stopping
                    if(!match) break;
                }
                //eary stopping
                if(!match) break;
            }
            // cout << endl;
            if(match){
                //set to foreground
                // cout << "(" << y << ", " << x << ") matches" << endl;
                target.at<uchar>(y, x) = 0;
            }
            // if(img.at<uchar>(y, x) != target.at<uchar>(y, x)){
            //     cout << "(" << y << ", " << x << ") ";
            // }
        }
    }

    img = target;
};

void Dilate(cv::Mat& img, BinaryKernel& kernel){
    //p.283
    //input img should be binary, containing only 0 or 255!
    int kernelHeight = kernel.kernelHeight;
    int kernelWidth = kernel.kernelWidth;
    int kernelMiddleY = kernel.kernelMiddleY;
    int kernelMiddleX = kernel.kernelMiddleX;
    vector<FB> arr = kernel.arr;

    /*
    convert S to its reflection wrt its middle point
    e.g.
    [[0,1,1],
     [1,0,1],
     [0,0,0]]

    is converted to:

    [[0,0,0],
     [1,0,1],
     [1,1,0]]
    */

    for(int i = 0; i <= kernelMiddleY; i++){
        // for(int j = 0; j <= kernelWidth-1-i; j++){ //from textbook, incorrect!
        for(int j = 0; j <= ((i == kernelMiddleY) ? kernelMiddleX : kernelWidth-1); j++){
            // cout << "(" << i << ", " << j << ") with (" << (kernelHeight-1-i) << ", " << (kernelWidth-1-j) << ")" << endl;
            arr[i*kernelWidth+j] = arr[(kernelHeight-1-i)*kernelWidth+(kernelWidth-1-j)];
        }
    }

    int height = img.rows, width = img.cols;

    //initialize as all 255(white, meaning background)
    cv::Mat target(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
    
    for(int y = kernelMiddleY; (y + kernelHeight - kernelMiddleY - 1) < height; y++){
        for(int x = kernelMiddleX; (x + kernelWidth - kernelMiddleX - 1) < width; x++){
            //y and i moves along same coordinate, x and j moves along another
            //0 is foreground, 255 is background

            //current pixel on the image

            //not ignore background here!
            bool match = false;
            for(int i = 0; i < kernelHeight; i++){
                for(int j = 0; j < kernelWidth; j++){
                    switch(arr[i*kernelWidth+j]){
                        case FB::X:
                            //we don't care
                            continue;
                        case FB::F:
                            //if there is any foreground pixel in its neighborhood
                            if((int)img.at<uchar>(y-kernelMiddleY+i, x-kernelMiddleX+j) == 0){
                                //set to foreground
                                target.at<uchar>(y, x) = 0;
                                match = true;
                            }
                            break;
                        default:
                            cout << "invalid value in dilate structure element!" << endl;
                            break;
                    }
                    //eary stopping
                    if(match) break;
                }
                //eary stopping
                if(match) break;
            }
            // if(img.at<uchar>(y, x) != target.at<uchar>(y, x)){
            //     cout << "(" << y << ", " << x << ") ";
            // }
        }
    }

    img = target;
};

void Open(cv::Mat& img, BinaryKernel& kernel){
    //p.286
    //input img should be binary, containing only 0 or 255!
    Erode(img, kernel);
    Dilate(img, kernel);
};

void Close(cv::Mat& img, BinaryKernel& kernel){
    //p.288
    //input img should be binary, containing only 0 or 255!
    Dilate(img, kernel);
    Erode(img, kernel);
};

void ExtractBoundary(cv::Mat& img, BinaryKernel& kernel){
    //p.292
    //input img should be binary, containing only 0 or 255!
    cv::Mat eroded = img.clone();
    Erode(eroded, kernel);
    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            //if a pixel is original black, but eroded to white, then its boundary
            //we set this kind of pixels to black(foreground)
            img.at<uchar>(row, col) = (img.at<uchar>(row, col) == 0 && eroded.at<uchar>(row, col) == 255) ? 0 : 255;
        }
    }
};

void CleanConnRgn(cv::Mat& img, int startRow, int startCol, int nConn){
    //from LabelConnRgn
    int width = img.cols, height = img.rows;

    BinaryKernel kernel = full3Kernel;
    if(nConn == 4){
        kernel = crossKernel;
    }

    //ignore foreground on boundary, set them to background first
    for(int i = 0; i < height; i++){
        img.at<uchar>(i, 0) = img.at<uchar>(i, width-1) = 255;
    }

    for(int j = 0; j < width; j++){
        img.at<uchar>(0, j) = img.at<uchar>(height-1, j) = 255;
    }

    //cc for connected component
    cv::Mat cc(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
    cv::Mat lastCC, diffCC;

    cc.at<uchar>(startRow, startCol) = 0;

    do{
        lastCC = cc;
        Dilate(cc, kernel); //assume "img" is 8-connected
        //revise "revImg" from FillRgn to "img"
        cv::bitwise_or(cc, img, cc);
        cv::bitwise_xor(cc, lastCC, diffCC);
    }while(cv::countNonZero(diffCC) > 0);

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            if((int)cc.at<uchar>(i, j) == 0){
                //set the connected component to bg
                img.at<uchar>(i, j) = 255;
            }
        }
    }
};

void TraceBoundary(cv::Mat& img, vector<vector<vector<int>>>& boundaries, bool traceAll){
    //p.293-295
    //input img should be binary, containing only 0 or 255!
    //support for trace boundaries for multiple objects already done
    int width = img.cols, height = img.rows;

    cv::Mat target(cv::Size(width, height), CV_8UC1, cv::Scalar(255));

    //first is in top-down(y) direction
    vector<vector<int>> dirs = {
        //left-down, down, right-down, right, right-up, up, left-up, left
        {1,-1}, {1,0}, {1,1}, {0,1}, {-1,0}, {-1,0}, {-1,-1}, {0,-1}
    };

    //ignore foreground on boundary, set them to background first
    for(int i = 0; i < height; i++){
        img.at<uchar>(i, 0) = img.at<uchar>(i, width-1) = 255;
    }

    for(int j = 0; j < width; j++){
        img.at<uchar>(0, j) = img.at<uchar>(height-1, j) = 255;
    }

    vector<vector<bool>> visited(height, vector<bool>(width, false));
    
    for(int row = 1; row < height-1; row++){ //need to look at its 3x3 neighborhood
        for(int col = 1; col < width-1; col++){
            //ignore background
            if(img.at<uchar>(row, col) == 255) continue;
            if(visited[row][col]) continue;
            
            int startRow = row, startCol = col;
            int curRow = row, curCol = col;
            int curDir = 0;
            vector<vector<int>> boundary;
            boundary.push_back({curRow, curCol});
            visited[curRow][curCol] = true;
            // cout << "start: (" << curRow << ", " << curCol << ") ";
            // cout << "start: (" << curRow << ", " << curCol << ") " << endl;

            do{
                int nextRow = curRow + dirs[curDir][0];
                int nextCol = curCol + dirs[curDir][1];
                int searchTimes = 1; //have tried one direction

                //try to find a foreground pixel in its neighborhood in specific order
                while(img.at<uchar>(nextRow, nextCol) == 255){
                    curDir = (curDir+1) % 8;
                    nextRow = curRow + dirs[curDir][0];
                    nextCol = curCol + dirs[curDir][1];
                    visited[nextRow][nextCol] = true;
                    if(++searchTimes == 8){
                        //current point's 8 neighbors are all background,
                        //so itself is an isolated point,
                        //don't do anything with isolated point
                        //?
                        nextRow = curRow;
                        nextCol = curCol;
                        break;
                    }
                    // cout << "try (" << nextRow << ", " << nextCol << ") " << endl;
                }

                //we've found the next boundary point
                //(nextRow, nextCol) is either current point(if isolated) or the next point in the boundary
                curRow = nextRow;
                curCol = nextCol;
                // cout << " (" << curRow << ", " << curCol << ") "; // << endl;
                target.at<uchar>(curRow, curCol) = 0;
                //building one boundary
                boundary.push_back({curRow, curCol});
                //turn clockwise 90 degrees
                curDir = (curDir-2+8)%8;
            }while((curRow != startRow) || (curCol != startCol)); //note the stop condition!

            if(traceAll){
                //need to clear its connected component after this finding!
                CleanConnRgn(img, startRow, startCol, 4);
            }else{
                img = target;
                return;
            }
            

            //also need to record the boundary into a vector
            //one boundary for an object
            boundaries.push_back(boundary);

            // cout << endl;
        }
    }

    img = target;
};

void FillRgn(cv::Mat& img, int seedRow, int seedCol, BinaryKernel& kernel){
    //p.298
    //fill region
    //input img should be binary, containing only 0 or 255!
    /*
    kernel is cross if the img is 8-connected, 
    kernel is 3*3 square if the img is 4-connected
    */
    int width = img.cols, height = img.rows;

    cv::Mat target(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
    if(seedRow == -1 || seedCol == -1){
        //center of the image
        seedRow = height/2;
        seedCol = width/2;
    }
    target.at<uchar>(seedRow, seedCol) = 0;
    
    //the complement of img
    cv::Mat revImg = img.clone();
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            revImg.at<uchar>(row, col) = 255 - (int)img.at<uchar>(row, col);
        }
    }

    cv::Mat last, diff;

    do{
        last = target;
        Dilate(target, kernel);
        /*
        0 in our case is foreground, 255 is background
        we want (0, 0) -> 0 and (0, 255) -> 255 and (255, 255) -> 255
        This can be written as !((!a) & (!b)), since our fg and bg are reverted
        By De Morgan's law, it's equal to (a || b)
        */
        cv::bitwise_or(target, revImg, target);
        /*
        check the difference of target and last result,
        continue the loop if they are not equal
        */
        cv::bitwise_xor(target, last, diff);
        // cout << "change " << cv::countNonZero(diff) << " bits" << endl;
    }while(cv::countNonZero(diff) > 0);

    /*
    do union
    !(!a || !b) by De Morgan's law is (a & b)
    (0,0) -> 0 and (0,255) -> 0 and (255,255)->255
    */
    cv::bitwise_and(target, img, img);
};

void LabelConnRgn(cv::Mat& img, int nConn){
    //p.303-304
    //input img should be binary, containing only 0 or 255!
    int height = img.rows, width = img.cols;

    BinaryKernel kernel = full3Kernel;
    if(nConn == 4){
        kernel = crossKernel;
    }

    //?
    //ignore foreground on boundary, set them to background first
    for(int i = 0; i < height; i++){
        img.at<uchar>(i, 0) = img.at<uchar>(i, width-1) = 255;
    }

    for(int j = 0; j < width; j++){
        img.at<uchar>(0, j) = img.at<uchar>(height-1, j) = 255;
    }

    //"target" will contain one connected component in each iteration
    //"last" stores the last result of "target"
    cv::Mat target(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
    cv::Mat last, diff;
    int nConnRgn = 0;

    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            //ignore background
            if(img.at<uchar>(row, col) != 0) continue;
            //reset to all background
            target.setTo(255); //or target = cv::Scalar:all(255);
            //set current point to foreground
            target.at<uchar>(row, col) = 0;

            //from FillRgn
            do{
                last = target;
                Dilate(target, kernel);
                //revise "revImg" from FillRgn to "img"
                cv::bitwise_or(target, img, target);
                cv::bitwise_xor(target, last, diff);
            }while(cv::countNonZero(diff) > 0);

            for(int i = 0; i < height; i++){
                for(int j = 0; j < width; j++){
                    if((int)target.at<uchar>(i, j) == 0){
                        //revise in-place
                        img.at<uchar>(i, j) = nConnRgn;
                    }
                }
            }

            nConnRgn++;
            if(nConnRgn > 255){
                cout << "only support at most 256 connected components!" << endl;
                break;
            }
        }
        if(nConnRgn > 255){
            break;
        }
    }

    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            //ignore background
            if(img.at<uchar>(row, col) == 255) continue;
            img.at<uchar>(row, col) = (int)(img.at<uchar>(row, col) *255.0 / (double)nConnRgn);
        }
    }
};

void Thining(cv::Mat& img){
    //p.307
    /*
    P3 P2 P9
    P4 P1 P8
    P5 P6 P7

    condition 1: 2 <= NZ(P1) <= 6, NZ: foreground pixel count in its 8-neighborhood
    condition 2: Z0(P1) = 1, Z0: when going through its 8-neighborhood counterclockwise, 
                 how many times the pixel changes from bg to fg
    condition 3: P2*P4*P8 = 0 or Z0(P2) not equal to 1
    condition 4: P2*P4*P6 = 0 or Z0(P4) not equal to 1
    */
    //todo: fix the problem that it results in double boundaries
    int height = img.rows, width = img.cols;

    /*
    only look at pixel that are at least 2 pixels from boundary, 
    because we will need to its 5*5 neighborhood later
    */
    bool cond1, cond2, cond3, cond4, modified;
    vector<vector<FB>> nbhd(5, vector<FB>(5, FB::X));
    //this is the coordinate used to index nbhd
    //pad {-1,-1} before for readibility
    vector<vector<int>> P1to9 = {
        {-1, -1}, {2,2}, {1,2}, {1, 1}, {2, 1}, {3, 1}, {3, 2}, {3, 3}, {2, 3}, {1, 3}
    };
    vector<int> P2 = P1to9[2], P4 = P1to9[4], P6 = P1to9[6], P8 = P1to9[8];

    //going counterclockwise
    vector<vector<int>> ccw = {
        {1, 2}, {1, 1}, {2, 1}, {3, 1}, {3, 2}, {3, 3}, {2, 3}, {1, 3}
    };

    vector<vector<int>> ccwP2 = ccw;
    //P2 is one pixel above P1, so as its neighborhood
    for(int i = 0; i < 8; i++){
        ccwP2[i][0] = ccw[i][0]-1;
    }

    vector<vector<int>> ccwP4 = ccw;
    //P4 is one pixel left to P1, so as its neighborhood
    for(int i = 0; i < 8; i++){
        ccwP4[i][1] = ccw[i][1]-1;
    }

    int iters = 0;

    vector<vector<int>> toDeletes;

    do{
        // cout << "iters: " << iters << endl;
        iters++;
        //reset for each iteration
        modified = false;

        toDeletes.clear();

        for(int row = 2; row < height-2; row++){
            for(int col = 2; col < width-2; col++){
                //ignore background
                if(img.at<uchar>(row, col) == 255) continue;

                //reset for each pixel
                cond1 = false; cond2 = false; cond3 = false; cond4 = false;

                //get 5*5 neighborhood's pixel value
                for(int i = -2; i <= 2; i++){
                    for(int j = -2; j <= 2; j++){
                        //foreground or background
                        nbhd[2+i][2+j] = (img.at<uchar>(row+i, col+j) == 0) ? FB::F : FB::B;
                    }
                }

                //condition1: 2 <= NZ(P1) <= 6
                int nz = 0; //NZ(P1)
                for(int i = 1; i <= 3; i++){
                    for(int j = 1; j <= 3; j++){
                        //skip P1 itself
                        if(i == 2 && j == 2) continue;
                        if(nbhd[i][j] == FB::F){
                            nz++;
                        }
                    }
                }
                cond1 = ((nz >= 2) && (nz <= 6));

                //condition2: Z0(P1) == 1
                int z0 = 0;
                for(int p = 0; p < 8; p++){
                    int ci = ccw[p][0], cj = ccw[p][1];
                    int ni = ccw[(p+1)%8][0], nj = ccw[(p+1)%8][1];
                    if(nbhd[ci][cj] == FB::B && nbhd[ni][nj] == FB::F){
                        //transform from bg to fg
                        z0++;
                    }
                }

                cond2 = (z0 == 1);

                //condition3: P2*P4*P8 = 0 or Z0(P2) not equal to 1
                if(nbhd[P2[0]][P2[1]] == FB::B || nbhd[P4[0]][P4[1]] == FB::B || nbhd[P8[0]][P8[1]] == FB::B){
                    cond3 = true;
                }else{
                    int z0P2 = 0;

                    for(int p = 0; p < 8; p++){
                        int ci = ccwP2[p][0], cj = ccwP2[p][1];
                        int ni = ccwP2[(p+1)%8][0], nj = ccwP2[(p+1)%8][1];
                        if(nbhd[ci][cj] == FB::B && nbhd[ni][nj] == FB::F){
                            //transform from bg to fg
                            z0P2++;
                        }
                    }

                    cond3 = (z0P2 != 1);
                }

                //condition4: P2*P4*P6 = 0 or Z0(P4) not equal to 1
                if(nbhd[P2[0]][P2[1]] == FB::B || nbhd[P4[0]][P4[1]] == FB::B || nbhd[P6[0]][P6[1]] == FB::B){
                    cond4 = true;
                }else{
                    int z0P4 = 0;

                    for(int p = 0; p < 8; p++){
                        int ci = ccwP4[p][0], cj = ccwP4[p][1];
                        int ni = ccwP4[(p+1)%8][0], nj = ccwP4[(p+1)%8][1];
                        if(nbhd[ci][cj] == FB::B && nbhd[ni][nj] == FB::F){
                            //transform from bg to fg
                            z0P4++;
                        }
                    }

                    cond4 = (z0P4 != 1);
                }

                if(cond1 && cond2 && cond3 && cond4){
                    //set to bg
                    toDeletes.push_back({row, col});
                    modified = true;
                }else{
                    //fg
                }
                // if(modified) cout << "modified" << endl;
            } //col
        } //row
        for(vector<int>& p : toDeletes){
            img.at<uchar>(p[0], p[1]) = 255;
        }
    }while(modified);
};

void ThiningZhangSuen(cv::Mat& img){
    //https://blog.csdn.net/jia20003/article/details/52142992
    /*
    P3 P2 P9
    P4 P1 P8
    P5 P6 P7

    step 1: scan image all set the pixels meet 4 conditions to bg
    condition 1: 2 <= NZ(P1) <= 6, NZ: foreground pixel count in its 8-neighborhood
    condition 2: Z0(P1) = 1, Z0: when going through its 8-neighborhood counterclockwise, 
                 how many times the pixel changes from bg to fg
    condition 3: P2*P4*P8 = 0
    condition 4: P2*P4*P6 = 0

    step 2: scan image all set the pixels meet 4 conditions to bg
    condition 1: 2 <= NZ(P1) <= 6, NZ: foreground pixel count in its 8-neighborhood
    condition 2: Z0(P1) = 1, Z0: when going through its 8-neighborhood counterclockwise, 
                 how many times the pixel changes from bg to fg
    condition 3: P2*P6*P8 = 0
    condition 4: P4*P6*P8 = 0
    */
    //todo: fix the problem that it results in double boundaries
    int height = img.rows, width = img.cols;

    /*
    only look at pixel that are at least 2 pixels from boundary, 
    because we will need to its 5*5 neighborhood later
    */
    bool cond1, cond2, cond3, cond4, modified;
    vector<vector<FB>> nbhd(5, vector<FB>(5, FB::X));
    //this is the coordinate used to index nbhd
    //pad {-1,-1} before for readibility
    vector<vector<int>> P1to9 = {
        {-1, -1}, {2,2}, {1,2}, {1, 1}, {2, 1}, {3, 1}, {3, 2}, {3, 3}, {2, 3}, {1, 3}
    };
    vector<int> P2 = P1to9[2], P4 = P1to9[4], P6 = P1to9[6], P8 = P1to9[8];

    //going counterclockwise
    vector<vector<int>> ccw = {
        {1, 2}, {1, 1}, {2, 1}, {3, 1}, {3, 2}, {3, 3}, {2, 3}, {1, 3}
    };

    vector<vector<int>> ccwP2 = ccw;
    //P2 is one pixel above P1, so as its neighborhood
    for(int i = 0; i < 8; i++){
        ccwP2[i][0] = ccw[i][0]-1;
    }

    vector<vector<int>> ccwP4 = ccw;
    //P4 is one pixel left to P1, so as its neighborhood
    for(int i = 0; i < 8; i++){
        ccwP4[i][1] = ccw[i][1]-1;
    }

    int iters = 0;

    vector<vector<int>> toDeletes;

    do{
        // cout << "iters: " << iters << endl;
        iters++;
        //reset for each iteration
        modified = false;
        //step 1
        toDeletes.clear();
        for(int row = 2; row < height-2; row++){
            for(int col = 2; col < width-2; col++){
                //ignore background
                if(img.at<uchar>(row, col) == 255) continue;

                //reset for each pixel
                cond1 = false; cond2 = false; cond3 = false; cond4 = false;

                //get 5*5 neighborhood's pixel value
                for(int i = -2; i <= 2; i++){
                    for(int j = -2; j <= 2; j++){
                        //foreground or background
                        nbhd[2+i][2+j] = (img.at<uchar>(row+i, col+j) == 0) ? FB::F : FB::B;
                    }
                }

                //condition1: 2 <= NZ(P1) <= 6
                int nz = 0; //NZ(P1)
                for(int i = 1; i <= 3; i++){
                    for(int j = 1; j <= 3; j++){
                        //skip P1 itself
                        if(i == 2 && j == 2) continue;
                        if(nbhd[i][j] == FB::F){
                            nz++;
                        }
                    }
                }
                cond1 = ((nz >= 2) && (nz <= 6));

                //condition2: Z0(P1) == 1
                int z0 = 0;
                for(int p = 0; p < 8; p++){
                    int ci = ccw[p][0], cj = ccw[p][1];
                    int ni = ccw[(p+1)%8][0], nj = ccw[(p+1)%8][1];
                    if(nbhd[ci][cj] == FB::B && nbhd[ni][nj] == FB::F){
                        //transform from bg to fg
                        z0++;
                    }
                }

                cond2 = (z0 == 1);

                //condition3: P2*P4*P8 = 0
                if(nbhd[P2[0]][P2[1]] == FB::B || nbhd[P4[0]][P4[1]] == FB::B || nbhd[P8[0]][P8[1]] == FB::B){
                    cond3 = true;
                }

                //condition4: P2*P4*P6 = 0
                if(nbhd[P2[0]][P2[1]] == FB::B || nbhd[P4[0]][P4[1]] == FB::B || nbhd[P6[0]][P6[1]] == FB::B){
                    cond4 = true;
                }

                if(cond1 && cond2 && cond3 && cond4){
                    //set to bg
                    // img.at<uchar>(row, col) = 255;
                    toDeletes.push_back({row, col});
                    modified = true;
                }/*else{
                    //fg
                }*/
                // if(modified) cout << "modified" << endl;
            } //col
        } //row

        if(toDeletes.size() == 0) break;
        for(vector<int>& p: toDeletes){
            img.at<uchar>(p[0], p[1]) = 255;
        }
        
        //step 2
        toDeletes.clear();
        for(int row = 2; row < height-2; row++){
            for(int col = 2; col < width-2; col++){
                //ignore background
                if(img.at<uchar>(row, col) == 255) continue;

                //reset for each pixel
                cond1 = false; cond2 = false; cond3 = false; cond4 = false;

                //get 5*5 neighborhood's pixel value
                for(int i = -2; i <= 2; i++){
                    for(int j = -2; j <= 2; j++){
                        //foreground or background
                        nbhd[2+i][2+j] = (img.at<uchar>(row+i, col+j) == 0) ? FB::F : FB::B;
                    }
                }

                //condition1: 2 <= NZ(P1) <= 6
                int nz = 0; //NZ(P1)
                for(int i = 1; i <= 3; i++){
                    for(int j = 1; j <= 3; j++){
                        //skip P1 itself
                        if(i == 2 && j == 2) continue;
                        if(nbhd[i][j] == FB::F){
                            nz++;
                        }
                    }
                }
                cond1 = ((nz >= 2) && (nz <= 6));

                //condition2: Z0(P1) == 1
                int z0 = 0;
                for(int p = 0; p < 8; p++){
                    int ci = ccw[p][0], cj = ccw[p][1];
                    int ni = ccw[(p+1)%8][0], nj = ccw[(p+1)%8][1];
                    if(nbhd[ci][cj] == FB::B && nbhd[ni][nj] == FB::F){
                        //transform from bg to fg
                        z0++;
                    }
                }

                cond2 = (z0 == 1);

                //condition3: P2*P6*P8 = 0
                if(nbhd[P2[0]][P2[1]] == FB::B || nbhd[P6[0]][P6[1]] == FB::B || nbhd[P8[0]][P8[1]] == FB::B){
                    cond3 = true;
                }

                //condition4: P4*P6*P8 = 0
                if(nbhd[P4[0]][P4[1]] == FB::B || nbhd[P6[0]][P6[1]] == FB::B || nbhd[P8[0]][P8[1]] == FB::B){
                    cond4 = true;
                }

                if(cond1 && cond2 && cond3 && cond4){
                    //set to bg
                    // img.at<uchar>(row, col) = 255;
                    toDeletes.push_back({row, col});
                    modified = true;
                }/*else{
                    //fg
                }*/
                // if(modified) cout << "modified" << endl;
            } //col
        } //row

        if(toDeletes.size() == 0) break;
        for(vector<int>& p: toDeletes){
            img.at<uchar>(p[0], p[1]) = 255;
        }
    }while(modified);
};

//8 directions
vector<vector<int>> dirs = {
    {-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}
};

int TestConnRgn(cv::Mat& img, vector<vector<bool>>& visited, vector<vector<int>>& ptVisited, 
    int& curConnRgnSize, int row, int col, int lowerThres, int upperThres){
    //p.313-315
    /*
    curConnRgnSize < lowerThres : -1
    lowerThres <= curConnRgnSize <= upperThres : 0
    curConnRgnSize >= upperThres : 1
    */
    
    curConnRgnSize++;
    visited[row][col] = true;
    ptVisited.push_back({row, col});
    if(curConnRgnSize > upperThres) return 1;

    for(vector<int>& dir : dirs){
        int nrow = row + dir[0], ncol = col + dir[1];
        if(nrow < 0 || nrow >= img.rows || ncol < 0 || ncol >= img.cols) continue;
        if(img.at<uchar>(nrow, ncol) == 0 && !visited[nrow][ncol]){
            int ret = TestConnRgn(img, visited, ptVisited, curConnRgnSize, nrow, ncol, lowerThres, upperThres);
            if(ret == 1) return 1;
            // if(curConnRgnSize > upperThres) return 1;
        }
    }

    // if(curConnRgnSize > upperThres) return 1;
    return (curConnRgnSize >= lowerThres) ? 0 : -1;
};

void PixelImage(cv::Mat& img, int lowerThres, int upperThres){
    //p.311-313
    //input img should be binary, containing only 0 or 255!

    if(upperThres < lowerThres){
        cout << "upperThres should >= lowerThres!" << endl;
        return;
    }
    lowerThres = max(lowerThres, 1);
    //set upper bound as 1000
    upperThres = min(upperThres, 1000);

    int width = img.cols, height = img.rows;

    cv::Mat target(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
    vector<vector<bool>> visited(height, vector<bool>(width, false));
    vector<vector<int>> ptVisited;
    int curConnRgnSize = 0;

    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            if(img.at<uchar>(row, col) == 255) continue;
            // cout << "(" << row << ", " << col << ")" << endl;
            visited = vector<vector<bool>>(height, vector<bool>(width, false));
            ptVisited.clear();
            curConnRgnSize = 0;
            int ret = TestConnRgn(img, visited, ptVisited, curConnRgnSize, row, col, lowerThres, upperThres);
            // cout << "ptVisited.size(): " << ptVisited.size() << ", curConnRgnSize: " << curConnRgnSize << endl;
            if(ret == 0){
                //calculate the centroid of current connected component
                int rowMean = 0, colMean = 0;
                for(vector<int>& pt : ptVisited){
                    rowMean += pt[0];
                    colMean += pt[1];
                }
                rowMean /= ptVisited.size();
                colMean /= ptVisited.size();
                //foreground
                target.at<uchar>(rowMean, colMean) = 0;
                // cout << "(" << rowMean << ", " << colMean << ")" << endl;
            }
            /*
            clear visited connected component to speed up
            */
            vector<int> pt = ptVisited[0];
            CleanConnRgn(img, pt[0], pt[1]);
            // Show(img, "process", false);
        }
    }

    img = target;
};

void Convex(cv::Mat& img, bool constrain){
    //p.317-319
    //input img should be binary, containing only 0 or 255!
    //not converge if we only compare with 1 last image?
    //different result from textbook?
    //todo: fix of the problem not converging
    
    int width = img.cols, height = img.rows;

    cv::Mat last = img, diff;
    cv::Mat last2 = img, diff2;
    vector<cv::Mat> D1to4 = {img.clone(), img.clone(), img.clone(), img.clone()};
    vector<BinaryKernel> convexKernels = {convex1Kernel, convex2Kernel, convex3Kernel, convex4Kernel};

    // cout << width*height - cv::countNonZero(diff) << " fgs in img" << endl;
    for(int i = 0; i < 4; i++){
        int iters = 0;
        do{
            iters++;
            last2 = last;
            last = D1to4[i];
            //hit-or-miss
            Erode(D1to4[i], convexKernels[i]);
            // cout << "hit or miss: " << width*height - cv::countNonZero(D1to4[i]) << " ";
            //union
            // the text in the book says union with original "img"
            // cv::bitwise_and(D1to4[i], img, D1to4[i]);
            // , but the code in the book says union with "last" image
            // , union with "last" image is the correct implementation
            cv::bitwise_and(D1to4[i], last, D1to4[i]);
            // cout << width*height - cv::countNonZero(D1to4[i]) << endl;
            cv::bitwise_xor(D1to4[i], last, diff);
            // cv::bitwise_xor(D1to4[i], last2, diff2);
            // cout << "diff in : " << cv::countNonZero(diff) << " pixels" << endl;
            // cout << "diff2 in : " << cv::countNonZero(diff2) << " pixels" << endl;
            // Show(D1to4[i], "convex process", false);

            //not converge if we only compare with "last"!?
            //so here we need to compare with both "last" and "last2"
        // }while(cv::countNonZero(diff) > 0 && cv::countNonZero(diff2) > 0);
        }while(cv::countNonZero(diff) > 0);
        
        // cout << "do " << iters << " iterations" << endl;
    }

    cv::Mat target(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            //calculate union
            if((int)D1to4[0].at<uchar>(row, col)*(int)D1to4[1].at<uchar>(row, col)*(int)D1to4[2].at<uchar>(row, col)*(int)D1to4[3].at<uchar>(row, col) == 0){
                target.at<uchar>(row, col) = 0;
            }
        }
    }

    if(constrain){
        //constrain the convex to not larger than object's bounding rectangle

        //find boudning box
        int l = width-1, r = 0, t = height-1, b = 0;
        for(int row = 0; row < height; row++){
            for(int col = 0; col < width; col++){
                if(img.at<uchar>(row, col) == 0){
                    l = min(l, col);
                    r = max(r, col);
                    t = min(t, row);
                    b = max(b, row);
                }
            }
        }

        //set those points outside the bounding box to bg
        for(int row = 0; row < height; row++){
            for(int col = 0; col < width; col++){
                if(target.at<uchar>(row, col) == 0){
                    if(row < t || row > b || col < l || col > r){
                        target.at<uchar>(row, col) = 255;
                    }
                }
            }
        }
    }

    img = target;
};

void GrayDilate(cv::Mat& img, BinaryKernel& kernel){
    //p.323-324
    //it only support flat structure element(kernel) for now
    int kernelHeight = kernel.kernelHeight;
    int kernelWidth = kernel.kernelWidth;
    int kernelMiddleY = kernel.kernelMiddleY;
    int kernelMiddleX = kernel.kernelMiddleX;
    vector<FB> arr = kernel.arr;

    int height = img.rows, width = img.cols;

    //initialize as all 255(white, meaning background)
    cv::Mat target(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
    
    int imgY, imgX;
#ifdef SHRINK
    // shrink neighborhood of the boundary
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
#else
    // ignore boundary
    for(int y = kernelMiddleY; (y + kernelHeight - kernelMiddleY - 1) < height; y++){
        for(int x = kernelMiddleX; (x + kernelWidth - kernelMiddleX - 1) < width; x++){
#endif
            //y and i moves along same coordinate, x and j moves along another
            //0 is foreground, 255 is background

            //current pixel on the image

            // //ignore background
            // if(arr[kernelMiddleY*kernelWidth+kernelMiddleX] != FB::B && 
            //     (int)img.at<uchar>(y, x) == 255) continue;
            int rangeMax = INT_MIN;
            for(int i = 0; i < kernelHeight; i++){
                for(int j = 0; j < kernelWidth; j++){
                    switch(arr[i*kernelWidth+j]){
                        case FB::X:
                            break;
                        case FB::F:
                            //it must be foreground
                            //cannot define variable in "case" block!
                            imgY = min(max((int)(y-kernelMiddleY+i), 0), height-1); 
                            imgX = min(max((int)(x-kernelMiddleX+j), 0), width-1); 
                            rangeMax = max(rangeMax, (int)img.at<uchar>(imgY, imgX));
                            break;
                        case FB::B:
                            break;
                        default:
                            cout << "invalid value in erode structure element!" << endl;
                            break;
                    }
                }
            }
            target.at<uchar>(y, x) = rangeMax;
        }
    }

    img = target;
};

void GrayErode(cv::Mat& img, BinaryKernel& kernel){
    //p.327
    //it only support flat structure element(kernel) for now
    int kernelHeight = kernel.kernelHeight;
    int kernelWidth = kernel.kernelWidth;
    int kernelMiddleY = kernel.kernelMiddleY;
    int kernelMiddleX = kernel.kernelMiddleX;
    vector<FB> arr = kernel.arr;

    int height = img.rows, width = img.cols;

    //initialize as all 255(white, meaning background)
    cv::Mat target(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
    
    int imgY, imgX;
#ifdef SHRINK
    // shrink neighborhood of the boundary
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
#else
    // ignore boundary
    for(int y = kernelMiddleY; (y + kernelHeight - kernelMiddleY - 1) < height; y++){
        for(int x = kernelMiddleX; (x + kernelWidth - kernelMiddleX - 1) < width; x++){
#endif
            //y and i moves along same coordinate, x and j moves along another
            //0 is foreground, 255 is background

            //current pixel on the image

            // //ignore background
            // if(arr[kernelMiddleY*kernelWidth+kernelMiddleX] != FB::B && 
            //     (int)img.at<uchar>(y, x) == 255) continue;
            int rangeMin = INT_MAX;
            for(int i = 0; i < kernelHeight; i++){
                for(int j = 0; j < kernelWidth; j++){
                    switch(arr[i*kernelWidth+j]){
                        case FB::X:
                            break;
                        case FB::F:
                            //it must be foreground
                            imgY = min(max((int)(y-kernelMiddleY+i), 0), height-1); 
                            imgX = min(max((int)(x-kernelMiddleX+j), 0), width-1); 
                            rangeMin = min(rangeMin, (int)img.at<uchar>(imgY, imgX));
                            break;
                        case FB::B:
                            break;
                        default:
                            cout << "invalid value in erode structure element!" << endl;
                            break;
                    }
                }
            }
            target.at<uchar>(y, x) = rangeMin;
        }
    }

    img = target;
};

void GrayOpen(cv::Mat& img, BinaryKernel& kernel){
    //p.329-330
    //it only support flat structure element(kernel) for now
    //gray open is used to remove the detail from light part
    GrayErode(img, kernel);
    GrayDilate(img, kernel);
};

void GrayClose(cv::Mat& img, BinaryKernel& kernel){
    //p.330
    //it only support flat structure element(kernel) for now
    //gray close is used to remove the detail from dark part
    GrayDilate(img, kernel);
    GrayErode(img, kernel);
};

void TopHat(cv::Mat& img, BinaryKernel& kernel){
    //p.333-334
    //it only support flat structure element(kernel) for now
    cv::Mat opened = img;
    GrayOpen(opened, kernel);
    //https://docs.opencv.org/master/dd/d4d/tutorial_js_image_arithmetics.html
    cv::subtract(img, opened, img);

    int height = img.rows, width = img.cols;

    //rescale the grayscale value to [0,255]
    int minVal = INT_MAX, maxVal = INT_MIN;
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            int pixel = (int)img.at<uchar>(row, col);
            minVal = min(minVal, pixel);
            maxVal = max(maxVal, pixel);
        }
    }
    
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            int pixel = img.at<uchar>(row, col);
            img.at<uchar>(row, col) = (double)(pixel - minVal)/(double)(maxVal - minVal) * 255;
        }
    }
};

#ifdef CH8
int main(){
    cv::Mat img_lenna = cv::imread("images/Lenna.png", 0);
    cv::Mat img_binary = cv::imread("images/binary.png", 0);
    cv::Mat img_small_binary = cv::imread("images/smallBinary.png", 0);
    cv::Mat img_contour = cv::imread("images/contour.png", 0);
    cv::Mat img_dip = cv::imread("images/thining_VC.bmp", 0);
    cv::Mat img_123 = cv::imread("images/123.png", 0);
    cv::Mat img_678 = cv::imread("images/678.png", 0);
    cv::Mat img_abc = cv::imread("images/abc.png", 0);
    cv::Mat img_mouth = cv::imread("images/bw_mouth_VC.bmp", 0);
    cv::Mat img_rice = cv::imread("images/rice.png", 0);
    cv::Mat img_square = cv::imread("images/binarySquare.png", 0);
    bool isSave = false;

    //p.279
    initializeKernels();

    //p.279
    cv::Mat erodeFullImg = img_binary.clone();
    Erode(erodeFullImg, full3Kernel);
    cv::Mat erodeCrossImg = img_binary.clone();
    Erode(erodeCrossImg, crossKernel);
    vector<cv::Mat> erodeImgs = {img_binary, erodeFullImg, erodeCrossImg};
    ShowHorizontal(erodeImgs, "erode full and erode cross", isSave);

    //p.284
    cv::Mat dilateFullImg = img_binary.clone();
    Dilate(dilateFullImg, full3Kernel);
    cv::Mat dilateCrossImg = img_binary.clone();
    Dilate(dilateCrossImg, crossKernel);
    vector<cv::Mat> dilateImgs = {img_binary, dilateFullImg, dilateCrossImg};
    ShowHorizontal(dilateImgs, "dilate full and dilate cross", isSave);

    //p.286
    cv::Mat openImg = img_binary.clone();
    Open(openImg, full3Kernel);
    vector<cv::Mat> openImgs = {img_binary, erodeFullImg, openImg};
    ShowHorizontal(openImgs, "erode and open", isSave);

    //p.288
    cv::Mat closeImg = img_binary.clone();
    Open(closeImg, full3Kernel);
    vector<cv::Mat> closeImgs = {img_binary, dilateFullImg, closeImg};
    ShowHorizontal(closeImgs, "dilate and close", isSave);

    //p.289-290
    //generate the image with 3 rectangles
    cv::Mat img_rects = cv::Mat(cv::Size(180, 120), CV_8UC1, cv::Scalar(255));
    for(int i = 11; i < 80; i++){
        for(int j = 16; j < 75; j++){
            img_rects.at<uchar>(i, j) = 0;
        }
    }
    for(int i = 56; i < 56+50; i++){
        for(int j = 86; j < 86+50; j++){
            img_rects.at<uchar>(i, j) = 0;
        }
    }
    for(int i = 26; i < 45; i++){
        for(int j = 133; j < 170; j++){
            img_rects.at<uchar>(i, j) = 0;
        }
    }

    cv::Mat hitOrMissImg = img_rects.clone();
    Erode(hitOrMissImg, square50Kernel);
    for(int i = 0; i < hitOrMissImg.rows; i++){
        for(int j = 0; j < hitOrMissImg.cols; j++){
            if((int)hitOrMissImg.at<uchar>(i, j) == 0){
                cv::arrowedLine(hitOrMissImg, cv::Point(max(0, j-10), max(0, i-10)), cv::Point(j, i), cv::Scalar(0));
            }
        }
    }
    vector<cv::Mat> hitOrMissImgs = {img_rects, hitOrMissImg};
    ShowHorizontal(hitOrMissImgs, "hit or miss with 50x50 square", isSave);

    //p.292
    cv::Mat extractBoundaryImg = img_binary.clone();
    ExtractBoundary(extractBoundaryImg, full3Kernel);
    vector<cv::Mat> extractBoundaryImgs = {img_binary, extractBoundaryImg};
    ShowHorizontal(extractBoundaryImgs, "extract boundary", isSave);

    //p.295
    cv::Mat traceBoundaryImg = img_binary.clone();
    vector<vector<vector<int>>> boundaries;
    TraceBoundary(traceBoundaryImg, boundaries);
    // TraceBoundary(traceBoundaryImg, boundaries, true);
    vector<cv::Mat> traceBoundaryImgs = {img_binary, traceBoundaryImg};
    ShowHorizontal(traceBoundaryImgs, "trace boundary", isSave);

    //p.298
    cv::Mat filledImg = img_contour.clone();
    FillRgn(filledImg);
    vector<cv::Mat> filledImgs = {img_contour, filledImg};
    ShowHorizontal(filledImgs, "fill region", isSave);

    //p.305
    cout << "labeling connected component..." << endl;
    cv::Mat labelConnImg = img_binary.clone();
    LabelConnRgn(labelConnImg);
    vector<cv::Mat> labelConnImgs = {img_binary, labelConnImg};
    ShowHorizontal(labelConnImgs, "label connected component", isSave);

    //p.310
    cv::Mat img_org_thin = img_dip;
    // cv::Mat img_org_thin = img_678;
    cv::Mat thinImg = img_org_thin.clone();
    // Thining(thinImg);
    ThiningZhangSuen(thinImg);
    vector<cv::Mat> thinImgs = {img_org_thin, thinImg};
    ShowHorizontal(thinImgs, "thining", isSave);

    //p.316
    cv::Mat pixelatedImg = img_small_binary.clone();
    PixelImage(pixelatedImg, 100);
    vector<cv::Mat> pixelatedImgs = {img_small_binary, pixelatedImg};
    ShowHorizontal(pixelatedImgs, "pixelate", isSave);

    //p.316
    cv::Mat convexUnconstrainedImg = img_mouth.clone();
    cv::Mat convexConstrainedImg = img_mouth.clone();
    Convex(convexUnconstrainedImg, false);
    Convex(convexConstrainedImg, true);
    vector<cv::Mat> convexedImgs = {img_mouth, convexUnconstrainedImg, convexConstrainedImg};
    ShowHorizontal(convexedImgs, "convex unconstrained and constrained", isSave);

    //p.324
    cv::Mat grayDilatedImg = img_lenna.clone();
    GrayDilate(grayDilatedImg, full3Kernel);
    vector<cv::Mat> grayDilatedImgs = {img_lenna, grayDilatedImg};
    ShowHorizontal(grayDilatedImgs, "gray dilated", isSave);

    //p.327
    cv::Mat grayErodedImg = img_lenna.clone();
    GrayErode(grayErodedImg, full3Kernel);
    vector<cv::Mat> grayErodedImgs = {img_lenna, grayErodedImg};
    ShowHorizontal(grayErodedImgs, "gray eroded", isSave);

    //p.330
    cv::Mat grayOpenedImg = img_lenna.clone();
    GrayOpen(grayOpenedImg, full3Kernel);
    vector<cv::Mat> grayOpenedImgs = {img_lenna, grayOpenedImg};
    ShowHorizontal(grayOpenedImgs, "gray opened", isSave);

    //p.331
    cv::Mat grayClosedImg = img_lenna.clone();
    GrayClose(grayClosedImg, full3Kernel);
    vector<cv::Mat> grayClosedImgs = {img_lenna, grayClosedImg};
    ShowHorizontal(grayClosedImgs, "gray closed", isSave);

    //p.334
    cv::Mat grayTopHatedImg = img_rice.clone();
    TopHat(grayTopHatedImg, full15Kernel);
    vector<cv::Mat> grayTopHatedImgs = {img_rice, grayTopHatedImg};
    ShowHorizontal(grayTopHatedImgs, "top hat", isSave);
}
#endif
