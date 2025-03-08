#include "MatchToolDlg.h"
#include <iostream>

int main() {
    std::string src_path = "../Test Images/Src1.bmp";
    // Very important to read the image in GRAYSCALE!
    cv::Mat src = cv::imread(src_path, IMREAD_GRAYSCALE);
    if(src.empty())
    {
        std::cout << "Could not read the image: " << src_path << std::endl;
        return 1;
    }
 
    std::string dst_path = "../Test Images/Dst1.bmp";
    cv::Mat dst = cv::imread(dst_path, IMREAD_GRAYSCALE);
    if(dst.empty())
    {
        std::cout << "Could not read the image: " << dst_path << std::endl;
        return 1;
    }


    CMatchToolDlg* matcher = new CMatchToolDlg();

    matcher->m_iMaxPos = 10;
    matcher->m_dToleranceAngle = 180;

    matcher->m_matSrc = src;
    matcher->m_matDst = dst;
    
    matcher->LearnPattern();
    BOOL result = matcher->Match();
    
    printf("RESULT: %d\n", result);
    printf("Rows cols: %d %d\n", src.rows, src.cols);

    cvtColor (src, src, CV_GRAY2BGR);
    for(int i = 0; i < matcher->m_vecSingleTargetData.size(); i++) {
        auto data = matcher->m_vecSingleTargetData.at(i);
        printf("(%.2f %.2f) (%.2f %.2f) (%.2f %.2f) (%.2f %.2f) %f %f\n",
            data.ptLT.x, data.ptLT.y, 
            data.ptRT.x, data.ptRT.y, 
            data.ptRB.x, data.ptRB.y,
            data.ptLB.x, data.ptLB.y,
            data.dMatchedAngle, data.dMatchScore
        );
        
        vector<Point> contour;
        contour.push_back(data.ptLT);
        contour.push_back(data.ptRT);
        contour.push_back(data.ptRB);
        contour.push_back(data.ptLB);

        const Point *pts = (const cv::Point*) cv::Mat(contour).data;
        int npts = Mat(contour).rows;
        cv::polylines(src, &pts, &npts, 1, true, CV_RGB(0,255,0), 5);
    }

    cv::imshow("Output", src);
    cv::waitKey(0);

    return 0;
}


// using namespace cv;
// using namespace std;

// int main(int, char)
// {
//  Mat img(500, 500, CV_8UC3);
//  img.setTo(255);
//  ///////////////////////////////////////////////////

//  ///////////////////////////////////////////////////
//  //polylines example 1 
//  vector< Point> contour;
//  contour.push_back(Point(50, 50));
//  contour.push_back(Point(300, 50));
//  contour.push_back(Point(350, 200));
//  contour.push_back(Point(300, 150));
//  contour.push_back(Point(150, 350));
//  contour.push_back(Point(100, 100));
//  // draw the polygon 
//  const cv::Point* pts = (const cv::Point*) Mat(contour).data;
//  int npts = Mat(contour).rows;
//  polylines(img, &pts, &npts, 1, false, Scalar(0, 255, 0));

//  imshow("show0", img);
//  waitKey(0);

//  return 0;
// }