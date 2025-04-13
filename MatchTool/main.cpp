#include "MatchToolDlg.h"
#include <iostream>

int main(int argc, char** argv) {

    if(argc != 3) {
        perror("argc\n");
        exit(EXIT_FAILURE);
    }


    std::string src_path = argv[1];
    // Very important to read the image in GRAYSCALE!
    cv::Mat src = cv::imread(src_path, IMREAD_GRAYSCALE);
    if(src.empty())
    {
        std::cout << "Could not read the image: " << src_path << std::endl;
        return 1;
    }
 
    std::string dst_path = argv[2];
    cv::Mat dst = cv::imread(dst_path, IMREAD_GRAYSCALE);
    if(dst.empty())
    {
        std::cout << "Could not read the image: " << dst_path << std::endl;
        return 1;
    }


    CMatchToolDlg* matcher = new CMatchToolDlg();

    matcher->m_iMaxPos = 300;
    matcher->m_dToleranceAngle = 180;
    matcher->m_matSrc = src;
    matcher->m_matDst = dst;
    matcher->m_ckSIMD = TRUE;
    matcher->m_ckBitwiseNot = FALSE;

    {
        auto t = Timer("match");
        matcher->LearnPattern();
        BOOL result = matcher->Match();

        cvtColor (src, src, CV_GRAY2BGR);
        printf("Matches:\n");
        for(int i = 0; i < matcher->m_vecSingleTargetData.size(); i++) {
            auto data = matcher->m_vecSingleTargetData.at(i);
            printf("(%.2f %.2f) (%.2f %.2f) (%.2f %.2f) (%.2f %.2f) Angle: %f  Score: %f\n",
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

		    string str = format ("%d", i);
            putText(src, str, data.ptCenter, FONT_HERSHEY_COMPLEX, 1,CV_RGB(0,255,0), 2);
        }
    }

//    cv::imshow("Output", src);
//    cv::waitKey(0);

    return 0;
}
