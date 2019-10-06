//
// Created by xiang on 2019/8/19.
//

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace cv;
using namespace std;

int main()
{
    string imgPath = "road.jpg";
    Mat img = imread(imgPath, CV_LOAD_IMAGE_COLOR);

    vector<string> detectorNames{"HARRIS","GFTT","SIFT",
                                 "SURF","FAST","STAR","ORB","BRISK"};

    cout << "DetectorName" << '\t'
         << "Number of corners" << '\t'
         << "Time used" <<'\t'<<"efficiency"
         << endl << endl;

    for (string detectorName:detectorNames)
    {
        cout <<detectorName<<+"\t\t";
        double t = (double)getTickCount();

        //--detect keypoints
        Ptr<FeatureDetector> detector= FeatureDetector::create(detectorName);
        vector<KeyPoint> keyPoints;
        detector->detect(img, keyPoints, Mat());
        cout << keyPoints.size() << "\t\t\t";

        //--draw keypoints
        Mat imgKeyPoints;
        drawKeypoints(img, keyPoints, imgKeyPoints,
                      Scalar::all(-1), DrawMatchesFlags::DEFAULT);

        imshow(detectorName+" KeyPoints", imgKeyPoints);

        //time used
        t = ((double)getTickCount() - t) / getTickFrequency();
        cout << t << "\t";

        //Number of coners detected per unit time（ms）
        double efficiency = keyPoints.size() / t / 1000;
        cout  << efficiency << endl<<endl;
    }

    waitKey(0);
    return 0;
}
