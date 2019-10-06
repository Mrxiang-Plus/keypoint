//
// Created by xiang on 2019/8/26.
//
///使用规范化的角点检测器接口
///角点检测对比
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "wrong input\nusage : keypoint path_dataset\nretry" << endl;
        return 1;
    }

    string datapath = argv[1];
    string datafile = datapath + "/file.txt";

    ifstream fin(datafile);
    int num =0; //当前图片索引

    while (true) {
        //必要的数据
        string picture_file;  //当前图片地址
        cv::Mat this_image, this_gray;  //当前图片及其灰度图

        if (!fin.eof()) {
            fin >> picture_file;
        } else break;

        this_image = cv::imread(picture_file);
        cv::cvtColor(this_image, this_gray, cv::COLOR_BGR2GRAY);

        // Fast特征检测器
        vector<cv::KeyPoint> points_fast;
        cv::Mat image_fast;
        this_image.copyTo(image_fast);
        cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(15, true,cv::FastFeatureDetector::TYPE_9_16);
        fastDetector -> detect(this_gray,points_fast);
        cv::drawKeypoints(this_image,points_fast,image_fast,cv::Scalar(255,0,0));//画出关键点
        cv::imshow("Fast",image_fast);

        //Harris
        vector<cv::KeyPoint> points_harris;
        cv::Mat image_harris;
        this_image.copyTo(image_harris);
        cv::Ptr<cv::GFTTDetector> HarrisDetector = cv::GFTTDetector::create(500,0.01,1,3, true,0.04);
        HarrisDetector -> detect(this_gray,points_harris);
        cv::drawKeypoints(this_image,points_harris,image_harris,cv::Scalar(0,255,0));
        cv::imshow("harris",image_harris);

        //Shi-Tomasi
        vector<cv::KeyPoint> points_Shi;
        cv::Mat image_shi;
        this_image.copyTo(image_shi);
        cv::Ptr<cv::GFTTDetector> ShiDetector = cv::GFTTDetector::create(500,0.01,1,3, false,0.04);
        ShiDetector -> detect(this_gray,points_Shi);
        cv::drawKeypoints(this_image,points_Shi,image_shi,cv::Scalar(0,0,255));
        cv::imshow("Shi-Tomasi",image_shi);

        //SIFT
        vector<cv::KeyPoint> points_SIFT;
        cv::Mat image_SIFT;
        this_image.copyTo(image_SIFT);
        cv::Ptr<cv::xfeatures2d::SIFT> SIFTDetector = cv::xfeatures2d::SIFT::create();
        SIFTDetector -> detect(this_gray,points_SIFT);
        cv::drawKeypoints(this_image,points_Shi,image_SIFT,cv::Scalar(125,125,0));
        cv::imshow("SIFT",image_SIFT);

        //SURF
        vector<cv::KeyPoint> points_SURF;
        cv::Mat image_SURF;
        this_image.copyTo(image_SURF);
        cv::Ptr<cv::xfeatures2d::SURF> SURFDetector = cv::xfeatures2d::SURF::create();
        SURFDetector -> detect(this_gray,points_SURF);
        cv::drawKeypoints(this_image,points_SURF,image_SURF,cv::Scalar(125,0,125));
        cv::imshow("SURF",image_SURF);

        //star
        vector<cv::KeyPoint> points_Star;
        cv::Mat image_Star;
        this_image.copyTo(image_Star);
        cv::Ptr<cv::xfeatures2d::StarDetector> StarDetector= cv::xfeatures2d::StarDetector::create();
        StarDetector -> detect(this_gray,points_Star);
        cv::drawKeypoints(this_image,points_Star,image_Star,cv::Scalar(0,125,255));
        cv::imshow("Star",image_Star);

        // ORB
        vector<cv::KeyPoint> points_ORB;
        cv::Mat image_ORB;
        this_image.copyTo(image_ORB);
        cv::Ptr<cv::ORB> ORBDetector = cv::ORB::create();
        ORBDetector -> detect(this_gray,points_ORB);
        cv::drawKeypoints(this_image,points_ORB,image_ORB,cv::Scalar(255,125,0));
        cv::imshow("ORB",image_ORB);

        cout<< "num : "<<num<<endl;
        num++;
        cv::waitKey(-1);
    }
    return 1;
}

