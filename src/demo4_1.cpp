//
// Created by xiang on 2019/9/10.
//
///使用特定的图片，对Fast角点，Harris角点，Shi-Tomasi角点进行对比分析
///测试提取所花时间

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace std;

int main(int argv, char ** argc){
    string image_path = "/home/xiang/Documents/dataset/images_point/5.jpg";
    cv::Mat image ,image_gray;

    image = cv::imread(image_path);//读取图片
    cv::cvtColor(image,image_gray,cv::COLOR_BGR2GRAY);//转化为灰度图
    cv::namedWindow("original",cv::WINDOW_NORMAL);
    cv::resizeWindow("original",1080,720);
    cv::imshow("original",image);
    //Fast特征检测器
    cv::Mat image_Fast;
    vector<cv::KeyPoint> points_fast;
    cv::namedWindow("Fast",cv::WINDOW_NORMAL);
    cv::resizeWindow("Fast",1080,720);
    image.copyTo(image_Fast);
    cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(40,true,cv::FastFeatureDetector::TYPE_9_16);
    double start_Fast = cv::getTickCount();// 检测开始时
    fastDetector -> detect(image_gray,points_fast);
    double time_Fast = (cv::getTickCount() - start_Fast) / (double)cv::getTickFrequency();//检测所花的时间
    //cv::KeyPointsFilter::removeDuplicated(points_fast);
    //cv::KeyPointsFilter::retainBest(points_fast,500);//设定角点数量为前500
    cv::drawKeypoints(image_Fast, points_fast, image_Fast, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Fast", image_Fast);
    cout << "fast point number: " << points_fast.size() << "\ntime for fast: " << time_Fast << " s\n" << endl;

    //Harris特征检测器
    cv::Mat image_Harris;
    vector<cv::KeyPoint> points_Harris;
    cv::namedWindow("Harris",cv::WINDOW_NORMAL);
    cv::resizeWindow("Harris",1080,720);
    image.copyTo(image_Harris);
    cv::Ptr<cv::GFTTDetector> HarrisDetector = cv::GFTTDetector::create(0,0.01,5,10, true,0.04);
    double start_Harris = cv::getTickCount();// 检测开始时
    HarrisDetector -> detect(image_gray,points_Harris);
    double time_Harris = (cv::getTickCount() - start_Harris) / (double)cv::getTickFrequency();//检测所花的时间
    cv::drawKeypoints(image_Harris,points_Harris,image_Harris,cv::Scalar(0,255,0),cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Harris",image_Harris);
    cout << "harris point number: " << points_Harris.size() << "\ntime for harris: " << time_Harris << " s\n" << endl;

    //Shi-Tomasi特征检测器
    cv::Mat image_Shi;
    vector<cv::KeyPoint> points_Shi;
    cv::namedWindow("Shi-Tomasi",cv::WINDOW_NORMAL);
    cv::resizeWindow("Shi-Tomasi",1080,720);
    image.copyTo(image_Shi);
    cv::Ptr<cv::GFTTDetector> ShiDetector = cv::GFTTDetector::create(0,0.01,5,10, false,0.04);
    double start_Shi = cv::getTickCount();//检测开始时
    ShiDetector -> detect(image_gray,points_Shi);
    double time_Shi = (cv::getTickCount() - start_Shi) / (double)cv::getTickFrequency();//检测所花的时间
    cv::drawKeypoints(image_Shi,points_Shi,image_Shi,cv::Scalar(0,0,255),cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Shi-Tomasi",image_Shi);
    cout<<"Shi-Tomasi point number: "<<points_Shi.size()<< "\ntime for Shi-Tomasi: " << time_Shi << " s\n" <<endl;

    cv::waitKey(-1);
    return 0;
}