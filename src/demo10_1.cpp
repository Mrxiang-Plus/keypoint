//
// Created by xiang on 2019/9/10.
//
///使用特定的图片集，
///测试提取所花时间与分辨率的关系,使用mask  版本一
///Fast角点

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <fstream>

using namespace std;

int main(int argc, char ** argv){

    if (argc != 2) {
        cout << "wrong input\nusage : keypoint  path_dataset" << endl;
        return 1;
    }

    string file_path = argv[1];
    string file = file_path + "/file.txt";
    cv::Mat image,mask ;
    string image_path;//当前图片路径
    vector<cv::KeyPoint> points_fast;
    //设定mask区域
    cv::Rect r1(300,225,40,30);
    cv::Rect r2(280,210,80,60);
    cv::Rect r3(240,180,160,120);
    cv::Rect r4(160,120,320,240);
    cv::Rect r5(0,0,640,480);
    vector<cv::Rect> rectlist;
    rectlist.push_back(r1);
    rectlist.push_back(r2);
    rectlist.push_back(r3);
    rectlist.push_back(r4);
    rectlist.push_back(r5);


    for (int i = 0; i < 5; ++i) {
        cv::Rect r = rectlist[i];
        int points_num = 0;//特征点总数
        ifstream fin(file);
        //设置mask
        mask = cv::Mat::zeros(480,640,CV_8UC1);
        mask(r).setTo(255);

        double start_Fast = cv::getTickCount();// 检测开始时
        //进行处理
        while(true){

            fin >> image_path;
            if (fin.eof())  break;

            image = cv::imread(image_path);//读取图片
            //cv::cvtColor(image,image_gray,cv::COLOR_BGR2GRAY);//转化为灰度图
            //用Fast检测器检测角点
            //cv::namedWindow("Fast",cv::WINDOW_AUTOSIZE);
            //cv::resizeWindow("Fast",640,480);

            cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(50,true,cv::FastFeatureDetector::TYPE_9_16);
            fastDetector -> detect(image,points_fast,mask);
            //cv::drawKeypoints(image, points_fast, image, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);
            //cv::rectangle(image,r5,cv::Scalar(0,255,0),1,8,0);
            //cv::imshow("Fast", image);
            points_num += points_fast.size();
            //cout << "fast point number: " << points_fast.size() << endl;
            //cv::waitKey(-1);
        }

        double time_Fast = (cv::getTickCount() - start_Fast) / (double)cv::getTickFrequency();//检测所花的时间
        cout <<"musk " << i+1 <<"\n   points amount :"<< points_num <<"\n   using time " << time_Fast <<"s"<<endl;
    }

    return 0;
}




//Fast特征检测器


//    cv::resizeWindow("Fast",1080,720);
//    cout << "image size: "<<image_Fast.size<<endl;
//    double start_Fast = cv::getTickCount();// 检测开始时
//    double time_Fast = (cv::getTickCount() - start_Fast) / (double)cv::getTickFrequency();//检测所花的时间
//    //cv::KeyPointsFilter::removeDuplicated(points_fast);
//    //cv::KeyPointsFilter::retainBest(points_fast,500);//设定角点数量为前500
