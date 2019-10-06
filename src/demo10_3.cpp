//
// Created by xiang on 2019/9/23.
//

///使用特定的图片集，
///测试提取所花时间与分辨率的关系，图片裁剪  版本三
///只计算在计算角点上的时间
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
    cv::Mat image;
    string image_path;//当前图片路径
    vector<cv::KeyPoint> points_fast;
    //设定去角点区域
    cv::Rect r0(310,232.5,20,15);//引入目的，第一次检测速度明显减慢。避免这个的影响
    cv::Rect r1(300,225,40,30);
    cv::Rect r2(280,210,80,60);
    cv::Rect r3(240,180,160,120);
    cv::Rect r4(160,120,320,240);
    cv::Rect r5(0,0,640,480);
    vector<cv::Rect> rectlist;
    rectlist.push_back(r0);
    rectlist.push_back(r1);
    rectlist.push_back(r2);
    rectlist.push_back(r3);
    rectlist.push_back(r4);
    rectlist.push_back(r5);


    for (int i = 0; i < 6; ++i) {
        cv::Rect r = rectlist[i];
        int points_num = 0;//特征点总数
        double time_Fast=0; // 采取角点所花费的时间
        ifstream fin(file);

        //进行处理

        while(true){

            fin >> image_path;
            if (fin.eof())  break;

            image = cv::imread(image_path);//读取图片
            //cv::cvtColor(image,image_gray,cv::COLOR_BGR2GRAY);//转化为灰度图
            //用Fast检测器检测角点
            //cv::namedWindow("Fast",cv::WINDOW_AUTOSIZE);
            //cv::resizeWindow("Fast",640,480);
            double start_Fast = cv::getTickCount();// 检测开始时间
            cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(50,true,cv::FastFeatureDetector::TYPE_9_16);
            fastDetector -> detect(image(r),points_fast);
            time_Fast += (cv::getTickCount() - start_Fast) / (double)cv::getTickFrequency();//检测所花的时间
            //cv::drawKeypoints(image, points_fast, image, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);
            //cv::rectangle(image,r,cv::Scalar(0,255,0),1,8,0);
            //cv::imshow("Fast", image);
            points_num += points_fast.size();
            //cout << "fast point number: " << points_fast.size() << endl;
            //cv::waitKey(-1);
        }


        cout <<"musk " << i <<"\n   points amount :"<< points_num <<"\n   using time " << time_Fast <<"s"<<endl;
    }

    return 0;
}
