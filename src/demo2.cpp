//
// Created by xiang on 2019/8/19.
//
///角点检测对比
///fast,Harris,shi-Tomasi
///多线程对比显示
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;


int main(int argc, char** argv){
    if(argc != 2){
        cout<<"wrong input\nusage: flow path_to_dataset"<<endl;
        return 1 ;
    }

    //一些必要的数据
    cv::namedWindow("fast",cv::WINDOW_AUTOSIZE);//创建一个显示窗口
    cv::namedWindow("harris",cv::WINDOW_AUTOSIZE);//创建一个显示窗口
    cv::namedWindow("Shi-Tomasi",cv::WINDOW_AUTOSIZE);//创建一个显示窗口
    cv::Mat this_image,this_gray;//图片信息
    cv::Mat this_image1,this_image2,this_image3;//对应3种角点的图片显示
    vector<cv::Point2f> points1;//存储采集到的角点
    vector<cv::Point2f> points2;//存储采集到的角点
    vector<cv::Point2f> points3;//存储采集到的角点
    cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.03);//停止迭代标准

    string dataset_path = argv[1];
    string dataset = dataset_path + "/file.txt";
    int num = 0;


    ifstream fin(dataset);//读取数据集文件
    ///读取数据并进行处理

    while(true){
        string picture_file;//存储当前图片地址

        if (!fin.eof()){
            fin >> picture_file;
        }
        else break;

        this_image = cv::imread(picture_file);
        cv::cvtColor(this_image, this_gray, cv::COLOR_BGR2GRAY);
        this_image.copyTo(this_image1);
        this_image.copyTo(this_image2);
        this_image.copyTo(this_image3);

        //Fast检测器
        vector<cv::KeyPoint> kps;
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
        detector->detect( this_gray, kps );
        //点导入
        for(auto kp:kps)
            points1.push_back(kp.pt);

        cv::cornerSubPix(this_gray,points1,cv::Size(10,10),cv::Size(-1,-1),termCriteria);
        //画图
        for ( int i=0; i < points1.size(); ++i) {
            cv::circle(this_image1,points1[i],3,cv::Scalar(255,0,0),-1,8);//画出点
        }
        cv::imshow("fast",this_image1);

        //Harris检测器
        goodFeaturesToTrack(this_gray,points2,500,0.01,10,cv::noArray(),3, true,0.04);
        cv::cornerSubPix(this_gray,points2,cv::Size(10,10),cv::Size(-1,-1),termCriteria);
        //画图
        for ( int i=0; i < points2.size(); ++i) {
            cv::circle(this_image2,points2[i],3,cv::Scalar(0,255,0),-1,8);//画出点
        }
        cv::imshow("harris",this_image2);
        //Shi-Tomasi检测器
        goodFeaturesToTrack(this_gray,points3,500,0.01,10,cv::noArray(),3, false,0.04);
        cv::cornerSubPix(this_gray,points3,cv::Size(10,10),cv::Size(-1,-1),termCriteria);
        //画图
        for ( int i=0; i < points3.size(); ++i) {
            cv::circle(this_image3,points3[i],3,cv::Scalar(0,0,255),-1,8);//画出点
        }
        cv::imshow("Shi-Tomasi",this_image3);

        num++;
        cout<<"num :"<<num<<endl;
        points1.clear();
        points2.clear();
        points3.clear();

        cv::waitKey(-1);

    }
    return 0;
}

