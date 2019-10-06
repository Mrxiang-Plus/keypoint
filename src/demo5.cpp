//
// Created by xiang on 2019/9/11.
//
///基于Fast,harris,Shi-Tomasi角点的简易光流演示程序
///多窗口演示
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
using namespace std;


int main(int argc, char** argv){
    if(argc != 2){
        cout<<"wrong input\nusage: flow path_to_dataset"<<endl;
        return 1 ;
    }

    //一些必要的数据
    cv::namedWindow("Fast",cv::WINDOW_AUTOSIZE);//创建一个显示窗口
    cv::namedWindow("Harris",cv::WINDOW_AUTOSIZE);//创建一个显示窗口
    cv::namedWindow("Shi-Tomasi",cv::WINDOW_AUTOSIZE);//创建一个显示窗口

    cv::Mat this_image, this_gray, prev_image, prev_gray;//图片信息
    cv::Mat this_image_Fast,this_image_Harris,this_image_Shi;
    vector<cv::Point2f> points1[2],points2[2],points3[2];//存储采集到的角点

    cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.03);//停止迭代标准

    string dataset_path = argv[1];
    string dataset = dataset_path + "/file.txt";
    int num = 0;
    bool isfrist = true;

    ifstream fin(dataset);//读取数据集文件
    ///读取数据并进行处理

    while(true){
        string picture_file;//存储当前图片地址

        if (!fin.eof()){
            fin >> picture_file;
        } else break;

        this_image = cv::imread(picture_file);
        cv::cvtColor(this_image, this_gray, cv::COLOR_BGR2GRAY);
        //对每个提取器复制当前帧
        this_image.copyTo(this_image_Fast);
        this_image.copyTo(this_image_Harris);
        this_image.copyTo(this_image_Shi);

        //第一帧提取角点
        if (isfrist){
            ///Fast检测器
            vector<cv::KeyPoint> points_Fast;
            cv::Ptr<cv::FastFeatureDetector> FastDetector = cv::FastFeatureDetector::create(10, true);
            FastDetector -> detect(this_gray, points_Fast );
            cv::KeyPointsFilter::retainBest(points_Fast,300);//取前300个特征点

            //点导入
            for(auto kp:points_Fast)
                points1[1].push_back(kp.pt);
            cv::cornerSubPix(this_gray,points1[1],cv::Size(10,10),cv::Size(-1,-1),termCriteria);
            //在当前帧上画出角点
            for (int i = 0; i < points1[1].size(); ++i) {
                cv::circle(this_image_Fast,points1[1][i],3,cv::Scalar(0,255,0),-1,8);//画出点
            }

            ///Harris检测器
            vector<cv::KeyPoint> points_Harris;
            cv::Ptr<cv::GFTTDetector> HarrisDetector = cv::GFTTDetector::create(300, 0.01,1,3, true,0.04);
            HarrisDetector -> detect(this_gray, points_Harris );

            //点导入
            for(auto kp:points_Harris)
                points2[1].push_back(kp.pt);
            cv::cornerSubPix(this_gray,points2[1],cv::Size(10,10),cv::Size(-1,-1),termCriteria);
            //在当前帧上画出角点
            for (int i = 0; i < points2[1].size(); ++i) {
                cv::circle(this_image_Harris,points2[1][i],3,cv::Scalar(0,255,0),-1,8);//画出点
            }
            ///Shi-Tomasi检测器
            vector<cv::KeyPoint> points_Shi;
            cv::Ptr<cv::GFTTDetector> ShiDetector = cv::GFTTDetector::create(300, 0.01,1,3, false,0.04);
            ShiDetector -> detect(this_gray, points_Shi );

            //点导入
            for(auto kp:points_Shi)
                points3[1].push_back(kp.pt);
            cv::cornerSubPix(this_gray,points3[1],cv::Size(10,10),cv::Size(-1,-1),termCriteria);
            //在当前帧上画出角点
            for (int i = 0; i < points3[1].size(); ++i) {
                cv::circle(this_image_Shi,points3[1][i],3,cv::Scalar(0,255,0),-1,8);//画出点
            }
            isfrist = false;
        }
            //后续进行光流追踪
        else{
            vector<uchar > status_Fast,status_Harris,status_Shi;
            vector<float > err_Fast,err_Harris,err_Shi;
            cv::calcOpticalFlowPyrLK(prev_gray,this_gray,points1[0],points1[1],status_Fast,err_Fast,cv::Size(10,10),3,termCriteria,0,0.001);
            cv::calcOpticalFlowPyrLK(prev_gray,this_gray,points2[0],points2[1],status_Harris,err_Harris,cv::Size(10,10),3,termCriteria,0,0.001);
            cv::calcOpticalFlowPyrLK(prev_gray,this_gray,points3[0],points3[1],status_Shi,err_Shi,cv::Size(10,10),3,termCriteria,0,0.001);

            int i_Fast,j_Fast,i_Harris,j_Harris,i_Shi,j_Shi;
            for (i_Fast=0, j_Fast=0; i_Fast < points1[1].size(); ++i_Fast) {
                if (!status_Fast[i_Fast]) continue;
                cv::line(this_image_Fast, points1[0][i_Fast], points1[1][i_Fast], cv::Scalar(255, 255, 255), 2, 8);
                points1[1][j_Fast++] = points1[1][i_Fast];//去除追踪失败的点
                cv::circle(this_image_Fast, points1[1][i_Fast], 3, cv::Scalar(0, 255, 0), -1, 8);//画出点
            }
            points1[1].resize(j_Fast);//调整当前帧的特征点数量

            for (i_Harris=0, j_Harris=0; i_Harris < points2[1].size(); ++i_Harris) {
                if (!status_Harris[i_Harris]) continue;
                cv::line(this_image_Harris, points2[0][i_Harris], points2[1][i_Harris], cv::Scalar(255, 255, 255), 2, 8);
                points2[1][j_Harris++] = points2[1][i_Harris];//去除追踪失败的点
                cv::circle(this_image_Harris, points2[1][i_Harris], 3, cv::Scalar(0, 255, 0), -1, 8);//画出点
            }
            points2[1].resize(j_Harris);//调整当前帧的特征点数量

            for (i_Shi=0, j_Shi=0; i_Shi < points3[1].size(); ++i_Shi) {
                if (!status_Shi[i_Shi]) continue;
                cv::line(this_image_Shi, points3[0][i_Shi], points3[1][i_Shi], cv::Scalar(255, 255, 255), 2, 8);
                points3[1][j_Shi++] = points3[1][i_Shi];//去除追踪失败的点
                cv::circle(this_image_Shi, points3[1][i_Shi], 3, cv::Scalar(0, 255, 0), -1, 8);//画出点
            }
            points3[1].resize(j_Shi);//调整当前帧的特征点数量

        }

        if(points1[1].size() == 0) break;
        cout<<"num of points of Fast : "<<points1[1].size()<<endl;
        cout<<"num of points of Harris : "<<points2[1].size()<<endl;
        cout<<"num of points of Shi-Tomasi : "<<points3[1].size()<<endl;

        cv::imshow("Fast",this_image_Fast);
        cv::imshow("Harris",this_image_Harris);
        cv::imshow("Shi-Tomasi",this_image_Shi);

        //进行帧迭代
        swap(points1[1],points1[0]);
        swap(points2[1],points2[0]);
        swap(points3[1],points3[0]);

        swap(this_gray,prev_gray);
        swap(this_image,prev_image)  ;
        num++;
        cout<<"num of image:"<<num<<endl;
        cv::waitKey(-1);

    }
    return 0;
}
