//
// Created by xiang on 2019/9/16.
//

///基于Fast,harris,Shi-Tomasi角点的简易光流演示程序
///多窗口演示,加入可框选区域
///使用视频

#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>

using namespace std;
cv::VideoCapture cap; //相机

cv::Point2f point_down,point_up,point_move;//存储鼠标点击点
bool select_self = false, pause = false, findpoint = false;;
//---------------------------------------------------------------------//
//         鼠标回调函数
//---------------------------------------------------------------------//
static  void onMouse(int event,int x,int y,int /*flag*/,void* /*param*/){
    if( event == CV_EVENT_LBUTTONDOWN){
        point_down = cv::Point2f((float)x,(float)y);
        point_move = point_down; //防止画框图时第一次显示不好看
        select_self = true;
    }

    if(event == CV_EVENT_MOUSEMOVE){
        if(select_self) point_move = cv::Point2f((float)x,(float)y);
    }

    if( event == CV_EVENT_LBUTTONUP){
        point_up = cv::Point2f((float)x,(float)y);
        select_self = false;
    }
}

int main(int argc, char** argv){
    if(argc != 2){
        cout<<"wrong input\nusage: keypoint path_to_video"<<endl;
        return 1 ;
    }
    string video = argv[1];//视频数据位置

    cap.open(video);
    if(!cap.isOpened()){
        cout<< "Could not initialize capturing...\n"<<endl;
        return 0;
    }

    //一些必要的数据
    cv::namedWindow("Fast",cv::WINDOW_NORMAL);//创建一个显示窗口
    cv::namedWindow("Harris",cv::WINDOW_NORMAL);//创建一个显示窗口
    cv::namedWindow("Shi-Tomasi",cv::WINDOW_NORMAL);//创建一个显示窗口
    cv::setMouseCallback("Fast",onMouse,0);//在Fast显示窗上进行框选

    cv::Mat this_image, this_gray, prev_image, prev_gray,mask;//图片信息
    cv::Mat this_image_Fast,this_image_Harris,this_image_Shi;
    vector<cv::Point2f> points1[2],points2[2],points3[2];//存储采集到的角点

    cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.03);//停止迭代标准

    int num = 0;


    while(true){


        cap >> this_image;
        if(this_image.empty()) break;

        cv::cvtColor(this_image, this_gray, cv::COLOR_BGR2GRAY);
        //对每个提取器复制当前帧
        this_image.copyTo(this_image_Fast);
        this_image.copyTo(this_image_Harris);
        this_image.copyTo(this_image_Shi);

        //画出框选区域
        while (select_self){
            cv::Mat image_copy = prev_image.clone();
            cv::waitKey(10);
            cv::Rect r1(point_down,point_move);
            rectangle(image_copy,r1,cv::Scalar(0,255,0),2,8);
            imshow("Fast",image_copy);
            imshow("Harris",image_copy);
            imshow("Shi-Tomasi",image_copy);
            mask = cv::Mat::zeros(this_image.size(),CV_8UC1);
            mask(r1).setTo(255);
            findpoint = true;
        }
        //暂停模式
        while(pause){
            int p = cv::waitKey(10);
            while(select_self){//开启实时框选
                cv::waitKey(10);
                cv::Mat image_copy = prev_image.clone();
                cv::Rect r1(point_down,point_move);
                rectangle(image_copy,r1,cv::Scalar(0,255,0),2,8);
                imshow("Fast",image_copy);
                //imshow("Harris",image_copy);
                //imshow("Shi-Tomasi",image_copy);
                mask = cv::Mat::zeros(this_image.size(),CV_8UC1);
                mask(r1).setTo(255);
                findpoint = true;
            }
            if(p == 112) pause = false;
        }
        //第一帧提取角点
        if (findpoint){
            ///Fast检测器
            vector<cv::KeyPoint> points_Fast;
            cv::Ptr<cv::FastFeatureDetector> FastDetector = cv::FastFeatureDetector::create(10, true);
            FastDetector -> detect(this_gray, points_Fast ,mask);
            cv::KeyPointsFilter::retainBest(points_Fast,100);//取前300个特征点

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
            cv::Ptr<cv::GFTTDetector> HarrisDetector = cv::GFTTDetector::create(100, 0.01,1,3, true,0.04);
            HarrisDetector -> detect(this_gray, points_Harris ,mask);

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
            cv::Ptr<cv::GFTTDetector> ShiDetector = cv::GFTTDetector::create(100, 0.01,1,3, false,0.04);
            ShiDetector -> detect(this_gray, points_Shi ,mask);

            //点导入
            for(auto kp:points_Shi)
                points3[1].push_back(kp.pt);
            cv::cornerSubPix(this_gray,points3[1],cv::Size(10,10),cv::Size(-1,-1),termCriteria);
            //在当前帧上画出角点
            for (int i = 0; i < points3[1].size(); ++i) {
                cv::circle(this_image_Shi,points3[1][i],3,cv::Scalar(0,255,0),-1,8);//画出点
            }
        }
            //后续进行光流追踪
        else{
            if(!points1[0].empty()){
                vector<uchar > status_Fast;
                vector<float > err_Fast;
                cv::calcOpticalFlowPyrLK(prev_gray,this_gray,points1[0],points1[1],status_Fast,err_Fast,cv::Size(10,10),5,termCriteria,0,0.001);

                int i_Fast,j_Fast;
                for (i_Fast=0, j_Fast=0; i_Fast < points1[1].size(); ++i_Fast) {
                    if (!status_Fast[i_Fast]) continue;
                    cv::line(this_image_Fast, points1[0][i_Fast], points1[1][i_Fast], cv::Scalar(255, 255, 255), 2, 8);
                    points1[1][j_Fast++] = points1[1][i_Fast];//去除追踪失败的点
                    cv::circle(this_image_Fast, points1[1][i_Fast], 3, cv::Scalar(0, 255, 0), -1, 8);//画出点
                }
                points1[1].resize(j_Fast);//调整当前帧的特征点数量

            }

            if(!points2[0].empty()){
                vector<uchar > status_Harris;
                vector<float >err_Harris;
                cv::calcOpticalFlowPyrLK(prev_gray,this_gray,points2[0],points2[1],status_Harris,err_Harris,cv::Size(10,10),3,termCriteria,0,0.001);

                int i_Harris,j_Harris;

                for (i_Harris=0, j_Harris=0; i_Harris < points2[1].size(); ++i_Harris) {
                    if (!status_Harris[i_Harris]) continue;
                    cv::line(this_image_Harris, points2[0][i_Harris], points2[1][i_Harris], cv::Scalar(255, 255, 255), 2, 8);
                    points2[1][j_Harris++] = points2[1][i_Harris];//去除追踪失败的点
                    cv::circle(this_image_Harris, points2[1][i_Harris], 3, cv::Scalar(0, 255, 0), -1, 8);//画出点
                }
                points2[1].resize(j_Harris);//调整当前帧的特征点数量

            }

            if(!points3[0].empty()){
                vector<uchar > status_Shi;
                vector<float > err_Shi;
                cv::calcOpticalFlowPyrLK(prev_gray,this_gray,points3[0],points3[1],status_Shi,err_Shi,cv::Size(10,10),3,termCriteria,0,0.001);

                int i_Shi,j_Shi;
                for (i_Shi=0, j_Shi=0; i_Shi < points3[1].size(); ++i_Shi) {
                    if (!status_Shi[i_Shi]) continue;
                    cv::line(this_image_Shi, points3[0][i_Shi], points3[1][i_Shi], cv::Scalar(255, 255, 255), 2, 8);
                    points3[1][j_Shi++] = points3[1][i_Shi];//去除追踪失败的点
                    cv::circle(this_image_Shi, points3[1][i_Shi], 3, cv::Scalar(0, 255, 0), -1, 8);//画出点
                }
                points3[1].resize(j_Shi);//调整当前帧的特征点数量
            }
        }
        findpoint = false;

        //if(points1[1].size() == 0) break;
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

        char c = (char)cv::waitKey(50);
        if(c == 27)   break;//ESC退出

        switch(c){
            case 'f'://开启寻找角点
                findpoint = true;
                break;

            case 'c'://清除当前角点信息
                points1[0].clear();
                points1[1].clear();
                points2[0].clear();
                points2[1].clear();
                points3[0].clear();
                points3[1].clear();
                break;

            case 'p'://暂停/启动
                pause = true;
                break;

        }

    }
    return 0;
}
