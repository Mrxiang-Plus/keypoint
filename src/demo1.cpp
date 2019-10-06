//
// Created by xiang on 2019/8/16.
//
///角点检测初步对比
///fast ,shi-Tomasi,Harris
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
        return 1;
    }

    //一些必要的数据
    cv::namedWindow("OpticalFlow",cv::WINDOW_AUTOSIZE);//创建一个显示窗口
    string dataset_path = argv[1];
    string dataset = dataset_path + "/file.txt";
    cv::Mat this_image,this_gray;//图片信息
    vector<cv::Point2f> points;//存储采集到的角点
    cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.03);//停止迭代标准
    int num = 0;
    int detector ;

    cout<<"请输入你要使用的角点检测方法\n"<<"Fast : f\n"<<"Harris: h\n"<<"Shi-Tomasi: t"<<endl;
    while (true){
        detector = cv::waitKey(-1);//确定用什么角点检测器
        if (detector!=102 && detector!=104 && detector!=116)
            cout<<"worng input,retry"<<endl;
        else break;
    }

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
        //Fast检测器
        if (detector==102){
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect( this_gray, kps );
            //点导入
            for(auto kp:kps)
                points.push_back(kp.pt);
        }
            //Harris检测器
        else if (detector==104){
            goodFeaturesToTrack(this_gray,points,500,0.01,10,cv::noArray(),3, true,0.04);
        }
            //Shi-Tomasi检测器
        else if (detector==116){
            goodFeaturesToTrack(this_gray,points,500,0.01,10,cv::noArray(),3, false,0.04);
        }

        cv::cornerSubPix(this_gray,points,cv::Size(10,10),cv::Size(-1,-1),termCriteria);


        int i;
        for ( i=0; i < points.size(); ++i) {
            cv::circle(this_image,points[i],3,cv::Scalar(0,255,0),-1,8);//画出点
        }
        cv::imshow("OpticalFlow",this_image);

        num++;
        cout<<"num :"<<num<<endl;
        cv::waitKey(-1);

    }

    return 0;

}

