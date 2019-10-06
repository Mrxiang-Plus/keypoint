//
// Created by xiang on 2019/9/23.
//
///使用特定的图片集，
///测试Harris和Shi-Tomasi提取点数

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
    int points_num_Harris = 0; //特征点总数
    int points_num_Shi = 0; //特征点总数
    ifstream fin(file);

    //进行处理
    while(true){
        string image_path;//当前图片路径
        cv::Mat image ;
        vector<cv::KeyPoint> points_Harris;
        vector<cv::KeyPoint> points_Shi;


        fin >> image_path;
        if (fin.eof())  break;

        image = cv::imread(image_path);//读取图片
        //cv::cvtColor(image,image_gray,cv::COLOR_BGR2GRAY);//转化为灰度图
        //用Fast检测器检测角点
        //cv::namedWindow("Fast",cv::WINDOW_AUTOSIZE);
        //cv::resizeWindow("Fast",640,480);

        //Harris 检测器
        cv::Ptr<cv::GFTTDetector> HarrisDetector = cv::GFTTDetector::create(0,0.01,10,3,true,0.04);
        HarrisDetector -> detect(image,points_Harris);

        //Shi-Tomasi 检测器
        cv::Ptr<cv::GFTTDetector> ShiDetector = cv::GFTTDetector::create(0,0.01,10,3, false,0.04);
        ShiDetector -> detect(image,points_Shi);
        //cv::drawKeypoints(image, points_fast, image, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);
        //cv::imshow("Fast", image);
        points_num_Harris += points_Harris.size();
        points_num_Shi += points_Shi.size();
        cout <<"\nHarris points amount : "<< points_Harris.size()<<"\nShi-Tomasi points amount : "<< points_Shi.size() <<endl;

        //cout << "fast point number: " << points_fast.size() << endl;
        //cv::waitKey(-1);
    }

    cout <<"Harris points amount : "<< points_num_Harris<<"\nHarris points amount : "<<points_num_Shi <<endl;
    return 0;
}
