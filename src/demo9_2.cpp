//
// Created by xiang on 2019/9/19.
//

//
// Created by xiang on 2019/9/10.
//
///使用特定的图片集，
///测试提取所花时间与点数多少的关系   版本二
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

    int threshold[11] = {5,10,20,30,40,50,60,70,80,90,100};

    for (int i = 0; i <= 10; ++i) {
        int points_num = 0;//特征点总数
        ifstream fin(file);//读取文件

        double start_Fast = cv::getTickCount();// 检测开始时
        //进行处理
        while(true){
            string image_path;//当前图片路径
            cv::Mat image ;
            vector<cv::KeyPoint> points_fast;//存放角点

            fin >> image_path;
            if (fin.eof())  break;

            image = cv::imread(image_path);//读取图片
            //cv::cvtColor(image,image_gray,cv::COLOR_BGR2GRAY);//转化为灰度图
            //用Fast检测器检测角点
            //cv::namedWindow("Fast",cv::WINDOW_AUTOSIZE);
            //cv::resizeWindow("Fast",640,480);

            cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(threshold[i],true,cv::FastFeatureDetector::TYPE_9_16);
            fastDetector -> detect(image,points_fast);
            //cv::drawKeypoints(image, points_fast, image, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);
            //cv::imshow("Fast", image);
            points_num += points_fast.size();
            //cout << "fast point number: " << points_fast.size() << endl;
            //cv::waitKey(-1);
        }

        double time_Fast = (cv::getTickCount() - start_Fast) / (double)cv::getTickFrequency();//检测所花的时间
        cout <<"\ntimes"<< i <<"\npoints amount :"<< points_num <<"\nusing time " << time_Fast <<"s"<<endl;
    }


    return 0;
}
