//
// Created by xiang on 2019/10/22.
//
/*
 * 测试fast，Harris，shi-Tomasi角点在纯光流下的稳定性
 * 多数据集测试
 *
 */
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace std;

double fx=0,fy=0,cx=0,cy=0;//相机内参
float depth_scale=0;//深度因子
std::string dataset_dir;//数据集位置
int grid_size = 30;//grid的尺寸
int points_num_grid = 5;//每个grid的最大取点数
double F_THRESHOLD = 1.0;//去outlier要用的参数
int MAXCONER = 200;//每帧的最大角点数
int MIN_DIST = 30;//点之间的最小距离
int min_inliers = 30;//两帧之间最小内联点
const int WINDOW_SIZE = 20;//追踪20次为完全追踪到的点
int outlier = 0; //outlier数量

void getParameter(){
    cv::FileStorage fs("../config/default.yaml",cv::FileStorage::READ);
    if (!fs.isOpened()){
        cout <<"No file!" << endl;
        return;
    }
    fx = (double)fs["camera.fx"];
    fy = (double)fs["camera.fy"];
    cx = (double)fs["camera.cx"];
    cy = (double)fs["camera.cy"];
    depth_scale = (float)fs["camera.depth_scale"];
    dataset_dir = (string)fs["dataset_dir"];
    grid_size = (int)fs["grid_size"];
    points_num_grid = (int)fs["points_num_grid"];
    F_THRESHOLD = (double)fs["F_THRESHOLD"];
    MAXCONER = (int)fs["MAXCONER"];
    MIN_DIST = (int)fs["MIN_DIST"];
    min_inliers = (int)fs["min_inliers"];
}

//用于定义点排序
bool compare(cv::KeyPoint a,cv::KeyPoint b){
    return a.response > b.response;
}
//对取的点按照最小距离进行筛选
void minDistance(cv::Mat image, vector<cv::Point2f> &points, int minDistance=30,int maxCorners=1000){
    size_t i, j, total = points.size(), ncorners = 0;
    vector<cv::Point2f> corners;
    if (minDistance >= 1)
    {
        // Partition the image into larger grids
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<cv::Point2f> > grid(grid_width*grid_height);

        minDistance *= minDistance;

        for( int i = 0; i < total; i++ )
        {
            int y = (int)points[i].y;
            int x = (int)points[i].x;

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            for( int yy = y1; yy <= y2; yy++ )
            {
                for( int xx = x1; xx <= x2; xx++ )
                {
                    std::vector <cv::Point2f> &m = grid[yy*grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(cv::Point2f((float)x, (float)y));

                corners.push_back(cv::Point2f((float)x, (float)y));
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for( i = 0; i < total; i++ )
        {
            int y = (int)points[i].y;
            int x = (int)points[i].x;

            corners.push_back(cv::Point2f((float)x, (float)y));
            ++ncorners;
            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }
    points.clear();
    points = corners;
}
//去除追踪失败的点
void reducePoints(vector<cv::Point2f> &v, vector<uchar> status){
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
//去除outlier
void rejectWithF(vector<cv::Point2f> &points1,vector<cv::Point2f> &points2,int & size){
    vector<uchar> status;
    if(points2.size() >= 8){
        cv::findFundamentalMat(points1,points2,cv::FM_RANSAC,F_THRESHOLD,0.99,status);
        for (int i = 0; i < int(status.size()); i++)
            if (status[i] == 0) size++;

        reducePoints(points1,status);
        reducePoints(points2,status);
    }
}



int main(int argc, char** argv) {
    if (argc != 1) {
        cout << "wrong input\nusage: grid " << endl;
        return 1;
    }

    getParameter();

    int image_index=0;//图像索引
    string dataset = dataset_dir + "/associate.txt"; // 数据文件
    ifstream fin(dataset);// 读取数据集文件
    string rgb_time, rgb_file, depth_time, depth_file; // 存放每次读取到的信息
    cv::Mat image_prev, image_this, depth_prev,depth_this; //图片信息
    cv::Mat image_show; // 用于展示的图像
    vector<cv::Point2f> points_prev,points_this; // 暂存相邻帧角点

    cv::namedWindow("VO",cv::WINDOW_AUTOSIZE); // 创建一个显示窗口
    cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,20,0.03);//停止迭代标准

    while (true){
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_file = dataset_dir + "/" + rgb_file;
        depth_file = dataset_dir + "/" + depth_file;
        if (fin.eof())   break;

        image_this = cv::imread(rgb_file);
        image_this.copyTo(image_show);
        //depth_this = cv::imread(depth_file);
        cv::cvtColor(image_this, image_this, cv::COLOR_BGR2GRAY);//转化为灰度图
        //判断是不是第一帧
        if(image_prev.empty()){
            vector<cv::KeyPoint> points_Fast; //暂时存储提取的点
            //cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(20, true);
            cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(100,0.01,30,3, true,0.04);
            detector -> detect(image_this,points_Fast);
            sort(points_Fast.begin(),points_Fast.end(),compare);//点排序
            //暂存
            vector<cv::Point2f> points_temp;
            for (int i = 0; i < points_Fast.size(); ++i) {
                points_temp.push_back(points_Fast[i].pt);
            }
            //进行距离筛选
            //minDistance(image_this, points_temp,30,100);
            for (int i = 0; i < points_temp.size(); ++i) {
                points_this.push_back(points_temp[i]);
            }
            //亚像素角点精确化
            cv::cornerSubPix(image_this,points_this,
                    cv::Size(10,10),cv::Size(-1,-1),termCriteria);
        }
        else{
            vector<uchar > status_Fast;//光流中记载异常值
            vector<uchar > status_depth;//深度中记载异常值
            vector<float > err_Fast;
            bool Lost = false;//当前帧是否丢失

            cv::calcOpticalFlowPyrLK(image_prev,image_this,points_prev,points_this,status_Fast
                    ,err_Fast,cv::Size(21,21),3,termCriteria,0,0.001);

            reducePoints(points_this,status_Fast);
            reducePoints(points_prev,status_Fast);
            rejectWithF(points_prev,points_this,outlier);//去除outlier

        }
        if(points_this.size() == 0 ) {
            cout<<"all point lose\noutlier size: "<<outlier<<endl;
            return 0;
        }
        //绘制点
        for (int i = 0; i < points_this.size(); ++i) {
            cv::circle(image_show, points_this[i], 3, cv::Scalar(0, 255, 0), -1, 8);//画出点
        }
        cv::imshow("VO",image_show);
        image_index++;
        cout<<"\npoints size : "<<points_this.size()<<"\nimage index : "<<image_index<<endl;
        //进行帧迭代
        swap(points_this,points_prev);
        swap(image_this,image_prev) ;
        swap(depth_this,depth_prev);

        cv::waitKey(0);
    }
    return 0;
}

