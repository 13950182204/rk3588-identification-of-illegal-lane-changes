#include <iostream>
#include <opencv2/opencv.hpp>
#include "capture.hpp"
#include "display.hpp"
#include "rknnPool.hpp" 
#include "rkYolov5s.hpp"
#include "rkUnet.hpp"
#include "rkPPseg.hpp"
#include "rkDeeplabV3.hpp"
#include "rkYoloP.hpp"


std::shared_ptr<CaptureInterface> capture = nullptr;  //线程 ->摄像头
std::shared_ptr<Display> display = nullptr; //线程 -> 图像显示及视频录制
std::shared_ptr<rknnPool<rkYoloP,Frame_Package, Results_Package>> RKPool = nullptr; //线程 -> AI推理线程池

BlockingQueue<Frame_Package> imageQueue;

BlockingQueue<Results_Package> Output_imageQueue;

int main (int argc, char *argv[]){
    //初始化图像采集

    capture = std::make_shared<CaptureInterface>(imageQueue);
    int ret = capture->init("4.mp4");//
    if(ret != 0){
            std::cout<<"device video can not open !!"<<std::endl;
            return -1;
        }

    //初始化显示
     display = std::make_shared<Display>( Output_imageQueue , true , false);
     ret = display->init();
     if(ret != 0){
             std::cout<<" can not display !!"<<std::endl;
             return -1;
        }
    // 初始化rknn线程池1
    int threadNum = 12;
    std::string model_name ="/home/cat/Documents/gratuate_project/model/RK3588/yolop_640x384.rknn";
    //"/home/cat/Documents/gratuate_project/model/RK3588/Unet_mutilLane_ZQ.rknn"
    //"/home/cat/Documents/gratuate_project/model/RK3588/deeplab-v3-plus-mobilenet-v2.rknn"
    //"/home/cat/Documents/gratuate_project/model/RK3588/yolov5s-640-640.rknn"
    //"/home/cat/Documents/gratuate_project/model/RK3588/pp_liteseg.rknn"
    RKPool = std::make_shared<rknnPool<rkYoloP, Frame_Package, Results_Package>>(model_name, threadNum , imageQueue , Output_imageQueue);
    ret = RKPool->init();
    if(ret != 0){
             std::cout<<" rknnPool init fail!!"<<std::endl;
             return -1;
    }

    

    capture->start();//启动线程
    display->start();//启动线程
    RKPool ->start();//启动线程

    cv::Mat frame_origin;  // 摄像头原始图像
    cv::Mat frame_imgpro;  // 图像处理输出
    
    while (true)
    {

    }
    
   
}