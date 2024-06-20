#pragma once

#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include "blocking_queue.h"
#include <mutex>
#include "common.hpp"

class Display {
   public:
    Display(BlockingQueue<Results_Package>& queue ,bool windows,bool video) : Output_imageQueue(queue), _video (video) ,_windows(windows){};
    ~Display(){};
    
   private:
    bool _windows = false;
    bool _video = false;
    bool _loop = false;
    BlockingQueue<Results_Package>&  Output_imageQueue;
    std::thread _worker_thread;
    std::string _win_name = "window1";
    cv::VideoWriter _out_video;

   public:
    
    int init();
    int start();
    int stop();

   private:
    void run();
};
