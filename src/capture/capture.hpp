#pragma once
#include <termios.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include "blocking_queue.h"
#include "common.hpp"

class CaptureInterface {
   public:
    CaptureInterface(BlockingQueue<Frame_Package>& queue) : BlockingQueue_(queue){}
    ~CaptureInterface(){};

   private:
    std::shared_ptr<cv::VideoCapture> _capture;
    std::thread _worker_thread;
    BlockingQueue<Frame_Package>&  BlockingQueue_;


   public:
    double totalFrames = 0;
    bool _loop = false;
    bool _launch = false;
    bool _Video = false;
    bool _cap = true;
   public:
    int init(std::string videopath);
    int init();
    int start();
    int stop();

    //void get(cv::Mat &frame);

   private:
    void run();
    image_buffer_t* matToImageBuffer(const cv::Mat& image);
    
    
};