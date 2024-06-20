#include "display.hpp"


int Display::init() {
    if(_windows == true){
        cv::namedWindow(_win_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(_win_name,1280, 960);
        cv::moveWindow(_win_name, 600, 500);
        std::cout << "Creat Window" << std::endl;
    } 
    else if(_video == true){
        _out_video.open("sample.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                       30, cv::Size(640, 480), true);
        if (!_out_video.isOpened()){
            std::cout << "no open" << std::endl;
                return -1;
        }
    }
    else{
        std::cout << "display err!!" << std::endl;
        return -1;
    }

    return 0;
}

int Display::start() {

    _loop = true;
    _worker_thread = std::thread(&Display::run, this);

    return 0;
}

int Display::stop() {

    _loop = false;
    _worker_thread.join();
    _out_video.release();

    cv::destroyAllWindows();

    
    return 0;
}


void Display::run() {
    char text[256];
    Results_Package Results_package;
    

    while (_loop) {    
        
        if (Output_imageQueue.Size() >= 1) {
            
            if (_windows){
                Results_package = Output_imageQueue.Take();
                cv::resize(Results_package.frame_origin,Results_package.frame_origin, cv::Size(1280,960));
                cv::imshow(_win_name,Results_package.frame_origin);
                cv::waitKey(1);
            }
            
            if (_video) {
                _out_video << Results_package.frame_origin;
            }
               
        }

       //std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}
