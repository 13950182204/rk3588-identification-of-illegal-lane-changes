#include "capture.hpp"



int CaptureInterface::init(std::string videopath) {
    _capture = std::make_shared<cv::VideoCapture>();
    _capture->open(videopath);
        if(!_capture->isOpened()){
                std::cout<< "Create Capture Failed." <<std::endl;
                return 1;
            }
        totalFrames = _capture->get(cv::CAP_PROP_FRAME_COUNT);
        _capture->set(cv::CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
        _capture->set(cv::CAP_PROP_FPS, 30);
        _capture->set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        _capture->set(cv::CAP_PROP_FRAME_HEIGHT, 960);
        _Video = true;
       return 0; 

}

int CaptureInterface::init(){
    _capture = std::make_shared<cv::VideoCapture>();
    _capture->open("/dev/video0", cv::CAP_V4L);  // ("/dev/video0", cv::CAP_V4L);
        if(!_capture->isOpened()){
                std::cout<< "Create Capture Failed." <<std::endl;
                return 1;
            }

        _capture->set(cv::CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
        _capture->set(cv::CAP_PROP_FPS, 30);
        _capture->set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        _capture->set(cv::CAP_PROP_FRAME_HEIGHT, 960);
       return 0; 
}

void CaptureInterface::run(){
    cv::Mat _frame;
    Frame_Package frame_package;

    while (_loop) {
        
        if (_capture->read(_frame)) {
            
            if (BlockingQueue_.Size() >= 20 )
                BlockingQueue_.Take();
            cv::resize(_frame,_frame,cv::Size(1280,960));
            frame_package.frame = _frame;
            BlockingQueue_.Put(frame_package);

        }
        else if(double currentFrame = _capture->get(cv::CAP_PROP_POS_FRAMES) >= totalFrames -1 || _Video == true){
            std::cout << "Video  finished" << std::endl;
            break;
        }
        else{
            std::cout << "no capture frame" << std::endl;
            continue;
        }
        if(_Video == true)
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    
}

//void CaptureInterface::get(cv::Mat &frame) { frame = BlockingQueue_1.Take().clone(); }

int CaptureInterface::start() {
    _loop = true;
    _worker_thread = std::thread(&CaptureInterface::run, this);
    return 0;
}

int CaptureInterface::stop() {
    _loop = false;
    _worker_thread.join();
    BlockingQueue_.ShutDown();
    if (_capture->isOpened()) {
        _capture->release();
    }
    return 0;
}

// 将 cv::Mat 转为 image_buffer_t*
image_buffer_t* CaptureInterface::matToImageBuffer(const cv::Mat& image) {
    image_buffer_t* buffer = new image_buffer_t;
    buffer->width = image.cols;
    buffer->height = image.rows;
    buffer->width_stride = image.step; // 或者使用 image.cols * image.elemSize()
    buffer->height_stride = image.rows;
    // 假设 format 已经被正确设置
    //buffer->format = /* 设置 image_format_t 的值 */;
    
    // 分配内存并复制数据
    buffer->virt_addr = new unsigned char[image.total() * image.elemSize()];
    std::memcpy(buffer->virt_addr, image.data, image.total() * image.elemSize());
    
    buffer->size = image.total() * image.elemSize();
    buffer->fd = -1; // 如果不使用文件描述符，可以设置为其他值
    
    return buffer;
}
