#pragma once

#include "rknn_api.h"
#include "common.hpp"
#include "opencv2/core/core.hpp"


static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

 // 定义一个全局的表
static Entry cityscapes_label[] = {
        {0, "road", Color(0, 255, 0)},
        {1, "sidewalk", Color(244, 35, 232)},
        {2, "building", Color(70, 70, 70)},
        {3, "wall", Color(102, 102, 156)},
        {4, "fence", Color(190, 153, 153)},
        {5, "pole", Color(153, 153, 153)},
        {6, "traffic light", Color(250, 170, 30)},
        {7, "traffic sign", Color(220, 220, 0)},
        {8, "vegetation", Color(107, 142, 35)},
        {9, "terrain", Color(152, 251, 152)},
        {10, "sky", Color(70, 130, 180)},
        {11, "person", Color(220, 20, 60)},
        {12, "rider", Color(255, 0, 0)},
        {13, "car", Color(0, 0, 142)},
        {14, "truck", Color(0, 0, 255)},
        {15, "bus", Color(0, 60, 100)},
        {16, "train", Color(0, 80, 100)},
        {17, "motorcycle", Color(0, 0, 230)},
        {18, "bicycle", Color(119, 11, 32)}
    };

class rkPPseg{
private:
    
    int ret;
    std::mutex mtx;
    std::string model_path;
    unsigned char *model_data;

    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];
    //rknn_output *outputs_tensor;
    
    int channel, width, height;
    int img_width, img_height;


public:
    rkPPseg(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    Results_Package infer(Frame_Package frame_package);
    ~rkPPseg();
private:
    Color getColorById(int id);
    int draw_segment_image(float* result, cv::Mat& result_img);
};

