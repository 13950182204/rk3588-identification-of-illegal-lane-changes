#pragma once

#include "rknn_api.h"
#include "common.hpp"
#include "opencv2/core/core.hpp"

static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

 // 定义一个全局的表
static Entry road_label[] = {
        {0, "NO lane markings ", Color(0, 0, 0)},
        {1, "Single white solid line", Color(0, 255, 0)},
        {2, "Single white dashed line", Color(0, 0, 255)},
        {3, "Single yellow solid line", Color(102, 102, 156)},
        {4, "Single yellow dashed line", Color(190, 153, 153)},
        {5, "Double solid while line", Color(153, 153, 153)},
        {6, "Double solid yelow line", Color(250, 170, 30)},
        {7, "Double yellow dashed line", Color(220, 220, 0)},
        {8, "Double white yellow solid line", Color(107, 142, 35)},
        {9, "Double white dashed line", Color(152, 251, 152)},
        {10, "Double white solid dashed line", Color(70, 130, 180)}
    };

class rkUnet{
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
    rkUnet(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    Results_Package infer(Frame_Package frame_package);
    ~rkUnet();
private:
    Color getColorById(int id);
    int draw_segment_image(float* seg_result, float* cls_result , cv::Mat& result_img);
};

