#pragma once

#include "rknn_api.h"
#include "common.hpp"
#include "opencv2/core/core.hpp"
#include "postprocess.hpp"

static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

 // 定义一个全局的表
static Entry lane_road_label[] = {
        {0, "road_area", Color(0, 0, 0)},
        {1, "lane", Color(255, 128, 128)},
    };

class rkYoloP{
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

    float nms_threshold, box_conf_threshold , objThreshold;
public:
    rkYoloP(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    Results_Package infer(Frame_Package frame_package);
    ~rkYoloP();
private:
    const int anchor0[6] = {3,9,5,11,4,20};
    const int anchor1[6] = {7,18,6,39,12,31};
    const int anchor2[6] = {19,50,38,81,68,157};
private:
    Color getColorById(int id);
    int draw_segment_image(     int8_t* seg_result , 
                                cv::Mat& result_img,
                                std::vector<int32_t> &out_zps,
                                std::vector<float> &scale);
    void post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);
    int process(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                   int32_t zp, float scale);
};

