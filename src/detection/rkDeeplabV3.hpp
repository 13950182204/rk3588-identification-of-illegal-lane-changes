#pragma once

#include "rknn_api.h"
#include "common.hpp"
#include "opencv2/core/core.hpp"

//mod it accroding to model io input
constexpr size_t OUT_SIZE = 65;  
constexpr size_t MASK_SIZE = 513;  
constexpr size_t NUM_LABEL = 21;  

static int crop_and_scale_image_c(int channel, unsigned char *src, int src_width, int src_height,
                                    int crop_x, int crop_y, int crop_width, int crop_height,
                                    unsigned char *dst, int dst_width, int dst_height,
                                    int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height);

static int crop_and_scale_image_yuv420sp(unsigned char *src, int src_width, int src_height,
                                    int crop_x, int crop_y, int crop_width, int crop_height,
                                    unsigned char *dst, int dst_width, int dst_height,
                                    int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height);

int get_image_size(image_buffer_t* image);
static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);
static int convert_image_rga(image_buffer_t* src_img, image_buffer_t* dst_img, image_rect_t* src_box, image_rect_t* dst_box, char color);
int convert_image(image_buffer_t* src_img, image_buffer_t* dst_img, image_rect_t* src_box, image_rect_t* dst_box, char color);
static int convert_image_cpu(image_buffer_t *src, image_buffer_t *dst, image_rect_t *src_box, image_rect_t *dst_box, char color);
static int get_rga_fmt(image_format_t fmt);


typedef struct {
    float value;
    int index;
} element_t;

static constexpr int FULL_COLOR_MAP[NUM_LABEL][3]= {
  {  0  ,0 ,  0},

  {128, 0 ,0},

  { 0, 128, 0},

  {128, 128, 0},

 {  0 , 0, 128},

 {128, 0, 128},

 {0, 128 , 128},

 {128, 128, 128},

 { 64  ,0  ,0},

 {192,   0  ,0},

 { 64, 128  , 0},

 {192, 128 , 0},

 {64  ,0  , 128},

 {192,  0 , 128},

 { 64 ,128 ,128},

 {192, 128, 128},

 {  0 , 64 , 0},

 {128 , 64  , 0},

 { 0, 192, 0},

 {128, 192  ,0},

{0, 64, 128}

};

//blending two images
//using 0 gamma, 0.5 a
static void compose_img(uint8_t *res_buf, uint8_t *img_buf, int height, int width)
{
  const float alpha = 0.5f;
  float beta = 1.0 - alpha;

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      unsigned char map_label = res_buf[h * width + w];
      // printf("[%d][%d]: %d\n",h,w,pixel);

      auto ori_pixel_r = img_buf[h * width * 3 + w * 3];
      auto ori_pixel_g = img_buf[h * width * 3 + w * 3 + 1];
      auto ori_pixel_b = img_buf[h * width * 3 + w * 3 + 2];
     
      img_buf[h * width * 3 + w * 3] = FULL_COLOR_MAP[map_label][0] * alpha + ori_pixel_r * beta; 
      img_buf[h * width * 3 + w * 3 + 1] = FULL_COLOR_MAP[map_label][1] * alpha + ori_pixel_g * beta; //g 
      img_buf[h * width * 3 + w * 3 + 2] = FULL_COLOR_MAP[map_label][2] * alpha + ori_pixel_b * beta; //b
    
    }
  }
}




inline void swap(element_t* a, element_t* b) {
    element_t temp = *a;
    *a = *b;
    *b = temp;
}

inline int partition(element_t arr[], int low, int high) {
    float pivot = arr[high].value;
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j].value >= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

inline void quick_sort(element_t arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}
 // 定义一个全局的表
static Entry label[] = {
        {0, "road", Color(128, 64, 128)},
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
        {14, "truck", Color(0, 0, 70)},
        {15, "bus", Color(0, 60, 100)},
        {16, "train", Color(0, 80, 100)},
        {17, "motorcycle", Color(0, 0, 230)},
        {18, "bicycle", Color(119, 11, 32)},
        {19, "bicycle", Color(119, 11, 32)},
        {20, "bicycle", Color(119, 11, 32)}
    };


class rkDeeplabV3{
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

    int channel, width, height;
    int img_width, img_height;

public:
    int draw_segment_image(float* result, cv::Mat& result_img);
    Color getColorById(int id);
    rkDeeplabV3(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    Results_Package infer(Frame_Package frame_package);
    ~rkDeeplabV3();
};
 
