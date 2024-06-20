#pragma once

#include <stdint.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)



typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);
int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);
float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);
int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold);
int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices);
int clamp(float val, int min, int max);
float sigmoid(float x);
void deinitPostProcess();


