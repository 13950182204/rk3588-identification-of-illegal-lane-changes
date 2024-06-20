
#include <stdio.h>
#include <mutex>
#include "rknn_api.h"
#include <iostream>
#include "preprocess.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "coreNum.hpp"
#include "rkYoloP.hpp"

#include "common.hpp"
using namespace std;
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }

    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
           "type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
           attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

rkYoloP::rkYoloP(const std::string &model_path)
{
    this->model_path = model_path;
    nms_threshold = 0.55;      // 默认的NMS阈值
    box_conf_threshold = 0.45; // 默认的置信度阈值
}

int rkYoloP::init(rknn_context *ctx_in, bool share_weight)
{
    //先创建一个模型，用share_weight来判断，后面线程池创建的模型用rknn_dup_context来共享权重
    
    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(model_path.c_str(), &model_data_size);
    // 模型参数复用/Model parameter reuse
    if (share_weight == true)
        ret = rknn_dup_context(ctx_in, &ctx);//目的是为了共享权重
    else                                     //是ctx赋值给ctx_in
                                            // 文档上有点问题
        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);//
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 设置模型绑定的核心/Set the core of the model that needs to be bound
    rknn_core_mask core_mask;
    switch (get_core_num())
    {
    case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
    case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
    case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    }
    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // 获取模型输入输出参数/Obtain the input and output parameters of the model
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // 设置输入参数/Set the input parameters
    input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // 设置输出参数/Set the output parameters
    output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    return 0;
}

rknn_context *rkYoloP::get_pctx()
{
    return &ctx;
}

Results_Package rkYoloP::infer(Frame_Package frame_package)
{
    std::lock_guard<std::mutex> lock(mtx);
    cv::Mat img;

    cv::cvtColor(frame_package.frame, img, cv::COLOR_BGR2RGB);
    img_width = img.cols;
    img_height = img.rows;

    cv::Size target_size(width, height);
    cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
    // 计算缩放比例/Calculate the scaling ratio
    float scale_w = (float)target_size.width / img.cols;
    float scale_h = (float)target_size.height / img.rows;

    // 图像缩放/Image scaling
    if (img_width != width || img_height != height)
    {
        // rga
        rga_buffer_t src;
        rga_buffer_t dst;
        memset(&src, 0, sizeof(src));
        memset(&dst, 0, sizeof(dst));
        ret = resize_rga(src, dst, img, resized_img, target_size);
        if (ret != 0)
        {
            fprintf(stderr, "resize with rga error\n");
        }
        /*********
        // opencv
        float min_scale = std::min(scale_w, scale_h);
        scale_w = min_scale;
        scale_h = min_scale;
        letterbox(img, resized_img, pads, min_scale, target_size);
        *********/
        inputs[0].buf = resized_img.data;
    }
    else
    {
        inputs[0].buf = img.data;
    }

    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));

    outputs[0].want_float = 0;
    outputs[1].want_float = 0;
    outputs[2].want_float = 0;
    outputs[3].want_float = 1;
    outputs[4].want_float = 0;

    Results_Package Results_package;
    // 模型推理/Model inference
    auto start = std::chrono::high_resolution_clock::now();
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "rknn run cost: " << float(duration.count()/1000.0) << " ms" << std::endl;
    // 后处理/Post-processing
    //float *det_output = (float *)outputs[0].buf;
    //float *cls_output = (float *)outputs[1].buf;
    int8_t* seg_output = (int8_t *)outputs[4].buf;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    cv::resize(frame_package.frame, frame_package.frame, cv::Size(width,height));

    
    ret = draw_segment_image(seg_output,frame_package.frame , out_zps , out_scales);
    //ret = draw_segment_image(cls_output,frame_package.frame);

    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));
    detect_result_group_t detect_result_group;
    DetectionResults detection_results;
    this->post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
                 box_conf_threshold, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    // 绘制框体/Draw the box
    char text[256];
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop);
        //打印预测物体的信息/Prints information about the predicted object
        // printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
        //        det_result->box.right, det_result->box.bottom, det_result->prop);

        detection_results.score = det_result->prop;
        detection_results.x1 = det_result->box.left;
        detection_results.y1 = det_result->box.top;
        detection_results.x2 = det_result->box.right;
        detection_results.y2 = det_result->box.bottom;
        Results_package.frame_origin = frame_package.frame;
        Results_package.Results.push_back(detection_results);
        rectangle(Results_package.frame_origin, cv::Point(Results_package.Results[i].x1, Results_package.Results[i].y1), cv::Point(Results_package.Results[i].x2, Results_package.Results[i].y2), cv::Scalar(0, 120, 0, 128), 1);
        putText(Results_package.frame_origin, text, cv::Point(Results_package.Results[i].x1, Results_package.Results[i].y1 -10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }
        


    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    
    return Results_package;
}

Color rkYoloP::getColorById(int id) {
    for (const auto& entry : lane_road_label) {
        if (entry.id == id) {
            return entry.color;
        }
    }
    return Color(0, 0, 0);
}
int rkYoloP::draw_segment_image(int8_t* seg_result , 
                                cv::Mat& result_img ,
                                std::vector<int32_t> &zp, 
                                std::vector<float> &scale) {
    int height = result_img.rows;
    int width = result_img.cols;
    int num_class = 2;
    Color foundColor;
    int maxClassPos;
    int currentIndex;
    // [1, class, height, width] -> [1, 3, height, width]
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int maxClassIndex = 0;

            // 找到概率最高的类别
            for (int c = 1; c < num_class; c++) {
                currentIndex = c * (height * width) + y * width + x;
                maxClassPos = maxClassIndex * (height * width) + y * width + x;
                if (deqnt_affine_to_f32(seg_result[currentIndex] , zp[4] , scale[4]) > deqnt_affine_to_f32(seg_result[maxClassPos] , zp[4] , scale[4])) {
                    maxClassIndex = c;

                }
            }
            //std::cout << seg_result[maxClassIndex] <<" "<<maxClassIndex << std::endl;
            // 根据类别索引获取颜色
                foundColor = getColorById(maxClassIndex);
                
                
            
            float alpha = 0.5;
            // 在结果图像中设置 RGB 值
           // 在结果图像中设置 RGB 值，同时考虑原始图像的颜色和透明度
            auto& pixel = result_img.at<cv::Vec3b>(y, x);
            cv::Vec3b blendedColor;
            blendedColor[0] = static_cast<uchar>(alpha * std::get<0>(foundColor));
            blendedColor[1] = static_cast<uchar>(alpha * std::get<1>(foundColor));
            blendedColor[2] = static_cast<uchar>(alpha * std::get<2>(foundColor));

            // 在结果图像中设置带有透明度的 RGB 值
            result_img.at<cv::Vec3b>(y, x) = blendedColor + pixel;
        }
    }
  return 0;
}

void rkYoloP::post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, BOX_RECT pads, float scale_w, float scale_h, std::vector<int32_t> &qnt_zps,
                 std::vector<float> &qnt_scales, detect_result_group_t *group){
    /////generate proposals

  memset(group, 0, sizeof(detect_result_group_t));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;

  // stride 8
  int stride0 = 8;
  int grid_h0 = model_in_h / stride0;
  int grid_w0 = model_in_w / stride0;
  int validCount0 = 0;
  validCount0 = this->process(input0, (int *)this->anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

  // stride 16
  int stride1 = 16;
  int grid_h1 = model_in_h / stride1;
  int grid_w1 = model_in_w / stride1;
  int validCount1 = 0;
  validCount1 = this->process(input1, (int *)this->anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

  // stride 32
  int stride2 = 32;
  int grid_h2 = model_in_h / stride2;
  int grid_w2 = model_in_w / stride2;
  int validCount2 = 0;
  validCount2 = this->process(input2, (int *)this->anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

  int validCount = validCount0 + validCount1 + validCount2;
  // no object detect
  if (validCount <= 0)
  {
    std::cout << "validCount=0 " << std::endl;
  }
  //std::cout << validCount << std::endl;
 
  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i)
  {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set)
  {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  group->count = 0;
  /* box valid detect target */
  for (int i = 0; i < validCount; ++i)
  {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
    {
      continue;
    }
    int n = indexArray[i];

    float x1 = filterBoxes[n * 4 + 0] - pads.left;
    float y1 = filterBoxes[n * 4 + 1] - pads.top;
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];

    group->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w));
    group->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h));
    group->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w));
    group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) );
    group->results[last_count].prop = obj_conf;
    //char *label = labels[id];
    //strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
    // group->results[last_count].box.top,
    //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
    last_count++;
  }
  group->count = last_count;

}

int rkYoloP::process(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                   int32_t zp, float scale)
{
  int validCount = 0;
  int grid_len = grid_h * grid_w;
  int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
  for (int i = 0; i < grid_h; i++)
  {
    for (int j = 0; j < grid_w; j++)
    {
      for (int a = 0; a < 3; a++)
      {
        int8_t box_confidence = input[(6 * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= thres_i8)
        {
          int offset = (6 * a) * grid_len + i * grid_w + j;
          int8_t *in_ptr = input + offset;
          float box_x = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
          float box_y = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
          float box_w = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
          float box_h = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
          box_x = (box_x + j) * (float)stride;
          box_y = (box_y + i) * (float)stride;
          box_w = box_w * box_w * (float)anchor[a * 2];
          box_h = box_h * box_h * (float)anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          int8_t maxClassProbs = in_ptr[5 * grid_len];
          int maxClassId = 0;
          if (maxClassProbs > thres_i8)
          {
            objProbs.push_back(sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale)) * sigmoid(deqnt_affine_to_f32(box_confidence, zp, scale)));
            classId.push_back(maxClassId);
            validCount++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
          }
        }
      }
    }
  }
  return validCount;
}

rkYoloP::~rkYoloP()
{
    deinitPostProcess();

    ret = rknn_destroy(ctx);

    if (model_data)
        free(model_data);

    if (input_attrs)
        free(input_attrs);
    if (output_attrs)
        free(output_attrs);
}
