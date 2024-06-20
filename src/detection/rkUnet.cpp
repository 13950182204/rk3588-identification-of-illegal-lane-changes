
#include <stdio.h>
#include <mutex>
#include "rknn_api.h"
#include <iostream>
#include "postprocess.hpp"
#include "preprocess.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "coreNum.hpp"
#include "rkUnet.hpp"

#include "common.hpp"

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

rkUnet::rkUnet(const std::string &model_path)
{
    this->model_path = model_path;
}

int rkUnet::init(rknn_context *ctx_in, bool share_weight)
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

rknn_context *rkUnet::get_pctx()
{
    return &ctx;
}

Results_Package rkUnet::infer(Frame_Package frame_package)
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

    rknn_output outputs_tensor[io_num.n_output];
    memset(outputs_tensor, 0, sizeof(outputs_tensor));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs_tensor[i].want_float = 1;
    }
    Results_Package Results_package;
    // 模型推理/Model inference
    auto start = std::chrono::high_resolution_clock::now();
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs_tensor, NULL);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "rknn run cost: " << float(duration.count()/1000.0) << " ms" << std::endl;
    // 后处理/Post-processing
    //float *seg_output = (float *)outputs_tensor[0].buf;
    //float *cls_output = (float *)outputs_tensor[1].buf;

    cv::resize(frame_package.frame, frame_package.frame, cv::Size(640, 480));
    //sigmoid(seg_output , output_attrs[0].n_elems);
    //sigmoid(cls_output , output_attrs[1].n_elems);
    //sigmoid(seg_output , output_attrs[0].n_elems);
    //ret = draw_segment_image(seg_output, cls_output,frame_package.frame);
    Results_package.frame_origin = frame_package.frame;


    ret = rknn_outputs_release(ctx, io_num.n_output, outputs_tensor);
    
    return Results_package;
}

Color rkUnet::getColorById(int id) {
    for (const auto& entry : road_label) {
        if (entry.id == id) {
            return entry.color;
        }
    }
    return Color(0, 0, 0);
}
int rkUnet::draw_segment_image(float* seg_result, float* cls_result , cv::Mat& result_img) {
    int height = result_img.rows;
    int width = result_img.cols;
    int num_class = 9;
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

                if (seg_result[currentIndex] > seg_result[maxClassPos]) {
                    maxClassIndex = c;
                    
                }
            }
            //std::cout << seg_result[maxClassPos] <<" "<<maxClassIndex << std::endl;
            // 根据类别索引获取颜色
            if(seg_result[maxClassPos] > 0.8)
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

rkUnet::~rkUnet()
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
