
#include <stdio.h>
#include <mutex>
#include "rknn_api.h"

#include "postprocess.hpp"
#include "preprocess.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "coreNum.hpp"
#include "rkDeeplabV3.hpp"

#include "common.hpp"
#include "gpu_compose_impl.h"
#include "cl_kernels/kernel_upsampleSoftmax.h"

using namespace gpu_postprocess;

namespace {
    const constexpr char UPSAMPLE_SOFTMAX_KERNEL_NAME[] = "UpsampleSoftmax";
    const constexpr char UP_SOFTMAX_IN0[] =  "UP_SOFTMAX_IN";
    const constexpr char UP_SOFTMAX_OUT0[] =  "UP_SOFTMAX_OUT";

    std::shared_ptr<gpu_compose_impl> Gpu_Impl = nullptr;
    
}  

static int crop_and_scale_image_yuv420sp(unsigned char *src, int src_width, int src_height,
                                    int crop_x, int crop_y, int crop_width, int crop_height,
                                    unsigned char *dst, int dst_width, int dst_height,
                                    int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height) {

    unsigned char* src_y = src;
    unsigned char* src_uv = src + src_width * src_height;

    unsigned char* dst_y = dst;
    unsigned char* dst_uv = dst + dst_width * dst_height;

    crop_and_scale_image_c(1, src_y, src_width, src_height, crop_x, crop_y, crop_width, crop_height,
        dst_y, dst_width, dst_height, dst_box_x, dst_box_y, dst_box_width, dst_box_height);
    
    crop_and_scale_image_c(2, src_uv, src_width / 2, src_height / 2, crop_x / 2, crop_y / 2, crop_width / 2, crop_height / 2,
        dst_uv, dst_width / 2, dst_height / 2, dst_box_x, dst_box_y, dst_box_width, dst_box_height);

    return 0;
}

static int crop_and_scale_image_c(int channel, unsigned char *src, int src_width, int src_height,
                                    int crop_x, int crop_y, int crop_width, int crop_height,
                                    unsigned char *dst, int dst_width, int dst_height,
                                    int dst_box_x, int dst_box_y, int dst_box_width, int dst_box_height) {
    if (dst == NULL) {
        printf("dst buffer is null\n");
        return -1;
    }

    float x_ratio = (float)crop_width / (float)dst_box_width;
    float y_ratio = (float)crop_height / (float)dst_box_height;

    // printf("src_width=%d src_height=%d crop_x=%d crop_y=%d crop_width=%d crop_height=%d\n",
    //     src_width, src_height, crop_x, crop_y, crop_width, crop_height);
    // printf("dst_width=%d dst_height=%d dst_box_x=%d dst_box_y=%d dst_box_width=%d dst_box_height=%d\n",
    //     dst_width, dst_height, dst_box_x, dst_box_y, dst_box_width, dst_box_height);
    // printf("channel=%d x_ratio=%f y_ratio=%f\n", channel, x_ratio, y_ratio);

    // 从原图指定区域取数据，双线性缩放到目标指定区域
    for (int dst_y = dst_box_y; dst_y < dst_box_y + dst_box_height; dst_y++) {
        for (int dst_x = dst_box_x; dst_x < dst_box_x + dst_box_width; dst_x++) {
            int dst_x_offset = dst_x - dst_box_x;
            int dst_y_offset = dst_y - dst_box_y;

            int src_x = (int)(dst_x_offset * x_ratio) + crop_x;
            int src_y = (int)(dst_y_offset * y_ratio) + crop_y;

            float x_diff = (dst_x_offset * x_ratio) - (src_x - crop_x);
            float y_diff = (dst_y_offset * y_ratio) - (src_y - crop_y);

            int index1 = src_y * src_width * channel + src_x * channel;
            int index2 = index1 + src_width * channel;    // down
            if (src_y == src_height - 1) {
                // 如果到图像最下边缘，变成选择上面的像素
                index2 = index1 - src_width * channel;
            }
            int index3 = index1 + 1 * channel;            // right
            int index4 = index2 + 1 * channel;            // down right
            if (src_x == src_width - 1) {
                // 如果到图像最右边缘，变成选择左边的像素
                index3 = index1 - 1 * channel;
                index4 = index2 - 1 * channel;
            }

            // printf("dst_x=%d dst_y=%d dst_x_offset=%d dst_y_offset=%d src_x=%d src_y=%d x_diff=%f y_diff=%f src index=%d %d %d %d\n",
            //     dst_x, dst_y, dst_x_offset, dst_y_offset,
            //     src_x, src_y, x_diff, y_diff,
            //     index1, index2, index3, index4);

            for (int c = 0; c < channel; c++) {
                unsigned char A = src[index1+c];
                unsigned char B = src[index3+c];
                unsigned char C = src[index2+c];
                unsigned char D = src[index4+c];

                unsigned char pixel = (unsigned char)(
                    A * (1 - x_diff) * (1 - y_diff) +
                    B * x_diff * (1 - y_diff) +
                    C * y_diff * (1 - x_diff) +
                    D * x_diff * y_diff
                );

                dst[(dst_y * dst_width  + dst_x) * channel + c] = pixel;
            }
        }
    }

    return 0;
}

static int convert_image_cpu(image_buffer_t *src, image_buffer_t *dst, image_rect_t *src_box, image_rect_t *dst_box, char color) {
    int ret;
    if (dst->virt_addr == NULL) {
        return -1;
    }
    if (src->virt_addr == NULL) {
        return -1;
    }
    if (src->format != dst->format) {
        return -1;
    }

    int src_box_x = 0;
    int src_box_y = 0;
    int src_box_w = src->width;
    int src_box_h = src->height;
    if (src_box != NULL) {
        src_box_x = src_box->left;
        src_box_y = src_box->top;
        src_box_w = src_box->right - src_box->left + 1;
        src_box_h = src_box->bottom - src_box->top + 1;
    }
    int dst_box_x = 0;
    int dst_box_y = 0;
    int dst_box_w = dst->width;
    int dst_box_h = dst->height;
    if (dst_box != NULL) {
        dst_box_x = dst_box->left;
        dst_box_y = dst_box->top;
        dst_box_w = dst_box->right - dst_box->left + 1;
        dst_box_h = dst_box->bottom - dst_box->top + 1;
    }

    // fill pad color
    if (dst_box_w != dst->width || dst_box_h != dst->height) {
        int dst_size = get_image_size(dst);
        memset(dst->virt_addr, color, dst_size);
    }

    int need_release_dst_buffer = 0;
    int reti = 0;
    if (src->format == IMAGE_FORMAT_RGB888) {
        reti = crop_and_scale_image_c(3, src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else if (src->format == IMAGE_FORMAT_RGBA8888) {
        reti = crop_and_scale_image_c(4, src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else if (src->format == IMAGE_FORMAT_GRAY8) {
        reti = crop_and_scale_image_c(1, src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else if (src->format == IMAGE_FORMAT_YUV420SP_NV12 || src->format == IMAGE_FORMAT_YUV420SP_NV12) {
        reti = crop_and_scale_image_yuv420sp(src->virt_addr, src->width, src->height,
            src_box_x, src_box_y, src_box_w, src_box_h,
            dst->virt_addr, dst->width, dst->height,
            dst_box_x, dst_box_y, dst_box_w, dst_box_h);
    } else {
        printf("no support format %d\n", src->format);
    }
    if (reti != 0) {
        printf("convert_image_cpu fail %d\n", reti);
        return -1;
    }
    printf("finish\n");
    return 0;
}

static int get_rga_fmt(image_format_t fmt) {
    switch (fmt)
    {
    case IMAGE_FORMAT_RGB888:
        return RK_FORMAT_RGB_888;
    case IMAGE_FORMAT_RGBA8888:
        return RK_FORMAT_RGBA_8888;
    case IMAGE_FORMAT_YUV420SP_NV12:
        return RK_FORMAT_YCbCr_420_SP;
    case IMAGE_FORMAT_YUV420SP_NV21:
        return RK_FORMAT_YCrCb_420_SP;
    default:
        return -1;
    }
}

int convert_image(image_buffer_t* src_img, image_buffer_t* dst_img, image_rect_t* src_box, image_rect_t* dst_box, char color)
{
    int ret;
 
    printf("src width=%d height=%d fmt=0x%x virAddr=0x%p fd=%d\n",
        src_img->width, src_img->height, src_img->format, src_img->virt_addr, src_img->fd);
    printf("dst width=%d height=%d fmt=0x%x virAddr=0x%p fd=%d\n",
        dst_img->width, dst_img->height, dst_img->format, dst_img->virt_addr, dst_img->fd);
    if (src_box != NULL) {
        printf("src_box=(%d %d %d %d)\n", src_box->left, src_box->top, src_box->right, src_box->bottom);
    }
    if (dst_box != NULL) {
        printf("dst_box=(%d %d %d %d)\n", dst_box->left, dst_box->top, dst_box->right, dst_box->bottom);
    }
    printf("color=0x%x\n", color);

    ret = convert_image_rga(src_img, dst_img, src_box, dst_box, color);
    if (ret != 0) {
        printf("try convert image use cpu\n");
        ret = convert_image_cpu(src_img, dst_img, src_box, dst_box, color);
    }
    return ret;
}

static int convert_image_rga(image_buffer_t* src_img, image_buffer_t* dst_img, image_rect_t* src_box, image_rect_t* dst_box, char color)
{
    int ret = 0;

    int srcWidth = src_img->width;
    int srcHeight = src_img->height;
    void *src = src_img->virt_addr;
    int src_fd = src_img->fd;
    void *src_phy = NULL;
    int srcFmt = get_rga_fmt(src_img->format);

    int dstWidth = dst_img->width;
    int dstHeight = dst_img->height;
    void *dst = dst_img->virt_addr;
    int dst_fd = dst_img->fd;
    void *dst_phy = NULL;
    int dstFmt = get_rga_fmt(dst_img->format);

    int rotate = 0;

    int use_handle = 0;
#if defined(LIBRGA_IM2D_HANDLE)
    use_handle = 1;
#endif

    // printf("src width=%d height=%d fmt=0x%x virAddr=0x%p fd=%d\n",
    //     srcWidth, srcHeight, srcFmt, src, src_fd);
    // printf("dst width=%d height=%d fmt=0x%x virAddr=0x%p fd=%d\n",
    //     dstWidth, dstHeight, dstFmt, dst, dst_fd);
    // printf("rotate=%d\n", rotate);

    int usage = 0;
    IM_STATUS ret_rga = IM_STATUS_NOERROR;

    // set rga usage
    usage |= rotate;

    // set rga rect
    im_rect srect;
    im_rect drect;
    im_rect prect;
    memset(&prect, 0, sizeof(im_rect));

    if (src_box != NULL) {
        srect.x = src_box->left;
        srect.y = src_box->top;
        srect.width = src_box->right - src_box->left + 1;
        srect.height = src_box->bottom - src_box->top + 1;
    } else {
        srect.x = 0;
        srect.y = 0;
        srect.width = srcWidth;
        srect.height = srcHeight;
    }

    if (dst_box != NULL) {
        drect.x = dst_box->left;
        drect.y = dst_box->top;
        drect.width = dst_box->right - dst_box->left + 1;
        drect.height = dst_box->bottom - dst_box->top + 1;
    } else {
        drect.x = 0;
        drect.y = 0;
        drect.width = dstWidth;
        drect.height = dstHeight;
    }

    // set rga buffer
    rga_buffer_t rga_buf_src;
    rga_buffer_t rga_buf_dst;
    rga_buffer_t pat;
    rga_buffer_handle_t rga_handle_src = 0;
    rga_buffer_handle_t rga_handle_dst = 0;
    memset(&pat, 0, sizeof(rga_buffer_t));

    im_handle_param_t in_param;
    in_param.width = srcWidth;
    in_param.height = srcHeight;
    in_param.format = srcFmt;

    im_handle_param_t dst_param;
    dst_param.width = dstWidth;
    dst_param.height = dstHeight;
    dst_param.format = dstFmt;

    if (use_handle) {
        if (src_phy != NULL) {
            rga_handle_src = importbuffer_physicaladdr((uint64_t)src_phy, &in_param);
        } else if (src_fd > 0) {
            rga_handle_src = importbuffer_fd(src_fd, &in_param);
        } else {
            rga_handle_src = importbuffer_virtualaddr(src, &in_param);
        }
        if (rga_handle_src <= 0) {
            printf("src handle error %d\n", rga_handle_src);
            ret = -1;
            goto err;
        }
        rga_buf_src = wrapbuffer_handle(rga_handle_src, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
    } else {
        if (src_phy != NULL) {
            rga_buf_src = wrapbuffer_physicaladdr(src_phy, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
        } else if (src_fd > 0) {
            rga_buf_src = wrapbuffer_fd(src_fd, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
        } else {
            rga_buf_src = wrapbuffer_virtualaddr(src, srcWidth, srcHeight, srcFmt, srcWidth, srcHeight);
        }
    }

    if (use_handle) {
        if (dst_phy != NULL) {
            rga_handle_dst = importbuffer_physicaladdr((uint64_t)dst_phy, &dst_param);
        } else if (dst_fd > 0) {
            rga_handle_dst = importbuffer_fd(dst_fd, &dst_param);
        } else {
            rga_handle_dst = importbuffer_virtualaddr(dst, &dst_param);
        }
        if (rga_handle_dst <= 0) {
            printf("dst handle error %d\n", rga_handle_dst);
            ret = -1;
            goto err;
        }
        rga_buf_dst = wrapbuffer_handle(rga_handle_dst, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
    } else {
        if (dst_phy != NULL) {
            rga_buf_dst = wrapbuffer_physicaladdr(dst_phy, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
        } else if (dst_fd > 0) {
            rga_buf_dst = wrapbuffer_fd(dst_fd, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
        } else {
            rga_buf_dst = wrapbuffer_virtualaddr(dst, dstWidth, dstHeight, dstFmt, dstWidth, dstHeight);
        }
    }

    if (drect.width != dstWidth || drect.height != dstHeight) {
        im_rect dst_whole_rect = {0, 0, dstWidth, dstHeight};
        char imcolor;
        char* p_imcolor = &imcolor;
        p_imcolor[0] = color;
        p_imcolor[1] = color;
        p_imcolor[2] = color;
        p_imcolor[3] = color;
        printf("fill dst image (x y w h)=(%d %d %d %d) with color=0x%x\n",
            dst_whole_rect.x, dst_whole_rect.y, dst_whole_rect.width, dst_whole_rect.height, imcolor);
        ret_rga = imfill(rga_buf_dst, dst_whole_rect, imcolor);
        if (ret_rga <= 0) {
            if (dst != NULL) {
                size_t dst_size = get_image_size(dst_img);
                memset(dst, color, dst_size);
            } else {
                printf("Warning: Can not fill color on target image\n");
            }
        }
    }

    // rga process
    ret_rga = improcess(rga_buf_src, rga_buf_dst, pat, srect, drect, prect, usage);
    if (ret_rga <= 0) {
        printf("Error on improcess STATUS=%d\n", ret_rga);
        printf("RGA error message: %s\n", imStrError((IM_STATUS)ret_rga));
        ret = -1;
    }

err:
    if (rga_handle_src > 0) {
        releasebuffer_handle(rga_handle_src);
    }

    if (rga_handle_dst > 0) {
        releasebuffer_handle(rga_handle_dst);
    }

    // printf("finish\n");
    return ret;
}

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

int get_image_size(image_buffer_t* image)
{
    if (image == NULL) {
        return 0;
    }
    switch (image->format)
    {
    case IMAGE_FORMAT_GRAY8:
        return image->width * image->height;
    case IMAGE_FORMAT_RGB888:
        return image->width * image->height * 3;    
    case IMAGE_FORMAT_RGBA8888:
        return image->width * image->height * 4;
    case IMAGE_FORMAT_YUV420SP_NV12:
    case IMAGE_FORMAT_YUV420SP_NV21:
        return image->width * image->height * 3 / 2;
    default:
        break;
    }
}

rkDeeplabV3::rkDeeplabV3(const std::string &model_path)
{
    this->model_path = model_path;
}

int rkDeeplabV3::init(rknn_context *ctx_in, bool share_weight)
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
    // if (!Gpu_Impl)
    //     Gpu_Impl = std::make_shared<gpu_compose_impl>(); 
    return 0;
}

rknn_context *rkDeeplabV3::get_pctx()
{
    return &ctx;
}

Results_Package rkDeeplabV3::infer(Frame_Package frame_package)
{
        std::lock_guard<std::mutex> lock(mtx);
    cv::Mat img;

    cv::cvtColor(frame_package.frame, img, cv::COLOR_BGR2RGB);
    cv::resize(img,img, cv::Size(513, 513));
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
    //outputs_tensor = (rknn_output*)malloc(sizeof(rknn_output)*(io_num.n_output));
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 1;
    }
    Results_Package Results_package;
    // 模型推理/Model inference
    auto start = std::chrono::high_resolution_clock::now();
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "rknn run cost: " << float(duration.count()/1000.0) << " ms" << std::endl;
    
    cv::resize(frame_package.frame, frame_package.frame, cv::Size(65, 65));
    // 后处理/Post-processing
    float *output = (float *)outputs[0].buf;
    sigmoid(output , output_attrs[0].n_elems);
    draw_segment_image(output , frame_package.frame);
    cv::resize(frame_package.frame, frame_package.frame,cv::Size(513, 513), 0, 0, cv::INTER_CUBIC);

    Results_package.frame_origin = frame_package.frame;
    
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    

    return Results_package;
}

int rkDeeplabV3::draw_segment_image(float* result, cv::Mat& result_img) {
    int height = result_img.rows;
    int width = result_img.cols;
    int num_class = 21;

    // [1, class, height, width] -> [1, 3, height, width]
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int maxClassIndex = 0;

            // 找到概率最高的类别
            for (int c = 1; c < num_class; c++) {
                int currentIndex = c * (height * width) + y * width + x;
                int maxClassPos = maxClassIndex * (height * width) + y * width + x;

                if (result[currentIndex] > result[maxClassPos]) {
                    maxClassIndex = c;
                }
            }
            Color foundColor;
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

Color rkDeeplabV3::getColorById(int id) {
    for (const auto& entry : label) {
        if (entry.id == id) {
            return entry.color;
        }
    }
    return Color(0, 0, 0);
}

// // 简单的 Upsample 实现
// void upsample(const std::vector<float>& input, std::vector<float>& output,
//               int inputHeight, int inputWidth, int outputHeight, int outputWidth) {
//     // 简单的双线性插值
//     for (int y = 0; y < outputHeight; ++y) {
//         for (int x = 0; x < outputWidth; ++x) {
//             int sourceY = static_cast<int>(std::floor(y * static_cast<float>(inputHeight) / outputHeight));
//             int sourceX = static_cast<int>(std::floor(x * static_cast<float>(inputWidth) / outputWidth));

//             // 简单复制像素值，你也可以使用更复杂的插值方法
//             output[y * outputWidth + x] = input[sourceY * inputWidth + sourceX];
//         }
//     }
// }

// // 简单的 Softmax 实现
// void softmax(std::vector<float>& input) {
//     // 计算指数和
//     float expSum = 0.0f;
//     for (float value : input) {
//         expSum += std::exp(value);
//     }

//     // 计算 Softmax
//     for (float& value : input) {
//         value = std::exp(value) / expSum;
//     }
// }


rkDeeplabV3::~rkDeeplabV3()
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
