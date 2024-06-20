#pragma once

// 定义颜色的类型
using Color = std::tuple<int, int, int>;

// 定义一个结构体，表示表的一行
struct Entry {
    int id;
    const char* name;
    Color color;
};

typedef enum {
    IMAGE_FORMAT_GRAY8,
    IMAGE_FORMAT_RGB888,
    IMAGE_FORMAT_RGBA8888,
    IMAGE_FORMAT_YUV420SP_NV21,
    IMAGE_FORMAT_YUV420SP_NV12,
} image_format_t;


/**
 * @brief Image buffer
 * 
 */
typedef struct {
    int width;
    int height;
    int width_stride;
    int height_stride;
    image_format_t format;
    unsigned char* virt_addr;
    int size;
    int fd;
} image_buffer_t;

/**
 * @brief Image rectangle
 * 
 */
typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

struct Frame_Package{
    uint32_t id;
    cv::Mat frame;
    image_buffer_t* image;
};

struct DetectionResults
{
    
    int x1;
    int y1;
    int x2;
    int y2;
    std::string name;
    float score;

};


typedef struct _Results_Package
{
    char type ; 
    uint32_t id;
    cv::Mat frame_origin;  // 摄像头原始图像
    std::vector<DetectionResults> Results;
    unsigned char out_mask[640*384];

}Results_Package;



inline void sigmoid(float* array, int size) {


    for (int i = 0; i < size; i++) {
        array[i] = 1.0f / (1.0f + expf(array[i]));
    }

    // // Find the maximum value in the array
    // float max_val = array[0];
    // for (int i = 1; i < size; i++) {
    //     std::cout << array[i] << std::endl;
    // }
}


// inline void softmax(float* array, int size) {
//     // Find the maximum value in the array
//     float max_val = array[0];
//     for (int i = 1; i < size; i++) {
//         if (array[i] > max_val) {
//             max_val = array[i];
//         }
        
//     }
//     std::cout << array[1] << " first1"<<  std::endl;
//     // Subtract the maximum value from each element to avoid overflow
//     for (int i = 0; i < size; i++) {
//         array[i] -= max_val;
//     }
//     std::cout << max_val << " max"<<  std::endl;
//     // Compute the exponentials and sum
//     float sum = 0.0;
//     for (int i = 0; i < size; i++) {
//         array[i] = expf(array[i]);
//         sum += array[i];
         
//     }
//     std::cout << array[1] << " first"<<  std::endl;
//     //std::cout << expf(array[1]) << " exp"<<  std::endl;
//     std::cout << sum << " sum"<<  std::endl;
//     // Normalize the array by dividing each element by the sum
//     for (int i = 0; i < size; i++) {
//         array[i] /= sum;
//     }
//     float max_val1 = array[0];
//     for (int i = 1; i < size; i++) {
//         if (array[i] > max_val) {
//             max_val1 = array[i];
//         }
//     }
//     std::cout << array[1] << " addfi"<<  std::endl;
// }

inline void softmax(float* array, int size) {
    // Find the maximum value in the array
    float max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
        // std::cout << array[i] << " addfi"<<  std::endl;
    }

    // Subtract the maximum value from each element to avoid overflow
    for (int i = 0; i < size; i++) {
        array[i] -= max_val;
    }

    // Compute the exponentials and sum
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        array[i] = expf(array[i]);
        sum += array[i];
    }

    // Normalize the array by dividing each element by the sum
    for (int i = 0; i < size; i++) {
        array[i] /= sum;
        // Apply a small constant to prevent division by zero
        array[i] += 1e-8;
    }
    
    for (int i = 1; i < size; i++) {
        std::cout << array[i] << std::endl;
    }
}