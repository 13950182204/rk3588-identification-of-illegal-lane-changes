#pragma once

#include "ThreadPool.hpp"
#include "blocking_queue.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <mutex>
#include <queue>
#include <memory>
#include <thread>

// rknnModel模型类, inputType模型输入类型, outputType模型输出类型
template <typename rknnModel, typename inputType, typename outputType>
class rknnPool
{
private:
    int threadNum;
    std::string modelPath;

    long long id;
    std::mutex idMtx, queueMtx;
    std::unique_ptr<dpool::ThreadPool> pool;
    std::queue<std::future<outputType>> futs;
    std::vector<std::shared_ptr<rknnModel>> models;
    std::thread _worker_thread;
    BlockingQueue<Frame_Package>&  BlockingQueue_;
    BlockingQueue<Results_Package>&  Output_Queue;
protected:
    int getModelId();
    void run();

public:
    rknnPool(const std::string modelPath, int threadNum , BlockingQueue<Frame_Package>&  BlockingQueue_ , BlockingQueue<Results_Package>&  Output_Queue);
    int init();
    // 模型推理/Model inference
    int put(inputType inputData);
    // 获取推理结果/Get the results of your inference
    outputType get(outputType outputData);
    // 开启线程
    void start();

    ~rknnPool();

};

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::rknnPool(const std::string modelPath, int threadNum , BlockingQueue<Frame_Package>&  BlockingQueue_ , BlockingQueue<Results_Package>&  Output_Queue)
: BlockingQueue_(BlockingQueue_) , Output_Queue(Output_Queue)
{
    this->modelPath = modelPath;
    this->threadNum = threadNum;
    this->id = 0;
}

template <typename rknnModel, typename inputType, typename outputType>
void rknnPool<rknnModel, inputType, outputType>::start(){
    _worker_thread = std::thread(&rknnPool<rknnModel, inputType, outputType>::run, this);
}

template <typename rknnModel, typename inputType, typename outputType>
void rknnPool<rknnModel, inputType, outputType>::run(){
    while(true){
        cv::Mat frame;
        Results_Package Results; 
        if (BlockingQueue_.Size() >= 1){
            
            put(BlockingQueue_.Take());
            
            Output_Queue.Put(get(Results));

        }
    }
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::init()
{
    try
    {
        this->pool = std::make_unique<dpool::ThreadPool>(this->threadNum);
        for (int i = 0; i < this->threadNum; i++)
            models.push_back( std::make_shared<rknnModel>(this->modelPath.c_str()) );
    }
    catch (const std::bad_alloc &e)
    {
        std::cout << "Out of memory: " << e.what() << std::endl;
        return -1;
    }
    // 初始化模型/Initialize the model
    for (int i = 0, ret = 0; i < threadNum; i++)
    {
        ret = models[i]->init(models[0]->get_pctx(), i != 0);
        if (ret != 0)
            return ret;
    }
    
    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::getModelId()
{
    std::lock_guard<std::mutex> lock(idMtx);
    int modelId = id % threadNum;
    id++;
    return modelId;
}


template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::put(inputType inputData)
{
    futs.push(pool->submit(&rknnModel::infer, models[this->getModelId()], inputData));
    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
outputType rknnPool<rknnModel, inputType, outputType>::get(outputType outputData)
{
    std::lock_guard<std::mutex> lock(queueMtx);
    if(futs.empty() == true)
        std::cout << "get information err!!" << std::endl;
    outputData = futs.front().get();
    futs.pop();
    return outputData;
}

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::~rknnPool()
{
    while (!futs.empty())
    {
        outputType temp = futs.front().get();
        futs.pop();
    }
}

