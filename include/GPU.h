#pragma once

#include "cuda_runtime.h"
#include <iostream>
#include <queue>
#include <deque>
#include <functional>
#include <memory>
#include <utility>
#include "Spinnaker.h"
#include <pthread.h>

#include <boost/lockfree/queue.hpp>

class CudaBufferManager
{
    public:
    CudaBufferManager(size_t width, size_t height, size_t nbuffers = 40):
        phaseMapBufferPool(nbuffers)
    {
        for(int i=0; i<nbuffers; i++)
        {
            float* phaseMapBuffer;
            cudaError_t error = cudaMalloc(&phaseMapBuffer, width*height*sizeof(float));
            if(error != cudaSuccess)
            {
                std::cout << "Failed to allocate cuda memory: " << std::string(cudaGetErrorString(error)) << std::endl;
            }
            phaseMapBufferPool.push(phaseMapBuffer);
        }
    }

    ~CudaBufferManager()
    {
        while(!phaseMapBufferPool.empty())
        {
            float* phaseMapBuffer;
            phaseMapBufferPool.pop(phaseMapBuffer);
            cudaFree(phaseMapBuffer);
        }
    }

    std::shared_ptr<float> allocPhaseMap()
    {
        float* phaseMapBuffer;
        if(!phaseMapBufferPool.pop(phaseMapBuffer))
            return nullptr;
        
        return std::shared_ptr<float>(phaseMapBuffer, [this](float* ptr){
            phaseMapBufferPool.push(ptr);
        });
    }

    private:
    boost::lockfree::queue<float*> phaseMapBufferPool;
};

class GPU
{
public:
    enum class Algorithm
    {
        NOVAK = 0,
        FOURPOINTS = 1,
        CARRE = 2
    };

    enum class BufferMode
    {
        SLIDEWIN = 0,
        NEWSET = 1
    };

    GPU(int width, int height);
    ~GPU();
    void getCudaVersion();
    bool calcPhaseMap(Spinnaker::ImagePtr image, 
                        std::shared_ptr<float> phaseMap, 
                        Algorithm algorithm = Algorithm::CARRE,
                        BufferMode bufferMode = BufferMode::NEWSET);

    bool calcPhaseImage(std::shared_ptr<float> phaseMap, 
                        cudaSurfaceObject_t phaseImage);

private:

    unsigned eleCount;
    int blockPerGrid;
    int threadPerBlock;
    int N;
    int width;

    uint8_t* imageBuffer;

    cudaStream_t stream1;

    uint8_t* cosineBuffer = nullptr;
    float* phaseBuffer = nullptr;

    std::deque<float*> buffers;
};
