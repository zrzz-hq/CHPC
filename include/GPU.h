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

template<typename T>
class CudaBufferPool: public std::enable_shared_from_this<CudaBufferPool<T>>
{
    public:
    CudaBufferPool(size_t width, size_t height, size_t nbuffers = 40):
        pool(nbuffers)
    {
        for(int i=0; i<nbuffers; i++)
        {
            T* buffer;
            cudaError_t error = cudaMalloc(&buffer, width*height*sizeof(T));
            if(error != cudaSuccess)
            {
                std::cout << "Failed to allocate cuda memory: " << std::string(cudaGetErrorString(error)) << std::endl;
            }
            pool.push(buffer);
        }
    }

    ~CudaBufferPool()
    {
        while(!pool.empty())
        {
            float* buffer;
            pool.pop(buffer);
            cudaFree(buffer);
        }
    }

    std::shared_ptr<T> alloc()
    {
        T* buffer;
        if(!pool.pop(buffer))
            return nullptr;
        
        return std::shared_ptr<T>(buffer, [self = this->shared_from_this()](float* ptr){
            self->pool.push(ptr);
        });
    }

    private:
    boost::lockfree::queue<T*> pool;
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
    bool calcPhaseMap(Spinnaker::ImagePtr image, 
                        std::shared_ptr<float> phaseMap, 
                        Algorithm algorithm = Algorithm::CARRE,
                        BufferMode bufferMode = BufferMode::NEWSET);

    void calcPhaseImage(std::shared_ptr<float> phaseMap, 
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
