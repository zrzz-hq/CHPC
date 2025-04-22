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

#include "buffer.h"

// #ifdef _cplusplus
// extern "C" {
// #endif
__global__ void convert_type(uint8_t *inp, float *outp, int N);
__global__ void novak(float* p1, float* p2, float* p3, float* p4, float* p5, float *phase, uint8_t* cosine, int N);
__global__ void four_point(float* p1, float* p2, float* p3, float* p4, float *phase, uint8_t* cosine, int N);
__global__ void carres(float* p1, float* p2, float* p3, float* p4, float *phase, uint8_t* cosine, int N);
// #ifdef _cplusplus
// }
// #endif

class GPU
{
public:
    struct Config
    {
        int algorithmIndex = 2;
        const char* algorithmNames[3] = {"Novak", "FourPoints", "Carre"};
        int nAlgorithms = 3;
    };

    class Future
    {
        public:
        friend class GPU;
        ~Future(){}

        std::pair<Buffer, Buffer> getResult()
        {
            return {phaseMap, phaseImage};
        }

        bool join()
        {
            if(expired)
                return false;
            
            expired = true;
            
            cudaError_t error = cudaStreamSynchronize(stream);
            if(error != cudaSuccess)
            {
                std::cout << "Cuda error: " << cudaGetErrorString(error) << std::endl;
                cudaDeviceReset();
                return false;
            }

            if(!phaseMap.isVaild() || !phaseImage.isVaild())
                return false;

            return true;
        }

        private:

        Buffer phaseMap;
        Buffer phaseImage;
        cudaStream_t stream;
        bool expired;

        Future(Buffer phaseMap, Buffer phaseImage, cudaStream_t stream):
            phaseMap(phaseMap),
            phaseImage(phaseImage),
            stream(stream)
        {
            expired = false;
        }
    };

    GPU(size_t width, size_t height);
    ~GPU();
    std::shared_ptr<Config> getConfig();
    void getCudaVersion();
    std::shared_ptr<Future> runAsync(Spinnaker::ImagePtr image);

private:

    unsigned eleCount;
    int blockPerGrid;
    int threadPerBlock;
    int N;

    uint8_t* imageBuffer;

    cudaStream_t stream1;
    cudaStream_t stream2;

    std::deque<float*> buffers;
    std::shared_ptr<Config> config;

    std::shared_ptr<BufferPool> phaseMapPool;
    std::shared_ptr<BufferPool> phaseImagePool;
};
