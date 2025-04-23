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
__global__ void novak(float* p1, float* p2, float* p3, float* p4, float* p5, float *phase, int N);
__global__ void four_point(float* p1, float* p2, float* p3, float* p4, float *phase, int N);
__global__ void carres(float* p1, float* p2, float* p3, float* p4, float *phase, int N);
__global__ void generate_image(float* map, uint8_t* image, int N);
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
        bool bufferMode = false;
    };

    class Future
    {
        public:
        friend class GPU;
        Future(){}
        ~Future(){}

        Buffer getResult()
        {
            return buffer;
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

            if(!buffer.isValid())
                return false;

            return true;
        }

        private:

        Buffer buffer;
        cudaStream_t stream;
        bool expired = true;

        Future(Buffer buffer, cudaStream_t stream):
            buffer(buffer),
            stream(stream)
        {
            expired = false;
        }
    };

    GPU(size_t width, size_t height);
    ~GPU();
    std::shared_ptr<Config> getConfig();
    void getCudaVersion();
    Buffer run(Spinnaker::ImagePtr image);
    Buffer generateImage(Buffer phaseMap);

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
