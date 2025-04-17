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
    // enum class PhaseAlgorithm
    // {
    //     NOVAK = 0,
    //     FOURPOINT = 1,
    //     CARRE = 2
    // };

    struct Config
    {
        int algorithmIndex = 2;
        const char* algorithmNames[3] = {"Novak", "FourPoints", "Carre"};
        int nAlgorithms = 3;
    };

    GPU(int width, int height, size_t nPhaseBuffers);
    ~GPU();
    std::shared_ptr<Config> getConfig();
    void getCudaVersion();
    std::pair<std::shared_ptr<uint8_t>,std::shared_ptr<float>>  runNovak(Spinnaker::ImagePtr image);

private:

    unsigned eleCount;
    int blockPerGrid;
    int threadPerBlock;
    int N;

    uint8_t* imageBuffer;

    cudaStream_t stream1;
    cudaStream_t stream2;

    boost::lockfree::queue<uint8_t*> cosineBuffers;
    boost::lockfree::queue<float*> phaseBuffers;
    std::deque<float*> buffers;

    void phaseBufferDeleter(float* ptr);
    void cosineBufferDeleter(uint8_t* ptr);
    std::shared_ptr<Config> config;

};
