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

    GPU(int width, int height, size_t nPhaseBuffers);
    ~GPU();
    std::shared_ptr<Config> getConfig();
    void getCudaVersion();
    std::pair<std::shared_ptr<float>, std::shared_ptr<uint8_t>> run(Spinnaker::ImagePtr image);

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

    uint8_t* cosineBuffer = nullptr;
    float* phaseBuffer = nullptr;

    std::deque<float*> buffers;

    void phaseBufferDeleter(float* ptr);
    void cosineBufferDeleter(uint8_t* ptr);
    std::shared_ptr<Config> config;

};
