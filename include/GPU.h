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
    bool run(Spinnaker::ImagePtr image, 
            std::shared_ptr<float> phaseMap, 
            std::shared_ptr<uint8_t> phaseImage,
            Algorithm algorithm = Algorithm::CARRE,
            BufferMode bufferMode = BufferMode::NEWSET);

private:

    unsigned eleCount;
    int blockPerGrid;
    int threadPerBlock;
    int N;

    uint8_t* imageBuffer;

    cudaStream_t stream1;
    cudaStream_t stream2;

    uint8_t* cosineBuffer = nullptr;
    float* phaseBuffer = nullptr;

    std::deque<float*> buffers;
};
