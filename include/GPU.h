#include "cuda_runtime.h"
#include <iostream>
#include <queue>
#include <deque>
#include <functional>
#include <memory>
#include "Spinnaker.h"
#include <pthread.h>

// #ifdef _cplusplus
// extern "C" {
// #endif
__global__ void convert_type(uint8_t *inp, float *outp, int N);
__global__ void compute_phase(float* p1, float* p2, float* p3, float* p4, float* p5, float *phase, uint8_t* cosine, int N);
// #ifdef _cplusplus
// }
// #endif

class GPU
{
public:
    GPU(int width, int height, size_t nPhaseBuffers);
    ~GPU();
    void getCudaVersion();
    std::shared_ptr<uint8_t> runNovak(Spinnaker::ImagePtr image);

private:

    unsigned eleCount;
    int blockPerGrid;
    int threadPerBlock;
    int N;

    float* phaseHostBuffer;

    std::queue<uint8_t*> cosineBuffers;
    std::deque<float*> buffers;

    void phaseBufferDeleter(uint8_t* ptr);
    pthread_mutex_t cosineBufferMutex;

};
