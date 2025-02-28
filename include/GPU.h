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
    // pthread_mutex_t cosineBufferMutex;

};
