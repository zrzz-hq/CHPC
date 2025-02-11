#include "cuda_runtime.h"
#include <iostream>
#include <deque>
#include "Spinnaker.h"

// #ifdef _cplusplus
// extern "C" {
// #endif
__global__ void convert_type(uint8_t *inp, float *outp, int N);
__global__ void compute_phase(float* p1, float* p2, float* p3, float* p4, float* p5, float *phase, float* cosine, int N);

// #ifdef _cplusplus
// }
// #endif

class GPU
{
public:
    GPU(int width, int height);
    ~GPU();
    void getCudaVersion();
    float* runNovak(Spinnaker::ImagePtr image);

private:

    unsigned eleCount;
    int blockPerGrid;
    int N;

    float* phaseHostBuffer;
    float* cosineHostBuffer;
    std::deque<float*> buffers;
};
