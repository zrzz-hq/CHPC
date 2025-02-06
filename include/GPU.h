#include "cuda_runtime.h"
#include <iostream>
#include "Spinnaker.h"

// #ifdef _cplusplus
// extern "C" {
// #endif

__global__ void compute_phase(uint8_t *p1, uint8_t *p2, uint8_t *p3, uint8_t *p4, uint8_t *p5, float *phase, int N);

// #ifdef _cplusplus
// }
// #endif

class GPU{
public:
GPU(int width, int height);
~GPU();
void getCudaVersion();
float* runNovak(Spinnaker::ImagePtr image);

private:
Spinnaker::ImagePtr buffer[5];
unsigned eleCount;
int N;
float* outputBuffer;
int blockPerGrid;
};
