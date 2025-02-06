#include <cuda_runtime.h>
#include <GPU.h>

__global__ void compute_phase(uint8_t *p1, uint8_t *p2, uint8_t *p3, uint8_t *p4, uint8_t *p5, float *phase, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) 
    {
        float p1f = __uint2float_rn(p1[idx]);
        float p2f = __uint2float_rn(p2[idx]);
        float p3f = __uint2float_rn(p3[idx]);
        float p4f = __uint2float_rn(p4[idx]);
        float p5f = __uint2float_rn(p5[idx]);


        float denominator = 2.0 * p3f - p1f - p5f;
        float A = p2f - p4f;
        float B = p1f - p5f + 10.0;
        float numerator = sqrt(fabs(4.0 * A * A - B * B));
        float pm = (A > 0) - (A < 0); // Sign function
        phase[idx] = atan2f(pm * numerator, denominator);
        phase[idx] = cos(phase[idx]);
    }
}


//__global__ void compute_cosine(float *phase, float *cosine, int N) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (idx < N) {
//        cosine[idx] = cos(phase[idx]);
//    }
//}