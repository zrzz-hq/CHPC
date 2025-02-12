#include <cuda_runtime.h>
#include <GPU.h>

__global__ void convert_type(uint8_t *inp, float *outp, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
    {
        outp[idx] = __uint2float_rn(inp[idx]);
    }
}

__global__ void compute_phase(float* p1, float* p2, float* p3, float* p4, float* p5, float *phase, uint8_t* cosine, int N) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) 
    {
        float denominator = 2.0 * p3[idx] - p1[idx] - p5[idx];
        float A = p2[idx] - p4[idx];
        float B = p1[idx] - p5[idx] + 10.0;
        float numerator = sqrt(fabs(4.0 * A * A - B * B));
        float pm = (A > 0) - (A < 0); // Sign function
        phase[idx] = atan2f(pm * numerator, denominator);
        cosine[idx] = __float2uint_rn(cos(phase[idx]) * 255);
    }
}


//__global__ void compute_cosine(float *phase, float *cosine, int N) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (idx < N) {
//        cosine[idx] = cos(phase[idx]);
//    }
//}