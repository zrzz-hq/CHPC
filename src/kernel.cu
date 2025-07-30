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

__global__ void novak(float* p1, float* p2, float* p3, float* p4, float* p5, float *phase, int N) 
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
    }
}


__global__ void four_point(float* p1, float* p2, float* p3, float* p4, float *phase, int N) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) 
    {
        float A = p4[idx] - p2[idx];
        float B = p1[idx] - p3[idx];
        phase[idx] = atan2f(A, B);
    }
}

__global__ void carres(float* p1, float* p2, float* p3, float* p4, float *phase, int N) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) 
    {
        float denominator = p2[idx] + p3[idx] - p1[idx] - p4[idx];
        float A = p2[idx] - p3[idx];
        float B = p1[idx] - p4[idx];
        float numerator = sqrt(fabs((A + B) * (3.0 * A - B)));
        float pm = (A > 0) - (A < 0); // Sign function
        phase[idx] = atan2f(pm * numerator, denominator);
    }
}

__global__ void create_phaseImage(float* phaseMap, cudaSurfaceObject_t phaseImage, int N, int width) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) 
    {
        int x = idx % width;
        int y = idx / width;
        float norm = fminf(fmaxf((phaseMap[idx] + M_PI) / (2.0 * M_PI), 0.0f), 1.0f);

        float r = fminf(fmaxf(1.5f - fabsf(4.0f * norm - 3.0f), 0.0f), 1.0f);
        float g = fminf(fmaxf(1.5f - fabsf(4.0f * norm - 2.0f), 0.0f), 1.0f);
        float b = fminf(fmaxf(1.5f - fabsf(4.0f * norm - 1.0f), 0.0f), 1.0f); 

        uchar4 pixel;
        pixel.x = __float2uint_rn(r * 255.0f);
        pixel.y = __float2uint_rn(g * 255.0f);
        pixel.z = __float2uint_rn(b * 255.0f);
        pixel.w = 255;
        surf2Dwrite(pixel, phaseImage, x * sizeof(uchar4), y);
    }
}