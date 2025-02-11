#include "GPU.h"
#include <opencv2/opencv.hpp>

GPU::GPU(int width, int height):
    phaseHostBuffer(nullptr),
    cosineHostBuffer(nullptr)
{
    eleCount = 0;
    N = width * height;
    blockPerGrid = (N + 256 - 1) / 256;

    cudaError_t error = cudaMallocManaged(&phaseHostBuffer, N*sizeof(float), cudaMemAttachHost);
    if(error != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate cuda memory: " + std::string(cudaGetErrorString(error)));
    }

    error = cudaMallocManaged(&cosineHostBuffer, N*sizeof(float), cudaMemAttachHost);
    if(error != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate cuda memory: " + std::string(cudaGetErrorString(error)));
    }

    for(int i=0; i<5; i++)
    {
        float* buffer;
        error = cudaMallocManaged(&buffer, N*sizeof(float));
        if(error != cudaSuccess)
        {
            throw std::runtime_error("Failed to allocate image buffer: " + std::string(cudaGetErrorString(error)));
        }
        buffers.push_back(buffer);
    }
};

GPU::~GPU()
{
    cudaFree(cosineHostBuffer);
    cudaFree(phaseHostBuffer);
    while(buffers.size() > 0)
    {
        float* buffer = buffers.front();
        cudaFree(buffer);
        buffers.pop_front();
    }
};

void GPU::getCudaVersion()
{
    int rVersion = 0;

    cudaError_t runtimeStatus = cudaRuntimeGetVersion(&rVersion);
    std:: cout << rVersion << std::endl;

}

float* GPU::runNovak(Spinnaker::ImagePtr newImage)
{
    // buffer[4]->Release()

    uint8_t* inDeviceBuffer = reinterpret_cast<uint8_t*>(newImage->GetData());

    float* outBuffer = buffers.back();
    buffers.pop_back();

    convert_type<<<blockPerGrid,256>>>(inDeviceBuffer, outBuffer, N);
    cudaDeviceSynchronize();

    buffers.push_front(outBuffer);

    if (eleCount < 5)
    {
        eleCount++;
        return nullptr;
    }

    newImage -> Release();
    
    compute_phase<<<blockPerGrid,256>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        buffers[4],
                                        phaseHostBuffer,
                                        cosineHostBuffer,
                                        N);
    cudaDeviceSynchronize();

    return cosineHostBuffer;
}