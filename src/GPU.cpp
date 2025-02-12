#include "GPU.h"
#include <opencv2/opencv.hpp>

GPU::GPU(int width, int height, size_t nPhaseBuffers):
    phaseHostBuffer(nullptr)
{
    eleCount = 0;
    N = width * height;
    threadPerBlock = 256;
    blockPerGrid = (N + threadPerBlock - 1) / threadPerBlock;

    cudaError_t error = cudaMallocManaged(&phaseHostBuffer, N*sizeof(float), cudaMemAttachHost);
    if(error != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate cuda memory: " + std::string(cudaGetErrorString(error)));
    }

    // error = cudaMallocManaged(&cosineHostBuffer, N*sizeof(uint8_t), cudaMemAttachHost);
    // if(error != cudaSuccess)
    // {
    //     throw std::runtime_error("Failed to allocate cuda memory: " + std::string(cudaGetErrorString(error)));
    // }

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

    for(int i=0; i<nPhaseBuffers; i++)
    {
        uint8_t* cosineBuffer;
        error = cudaMallocManaged(&cosineBuffer, N*sizeof(uint8_t), cudaMemAttachHost);
        if(error != cudaSuccess)
        {
            throw std::runtime_error("Failed to allocate phase buffer: " + std::string(cudaGetErrorString(error)));
        }
        cosineBuffers.push(cosineBuffer);
    }

    pthread_mutex_init(&cosineBufferMutex, NULL);
};

GPU::~GPU()
{
    // cudaFree(cosineHostBuffer);
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

std::shared_ptr<uint8_t> GPU::runNovak(Spinnaker::ImagePtr newImage)
{
    // buffer[4]->Release()

    uint8_t* inDeviceBuffer = reinterpret_cast<uint8_t*>(newImage->GetData());

    float* outBuffer = buffers.back();
    buffers.pop_back();

    convert_type<<<blockPerGrid,threadPerBlock>>>(inDeviceBuffer, outBuffer, N);
    cudaDeviceSynchronize();

    buffers.push_front(outBuffer);

    if (eleCount < 5)
    {
        eleCount++;
        return nullptr;
    }
    
    uint8_t* cosineBuffer;
    pthread_mutex_lock(&cosineBufferMutex);
    if(cosineBuffers.size() > 0)
    {
        cosineBuffer = cosineBuffers.front();
        cosineBuffers.pop();
    }
    pthread_mutex_unlock(&cosineBufferMutex);

    if(cosineBuffer == nullptr)
        return nullptr;
    
    compute_phase<<<blockPerGrid,threadPerBlock>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        buffers[4],
                                        phaseHostBuffer,
                                        cosineBuffer,
                                        N);
    
    cudaDeviceSynchronize();

    return std::shared_ptr<uint8_t>(cosineBuffer, std::bind(&GPU::phaseBufferDeleter, this, std::placeholders::_1));
}

void GPU::phaseBufferDeleter(uint8_t* ptr)
{
    if(ptr != nullptr)
    {
        pthread_mutex_lock(&cosineBufferMutex);
        cosineBuffers.push(ptr);
        pthread_mutex_unlock(&cosineBufferMutex);
    }
    
}