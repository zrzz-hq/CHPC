#include "GPU.h"
#include <opencv2/opencv.hpp>

GPU::GPU(int width, int height, size_t nPhaseBuffers):
    phaseHostBuffer(nullptr),
    cosineBuffers(nPhaseBuffers)
{
    eleCount = 0;
    N = width * height;
    threadPerBlock = 256;
    blockPerGrid = (N + threadPerBlock - 1) / threadPerBlock;

    cudaError_t error = cudaMalloc(&phaseHostBuffer, N*sizeof(float));
    if(error != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate cuda memory: " + std::string(cudaGetErrorString(error)));
    }

    error = cudaMalloc(&imageBuffer, N*sizeof(uint8_t));
    if(error != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate cuda memory: " + std::string(cudaGetErrorString(error)));
    }

    for(int i=0; i<6; i++)
    {
        float* buffer;
        error = cudaMalloc(&buffer, N*sizeof(float));
        if(error != cudaSuccess)
        {
            throw std::runtime_error("Failed to allocate image buffer: " + std::string(cudaGetErrorString(error)));
        }
        buffers.push_back(buffer);
    }

    for(int i=0; i<nPhaseBuffers; i++)
    {
        uint8_t* cosineBuffer;
        error = cudaMalloc(&cosineBuffer, N*sizeof(uint8_t));
        if(error != cudaSuccess)
        {
            throw std::runtime_error("Failed to allocate phase buffer: " + std::string(cudaGetErrorString(error)));
        }
        cosineBuffers.push(cosineBuffer);
    }

    error = cudaStreamCreate(&stream1);
    if(error != cudaSuccess)
    {
        throw std::runtime_error("Failed to create cuda stream 1: "+ std::string(cudaGetErrorString(error)));
    }
    
    error = cudaStreamCreate(&stream2);
    if(error != cudaSuccess)
    {
        throw std::runtime_error("Failed to create cuda stream 2: "+ std::string(cudaGetErrorString(error)));
    }

    // pthread_mutex_init(&cosineBufferMutex, NULL);
};

GPU::~GPU()
{
    // cudaFree(cosineHostBuffer);
    cudaFree(phaseHostBuffer);
    cudaFree(imageBuffer);
    while(buffers.size() > 0)
    {
        float* buffer = buffers.front();
        cudaFree(buffer);
        buffers.pop_front();
    }

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
};

void GPU::getCudaVersion()
{
    int rVersion = 0;

    cudaError_t runtimeStatus = cudaRuntimeGetVersion(&rVersion);
    std:: cout << rVersion << std::endl;

}

std::shared_ptr<uint8_t> GPU::runNovak(Spinnaker::ImagePtr newImage)
{
    std::shared_ptr<uint8_t> phaseImage = nullptr;
    float* newImageDev = nullptr;
    uint8_t* cosineBuffer = nullptr;

    cudaError_t error = cudaMemcpyAsync(imageBuffer, newImage->GetData(), N*sizeof(uint8_t), cudaMemcpyHostToDevice, stream1);

    if(error != cudaSuccess)
    {
        std::cout << "Failed to copy the image to gpu: " << cudaGetErrorString(error) << std::endl;
        goto ret;
    }

    newImageDev = buffers.back();

    convert_type<<<blockPerGrid,threadPerBlock, 0, stream1>>>(imageBuffer, newImageDev, N);

    if (eleCount < 5)
    {
        eleCount++;
        goto updateBuffers;
    }

    if(!cosineBuffers.pop(cosineBuffer))
        goto updateBuffers;
    
    compute_phase<<<blockPerGrid,threadPerBlock, 0, stream2>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        buffers[4],
                                        phaseHostBuffer,
                                        cosineBuffer,
                                        N);
    phaseImage = std::shared_ptr<uint8_t>(cosineBuffer, std::bind(&GPU::phaseBufferDeleter, this, std::placeholders::_1));

updateBuffers:    
    buffers.pop_back();
    buffers.push_front(newImageDev);
ret:
    cudaDeviceSynchronize();
    return phaseImage;
}

void GPU::phaseBufferDeleter(uint8_t* ptr)
{
    if(ptr != nullptr && !cosineBuffers.push(ptr))
    {
        cudaFree(ptr);
    }
    
}