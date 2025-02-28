#include "GPU.h"

GPU::GPU(int width, int height, size_t nPhaseBuffers):
    cosineBuffers(nPhaseBuffers),
    phaseBuffers(nPhaseBuffers)
{
    eleCount = 0;
    N = width * height;
    threadPerBlock = 256;
    blockPerGrid = (N + threadPerBlock - 1) / threadPerBlock;

    cudaError_t error;

    for(int i=0; i<nPhaseBuffers; i++)
    {
        float* phaseBuffer;
        error = cudaMalloc(&phaseBuffer, N*sizeof(float));
        if(error != cudaSuccess)
        {
            throw std::runtime_error("Failed to allocate cuda memory: " + std::string(cudaGetErrorString(error)));
        }
        phaseBuffers.push(phaseBuffer);
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
        error = cudaMallocManaged(&cosineBuffer, N*sizeof(uint8_t)*3, cudaMemAttachHost);
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

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Maximum threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Maximum threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Maximum grid size: (" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << ")" << std::endl;

    // pthread_mutex_init(&cosineBufferMutex, NULL);
};

GPU::~GPU()
{
    // cudaFree(cosineHostBuffer);
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

std::pair<std::shared_ptr<uint8_t>,std::shared_ptr<float>> GPU::runNovak(Spinnaker::ImagePtr newImage)
{
    std::shared_ptr<uint8_t> phaseImage = nullptr;
    std::shared_ptr<float> phaseMap = nullptr;
    float* newImageDev = nullptr;
    uint8_t* cosineBuffer = nullptr;
    float* phaseBuffer = nullptr;

    // cudaError_t error = cudaMemcpyAsync(imageBuffer, newImage->GetData(), N*sizeof(uint8_t), cudaMemcpyHostToDevice, stream1);

    // if(error != cudaSuccess)
    // {
    //     std::cout << "Failed to copy the image to gpu: " << cudaGetErrorString(error) << std::endl;
    //     goto ret;
    // }

    newImageDev = buffers.back();

    convert_type<<<blockPerGrid,threadPerBlock, 0, stream1>>>(reinterpret_cast<uint8_t*>(newImage->GetData()), newImageDev, N);

    if (eleCount < 5)
    {
        eleCount++;
        goto updateBuffers;
    }


    if(!cosineBuffers.pop(cosineBuffer) || !phaseBuffers.pop(phaseBuffer))
        goto updateBuffers;
    
    
    compute_phase<<<blockPerGrid,threadPerBlock, 0, stream2>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        buffers[4],
                                        phaseBuffer,
                                        cosineBuffer,
                                        N);
    phaseImage = std::shared_ptr<uint8_t>(cosineBuffer, std::bind(&GPU::cosineBufferDeleter, this, std::placeholders::_1));
    phaseMap = std::shared_ptr<float>(phaseBuffer, std::bind(&GPU::phaseBufferDeleter, this, std::placeholders::_1));

updateBuffers:    
    buffers.pop_back();
    buffers.push_front(newImageDev);
ret:
    cudaDeviceSynchronize();
    return {phaseImage, phaseMap};
}

void GPU::cosineBufferDeleter(uint8_t* ptr)
{
    if(ptr != nullptr && !cosineBuffers.push(ptr))
    {
        cudaFree(ptr);
    }
    
}

void GPU::phaseBufferDeleter(float* ptr)
{
    if(ptr != nullptr && !phaseBuffers.push(ptr))
    {
        cudaFree(ptr);
    }
    
}