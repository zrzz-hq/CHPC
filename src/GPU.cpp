#include "GPU.h"

#ifdef _cplusplus
extern "C" {
#endif
    __global__ void convert_type(uint8_t *inp, float *outp, int N);
    __global__ void novak(float* p1, float* p2, float* p3, float* p4, float* p5, float *phase, uint8_t* cosine, int N);
    __global__ void four_point(float* p1, float* p2, float* p3, float* p4, float *phase, uint8_t* cosine, int N);
    __global__ void carres(float* p1, float* p2, float* p3, float* p4, float *phase, uint8_t* cosine, int N);
#ifdef _cplusplus
}
#endif

GPU::GPU(int width, int height, size_t nPhaseBuffers):
    cosineBuffers(nPhaseBuffers),
    phaseBuffers(nPhaseBuffers),
    config(std::make_shared<GPU::Config>())
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
        // error = cudaMallocHost(&cosineBuffer, N*sizeof(uint8_t)*3);
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

std::shared_ptr<GPU::Config> GPU::getConfig()
{
    return config;
}

std::pair<std::shared_ptr<float>, std::shared_ptr<uint8_t>> GPU::run(Spinnaker::ImagePtr newImage)
{
    float* newImageDev = buffers.back();

    cudaError_t error = cudaMemcpyAsync(imageBuffer, newImage->GetData(), N, cudaMemcpyHostToDevice, stream1);
    if(error != cudaSuccess)
    {
        std::cout << "Failed to copy image memory: " << cudaGetErrorString(error) << std::endl;
        return {nullptr, nullptr};
    }

    buffers.pop_back();
    buffers.push_front(newImageDev);
    
    convert_type<<<blockPerGrid, threadPerBlock, 0, stream1>>>(imageBuffer, newImageDev, N);

    if (eleCount < 4)
    {
        eleCount++;
        return {nullptr, nullptr};
    }
    else
    {
        if(config->bufferMode)
        {
            eleCount = 0;
        }
    }

    if(!cosineBuffers.pop(cosineBuffer))
        return {nullptr, nullptr};
    
    if(!phaseBuffers.pop(phaseBuffer))
    {
        cosineBuffers.push(cosineBuffer);
        cosineBuffer = nullptr;
        return {nullptr, nullptr};
    }
 
    switch (config->algorithmIndex)
    {
    case 0:
        novak<<<blockPerGrid,threadPerBlock, 0, stream1>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        buffers[4],
                                        phaseBuffer,
                                        cosineBuffer,
                                        N);
        break;
    case 1:
        four_point<<<blockPerGrid,threadPerBlock, 0, stream1>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        phaseBuffer,
                                        cosineBuffer,
                                        N);
        break;
    case 2:
        carres<<<blockPerGrid,threadPerBlock, 0, stream1>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        phaseBuffer,
                                        cosineBuffer,
                                        N);
    default:
        break;
    }

    error = cudaStreamSynchronize(stream1);
    if(error != cudaSuccess)
    {
        std::cout << "Failed to run phase algorithm: " << cudaGetErrorString(error) << std::endl;
        return {nullptr, nullptr};
    }

    std::shared_ptr<float> phaseMap(phaseBuffer, std::bind(&GPU::phaseBufferDeleter, this, std::placeholders::_1));
    std::shared_ptr<uint8_t> phaseImage(cosineBuffer, std::bind(&GPU::cosineBufferDeleter, this, std::placeholders::_1));

    return {phaseMap, phaseImage};
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