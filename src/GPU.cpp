#include "GPU.h"

__global__ void convert_type(uint8_t *inp, float *outp, int N);
__global__ void novak(float* p1, float* p2, float* p3, float* p4, float* p5, float *phase, int N);
__global__ void four_point(float* p1, float* p2, float* p3, float* p4, float *phase, int N);
__global__ void carres(float* p1, float* p2, float* p3, float* p4, float *phase, int N);
__global__ void create_phaseImage(float* phaseMap, uint8_t* phaseImage, int N);

GPU::GPU(int width, int height)
{
    eleCount = 0;
    N = width * height;
    threadPerBlock = 256;
    blockPerGrid = (N + threadPerBlock - 1) / threadPerBlock;

    cudaError_t error;

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

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
};

void GPU::getCudaVersion()
{
    int rVersion = 0;

    cudaError_t runtimeStatus = cudaRuntimeGetVersion(&rVersion);
    std:: cout << rVersion << std::endl;

}

bool GPU::calcPhaseMap(Spinnaker::ImagePtr image, 
            std::shared_ptr<float> phaseMap, 
            Algorithm algorithm, 
            BufferMode bufferMode)
{
    float* newImageDev = buffers.back();

    cudaError_t error = cudaMemcpyAsync(imageBuffer, image->GetData(), N, cudaMemcpyHostToDevice, stream1);
    if(error != cudaSuccess)
    {
        std::cout << "Failed to copy image memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    buffers.pop_back();
    buffers.push_front(newImageDev);
    
    convert_type<<<blockPerGrid, threadPerBlock, 0, stream1>>>(imageBuffer, newImageDev, N);

    if (eleCount < 4)
    {
        eleCount++;
        return false;
    }
    else
    {
        if(bufferMode == BufferMode::NEWSET)
        {
            eleCount = 0;
        }
    }
 
    switch (algorithm)
    {
    case Algorithm::NOVAK:
        novak<<<blockPerGrid,threadPerBlock, 0, stream1>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        buffers[4],
                                        phaseMap.get(),
                                        N);
        break;
    case Algorithm::FOURPOINTS:
        four_point<<<blockPerGrid,threadPerBlock, 0, stream1>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        phaseMap.get(),
                                        N);
        break;
    case Algorithm::CARRE:
        carres<<<blockPerGrid,threadPerBlock, 0, stream1>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        phaseMap.get(),
                                        N);
    default:
        break;
    }

    error = cudaStreamSynchronize(stream1);
    if(error != cudaSuccess)
    {
        std::cout << "Failed to run phase algorithm: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}

bool GPU::calcPhaseImage(std::shared_ptr<float> phaseMap, std::shared_ptr<uint8_t> phaseImage)
{
    create_phaseImage<<<blockPerGrid,threadPerBlock, 0, stream1>>>(phaseMap.get(), phaseImage.get(), N);
    
    cudaError_t error = cudaStreamSynchronize(stream1);
    if(error != cudaSuccess)
    {
        std::cout << "Failed to generate phase image: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    return true;
}
