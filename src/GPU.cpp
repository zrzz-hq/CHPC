#include "GPU.h"

#include <boost/log/trivial.hpp>

__global__ void convert_type(uint8_t *inp, float *outp, int N);
__global__ void novak(float* p1, float* p2, float* p3, float* p4, float* p5, float *phase, int N);
__global__ void four_point(float* p1, float* p2, float* p3, float* p4, float *phase, int N);
__global__ void carres(float* p1, float* p2, float* p3, float* p4, float *phase, int N);
__global__ void create_phaseImage(float* phaseMap, cudaSurfaceObject_t phaseImage, int N, int width);

GPU::GPU(int width, int height):width(width)
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

    int version;
    error = cudaRuntimeGetVersion(&version);
    if(error != cudaSuccess)
    {
        BOOST_LOG_TRIVIAL(warning) << "Failed to get cuda runtime version";
    }
    else
    {
        BOOST_LOG_TRIVIAL(info) << "Cuda runtime version: " << version;
    }

    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, 0);
    if(error != cudaSuccess)
    {
        BOOST_LOG_TRIVIAL(warning) << "Failed to get cuda device information";
    }
    else
    {
        BOOST_LOG_TRIVIAL(info) << "Number of SMs: " << prop.multiProcessorCount;
        BOOST_LOG_TRIVIAL(info) << "Maximum threads per SM: " << prop.maxThreadsPerMultiProcessor;
        BOOST_LOG_TRIVIAL(info) << "Maximum threads per block: " << prop.maxThreadsPerBlock;
        BOOST_LOG_TRIVIAL(info) << "Maximum grid size: (" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << ")";
    }
};

GPU::~GPU()
{
    cudaFree(imageBuffer);
    cudaStreamDestroy(stream1);
};

bool GPU::calcPhaseMap(Spinnaker::ImagePtr image, 
            std::shared_ptr<float> phaseMap, 
            Algorithm algorithm, 
            BufferMode bufferMode)
{
    float* newImageDev = buffers.back();

    cudaError_t error = cudaMemcpyAsync(imageBuffer, image->GetData(), N, cudaMemcpyHostToDevice, stream1);
    if(error != cudaSuccess)
    {
        throw std::runtime_error("Failed to copy image memory: " + std::string(cudaGetErrorString(error)));
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
        throw std::runtime_error("Failed to run phase algorithm: " + std::string(cudaGetErrorString(error)));
    }

    return true;
}

void GPU::calcPhaseImage(std::shared_ptr<float> phaseMap, cudaSurfaceObject_t phaseImage)
{
    create_phaseImage<<<blockPerGrid,threadPerBlock, 0, stream1>>>(phaseMap.get(), phaseImage, N, width);
    
    cudaError_t error = cudaStreamSynchronize(stream1);
    if(error != cudaSuccess)
    {
        throw std::runtime_error("Failed to generate phase image: " + std::string(cudaGetErrorString(error)));
    }
}