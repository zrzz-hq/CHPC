#include "GPU.h"

GPU::GPU(size_t width, size_t height):
    config(std::make_shared<GPU::Config>())
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

    phaseImagePool = std::make_shared<BufferPool>(BufferDesc{width, height, sizeof(uint8_t) * 3 * 8}, 30);
    phaseMapPool = std::make_shared<BufferPool>(BufferDesc{width, height, sizeof(float) * 8}, 30);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Maximum threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Maximum threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Maximum grid size: (" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << ")" << std::endl;
};

GPU::~GPU()
{
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

Buffer GPU::run(Spinnaker::ImagePtr image)
{
    float* newImageDev = buffers.back();
    buffers.pop_back();
    buffers.push_front(newImageDev);

    cudaMemcpy(imageBuffer, image->GetData(), N * sizeof(uint8_t), cudaMemcpyHostToDevice);

    convert_type<<<blockPerGrid,threadPerBlock, 0, stream2>>>(imageBuffer, newImageDev, N);

    if (eleCount < (config->algorithmIndex == 0 ? 5 : 4))
    {
        eleCount++;
        return Buffer();
    }
    else
    {
        if(config->bufferMode)
            eleCount = 0;
    }

    Buffer phaseMap(phaseMapPool);

    if(!phaseMap.isValid())
        return Buffer();

    float* phaseMapBuffer = reinterpret_cast<float*>(phaseMap.get());
 
    switch (config->algorithmIndex)
    {
    case 0:
        novak<<<blockPerGrid,threadPerBlock, 0, stream2>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        buffers[4],
                                        phaseMapBuffer,
                                        N);
        break;
    case 1:
        four_point<<<blockPerGrid,threadPerBlock, 0, stream2>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        phaseMapBuffer,
                                        N);
        break;
    case 2:
        carres<<<blockPerGrid,threadPerBlock, 0, stream2>>>(buffers[0],
                                        buffers[1],
                                        buffers[2],
                                        buffers[3],
                                        phaseMapBuffer,
                                        N);
    default:
        break;
    }

    cudaStreamSynchronize(stream2);
    return phaseMap;
}

Buffer GPU::generateImage(Buffer phaseMap)
{   
    Buffer phaseImage(phaseImagePool);

    generate_image<<<blockPerGrid,threadPerBlock, 0, stream2>>>(
        reinterpret_cast<float*>(phaseMap.get()), 
        reinterpret_cast<uint8_t*>(phaseImage.get()), N);
    cudaStreamSynchronize(stream2);

    return phaseImage;
}