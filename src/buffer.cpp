#include "buffer.h"
#include "cuda_runtime.h"

#include <iostream>

std::unordered_map<size_t, std::queue<void*>> Buffer::pool;
std::mutex Buffer::poolMutex;

struct Buffer::Impl
{
    size_t width;
    size_t height;
    size_t pixelBits;
    size_t byteSize;
    void* data;
};

Buffer::Buffer(size_t width, size_t height, size_t pixelBits, size_t alignBytes)
{
    
    size_t unaligned = (width * height * pixelBits + 8 - 1) / 8;
    size_t byteSize = (unaligned + alignBytes - 1) / alignBytes * alignBytes;
    void* data = nullptr;

    {
        std::lock_guard guard(poolMutex);
        auto it = pool.find(byteSize);
        if(it != pool.end() && it->second.size() > 0)
        {
            data = it->second.front();
            it->second.pop();
        }
    }

    cudaError_t error = cudaMallocHost(&data, byteSize);
    if(error == cudaSuccess)
    {
        std::cout << "Allocated cuda host buffer of size: " << byteSize << std::endl;
    }
    else
    {
        std::cout << "Failed to allocate host buffer of size: " << byteSize <<": " << cudaGetErrorString(error) << std::endl;
    }

    if(data)
    {
        impl = std::shared_ptr<Buffer::Impl>(new Buffer::Impl, [&](Buffer::Impl* impl){
            {
                std::lock_guard guard(poolMutex);
                pool[impl->byteSize].push(impl->data);
            }

            std::cout << "buffer freed" << std::endl;
            delete impl;
        });

        impl->width = width;
        impl->height = height;
        impl->pixelBits = pixelBits;
        impl->byteSize = byteSize;
        impl->data = data;
    }
}

void* Buffer::get()
{
    return impl->data;
}

size_t Buffer::getByteSize()
{
    return impl->byteSize;
}

size_t Buffer::getWidth()
{
    return impl->width;
}

size_t Buffer::getHeight()
{
    return impl->height;
}

// for(int i=0; i<nPhaseBuffers; i++)
// {
//     uint8_t* cosineBuffer;
//     error = cudaMallocHost(&cosineBuffer, N*sizeof(uint8_t)*3);
//     if(error != cudaSuccess)
//     {
//         throw std::runtime_error("Failed to allocate phase buffer: " + std::string(cudaGetErrorString(error)));
//     }
//     cosineBuffers.push(cosineBuffer);
// }