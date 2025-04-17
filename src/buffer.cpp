#include "buffer.h"
#include "cuda_runtime.h"

#include <iostream>

struct Buffer::Impl
{
    size_t width;
    size_t height;
    size_t pixelBits;
    size_t byteSize;
    void* data;
};

Buffer::Buffer(size_t width, size_t height, size_t pixelBits, size_t alignBytes):
    impl(new Buffer::Impl, [](Buffer::Impl* impl){
        cudaFree(impl->data);
        delete impl;
    })
{
    impl->width = width;
    impl->height = height;
    impl->pixelBits = pixelBits;
    size_t unaligned = (width * height * pixelBits + 8 - 1) / 8;
    impl->byteSize = (unaligned + alignBytes - 1) / alignBytes * alignBytes;
    cudaMallocHost(&impl->data, impl->byteSize);
}

void* Buffer::get()
{
    return impl->data;
}

size_t Buffer::getByteSize()
{
    return impl->byteSize;
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