#include "buffer.h"
#include "cuda_runtime.h"

#include <iostream>


BufferPool::BufferPool(const BufferDesc& desc, size_t nBuffers):
    width(desc.width),
    height(desc.height),
    pixelBits(desc.pixelBits),
    queue(nBuffers)
{
    size_t unaligned = (width * height * pixelBits + 8 - 1) / 8;
    byteSize = (unaligned + desc.alignBytes - 1) / desc.alignBytes * desc.alignBytes;

    cudaError_t error = cudaMallocHost(&buffers, nBuffers * byteSize);
    if(error != cudaSuccess)
    {
        std::cout << "Failed to allocate " << nBuffers << " buffer of size " << byteSize << std::endl;
    }

    for(size_t i=0;i<nBuffers;i++)
    {
        queue.push(i);
    }
}

BufferPool::~BufferPool()
{
    cudaFree(buffers);
}

Buffer::Buffer(std::shared_ptr<BufferPool> pool)
{
    size_t id;
    if(!pool->queue.pop(id))
        return;

    impl = std::shared_ptr<Impl>(new Impl{pool, id}, [](Impl* impl){
        impl->pool->queue.push(impl->id);
    });
}

Buffer::~Buffer()
{
    
}

void* Buffer::get()
{
    return reinterpret_cast<uint8_t*>(impl->pool->buffers) + impl->pool->byteSize * impl->id;
}

size_t Buffer::getByteSize()
{
    return impl->pool->byteSize;
}

size_t Buffer::getWidth()
{
    return impl->pool->width;
}

size_t Buffer::getHeight()
{
    return impl->pool->height;
}