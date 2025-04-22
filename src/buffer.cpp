#include "buffer.h"
#include "cuda_runtime.h"

#include <iostream>

std::unordered_map<size_t, std::queue<Buffer::Impl*>> Buffer::pool;
std::mutex Buffer::poolMutex;


Buffer::Buffer(size_t width, size_t height, size_t pixelBits, size_t alignBytes)
{
    Impl* implRaw = nullptr;
    size_t unaligned = (width * height * pixelBits + 8 - 1) / 8;
    size_t byteSize = (unaligned + alignBytes - 1) / alignBytes * alignBytes;

    {
        std::lock_guard guard(poolMutex);
        auto it = pool.find(byteSize);
        if(it != pool.end() && it->second.size() > 0)
        {
            implRaw = it->second.front();
            it->second.pop();
        }
    }

    if(!implRaw)
    {
        cudaError_t error = cudaMallocHost(&implRaw, sizeof(Impl) + byteSize);
        if(error != cudaSuccess)
        {
            std::cout << "Failed to allocate host buffer of size: " << byteSize <<": " << cudaGetErrorString(error) << std::endl;
            return;
        }

        std::cout << "Allocated cuda host buffer of size: " << byteSize << std::endl;
    }

    impl = std::shared_ptr<Impl>(new (implRaw) Impl{
        width,
        height,
        pixelBits,
        byteSize,
    }, 
    [&](Impl* implRaw){
        std::lock_guard guard(poolMutex);
        pool[implRaw->byteSize].push(implRaw);
    });
}

Buffer::~Buffer()
{

}

void* Buffer::get()
{
    return reinterpret_cast<uint8_t*>(impl.get()) + sizeof(Impl);
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