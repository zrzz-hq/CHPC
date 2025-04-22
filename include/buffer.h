#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <atomic>

#include <boost/lockfree/queue.hpp>


struct BufferDesc
{
    size_t width;
    size_t height;
    size_t pixelBits;
    size_t alignBytes = 1;
};

class Buffer;

class BufferPool
{
    public:
    friend class Buffer;
    BufferPool(const BufferDesc& desc, size_t nBuffers);
    ~BufferPool();
    
    private:
    size_t width;
    size_t height;
    size_t pixelBits;
    size_t byteSize;
    void* buffers;
    boost::lockfree::queue<size_t> queue;
};

class Buffer
{
    public:
    Buffer(){}
    Buffer(std::shared_ptr<BufferPool> pool);
    
    Buffer(const Buffer& buffer)
    {
        this->impl = buffer.impl;
    }
    Buffer(Buffer&& buffer)
    {
        this->impl = std::move(buffer.impl);
    }

    Buffer& operator=(const Buffer& buffer)
    {
        this->impl = buffer.impl;
        return *this;
    }
    Buffer& operator=(Buffer&& buffer) 
    {
        this->impl = std::move(buffer.impl); 
        return *this;
    }
    ~Buffer();

    bool isVaild() {return impl != nullptr;}
    void* get(); 
    size_t getByteSize();
    size_t getWidth();
    size_t getHeight();

    private:

    struct Impl
    {
        std::shared_ptr<BufferPool> pool;
        size_t id;
    };

    std::shared_ptr<Impl> impl;
};
