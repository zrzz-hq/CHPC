#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>

class Buffer
{
    public:

    Buffer(){};
    Buffer(size_t width, size_t height, size_t pixelBits, size_t alignBytes=1);
    Buffer(const Buffer& buffer){this->impl = buffer.impl;}
    Buffer(Buffer&& buffer){this->impl=std::move(buffer.impl);}
    Buffer& operator=(const Buffer& buffer) {this->impl = buffer.impl; return *this;}
    Buffer& operator=(Buffer&& buffer) {this->impl = std::move(buffer.impl); return *this;}
    ~Buffer() = default;

    bool isVaild() {return impl != nullptr;}
    void* get(); 
    size_t getByteSize();
    size_t getWidth();
    size_t getHeight();

    private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    static std::unordered_map<size_t, std::queue<void*>> pool;
    static std::mutex poolMutex;
};
