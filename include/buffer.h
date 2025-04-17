#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>

class Buffer
{
    public:
    Buffer() = delete;
    Buffer(size_t width, size_t height, size_t pixelBits, size_t alignBytes=1);
    ~Buffer() = default;

    void* get();
    size_t getByteSize();

    private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    static std::unordered_map<>
};
