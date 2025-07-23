#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class DataQueue
{
    public:
    DataQueue()
    {

    }

    ~DataQueue()
    {

    }

    T pop()
    {
        std::unique_lock lock(this->mutex);
        this->cond.wait(lock, [this]{return queue.size() > 0;});
        T data = std::move(this->queue.back());
        this->queue.clear();
        return data;
    }

    boost::optional<T> tryPop(std::chrono::milliseconds timeout)
    {
        std::unique_lock lock(this->mutex);
        if(!this->cond.wait_for(lock, timeout, [this]{return queue.size() > 0;}))
            return boost::none;
        
        T data = std::move(this->queue.back());
        this->queue.clear();
        return data;
    }

    void push(T&& data)
    {
        std::unique_lock lock(this->mutex);
        this->queue.push_back(std::move(data));
        this->cond.notify_all();
    }

    void push(const T& data)
    {
        std::unique_lock lock(this->mutex);
        this->queue.push_back(data);
        this->cond.notify_all();
    }

    void clear()
    {
        std::unique_lock lock(this->mutex);
        this->queue.clear();
    }

    private:
    std::deque<T> queue;
    std::mutex mutex;
    std::condition_variable cond;
};