#include "FLIRCamera.h"
#include "GPU.h"
#include "ui.h"

#include <memory>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>

#include <pthread.h>

#include <boost/filesystem.hpp>
#include <boost/asio.hpp>
#include <boost/optional.hpp>

#include <GL/gl.h>
#include <GL/glx.h>
#include <GLFW/glfw3.h>

#include <opencv2/opencv.hpp>

#include "cnpy.h"

#define WIDTH 720
#define HEIGHT 540
#define FRAMERATE 60
#define EXPOSURETIME -1
#define GAIN 0

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

    private:
    std::deque<T> queue;
    std::mutex mutex;
    std::condition_variable cond;
};

DataQueue<Spinnaker::ImagePtr> queue1;
DataQueue<std::tuple<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>, std::shared_ptr<float>>> queue2;

void cameraThreadCleanUp(void* arg)
{
    FLIRCamera* cam = reinterpret_cast<FLIRCamera*>(arg);

    cam->stop();

    std::cout << "camera thread exited" << std::endl;
}

void* cameraThreadFunc(void* arg)
{
    pthread_cleanup_push(cameraThreadCleanUp, arg);
    FLIRCamera* cam = reinterpret_cast<FLIRCamera*>(arg);

    cam->start();
    while(1)
    {
        Spinnaker::ImagePtr imagePtr = cam->read();
        
        if(imagePtr.IsValid())
        {
            queue1.push(imagePtr);
        }
    }

    cam->stop();

    pthread_cleanup_pop(1);
    return 0;
}

void* gpuThreadFunc(void* arg)
{
    GPU* gpu = reinterpret_cast<GPU*>(arg);
    while(1)
    {
        auto imageOpt = queue1.tryPop(std::chrono::milliseconds(100));
        if(imageOpt)
        {
            Spinnaker::ImagePtr image = imageOpt.get();
            auto [phaseMap, phaseImage] = gpu->run(image);

            queue2.push({image, phaseImage, phaseMap});
        }
    }
}

int main(int argc, char* argv[])
{
    FLIRCamera cam;
    //Start up Window

    std::shared_ptr<FLIRCamera::Config> cameraConfig;
    std::vector<std::string> deviceIds = cam.enumCamera();
    if(deviceIds.size() == 0)
    {
        ErrorWindow errorWin;
        errorWin.spin();
        return -1;
    }
    else
    {
        cameraConfig = cam.open(0);
        StartupWindow startupWin(cameraConfig);
        if(startupWin.spin() == -1)
        {
            cam.close();
            return -1;
        }
    }

    int width = cameraConfig->width->GetValue();
    int height = cameraConfig->height->GetValue();
    GPU gpu(width, height, 40);

    pthread_t cameraThread;
    if(pthread_create(&cameraThread, NULL, cameraThreadFunc, &cam) == -1)
    {
        std::cout << "Failed to create camera thread" << std::endl;
        cam.close();
        return -1;
    }

    pthread_t gpuThread;
    if(pthread_create(&gpuThread, NULL, gpuThreadFunc, &gpu) == -1)
    {
        std::cout << "Failed to create gpu thread" << std::endl;
        cam.close();
        pthread_cancel(cameraThread);
        pthread_join(cameraThread, NULL);
        return -1;
    }

    MainWindow mainWindow(cameraConfig, gpu.getConfig());

    boost::filesystem::path imageFolder("images");
    if(!boost::filesystem::exists(imageFolder))
    {
        boost::filesystem::create_directory(imageFolder);
    }

    boost::asio::io_service service;
    boost::asio::io_service::work work(service);
    std::thread workThread([&]{service.run();});

    while(mainWindow.ok())
    {
        std::tuple<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>, std::shared_ptr<float>> tuple;

        auto tupleOpt = queue2.tryPop(std::chrono::milliseconds(0));
        if(tupleOpt)
        {
            const auto& [image, phaseImage, phaseMap] = tupleOpt.get();
            if(image.IsValid())
            {
                mainWindow.updateFrame(image->GetData());
            }
            if(phaseImage != nullptr)
            {
                mainWindow.updatePhase(phaseImage.get());
            }
        
            if (mainWindow.nSavedPhaseMap > 0 && image.IsValid())
            {
                //Save Phase Maps
                if(mainWindow.output && phaseMap)
                {
                    boost::filesystem::path path = mainWindow.folder / (mainWindow.filename + 
                    std::to_string(mainWindow.numSuccessiveImages - mainWindow.nSavedPhaseMap));
                    path.replace_extension("npy");
                    
                    service.post([=]{
                        std::vector<float> phaseMat(width*height, 0);
                        cudaMemcpy(phaseMat.data(), phaseMap.get(), width * height * sizeof(float), cudaMemcpyDeviceToHost);
                        cnpy::npy_save(
                            path.string(), 
                            phaseMat.data(),
                            {static_cast<size_t>(width), static_cast<size_t>(height)}
                        );
                    });
                }

                if(mainWindow.input)
                {
                    boost::filesystem::path path = mainWindow.folder/ (mainWindow.filename + 
                        std::to_string(mainWindow.numSuccessiveImages - mainWindow.nSavedPhaseMap));
                    
                    path.replace_extension("png");

                    service.post([=]{
                        cv::Mat imageMat(height, width, CV_8UC1, image->GetData());
                        cv::imwrite(path.string(), imageMat);
                    });   
                }

                mainWindow.nSavedPhaseMap--;
            }

        }

        mainWindow.spinOnce();
    }

    pthread_cancel(cameraThread);
    pthread_join(cameraThread, NULL);

    pthread_cancel(gpuThread);
    pthread_join(gpuThread, NULL);

    cam.close();

    work.~work();
    service.stop();
    workThread.join();

    return 0;
}
