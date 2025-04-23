#include "FLIRCamera.h"
#include "GPU.h"
#include "ui.h"

#include <memory>
#include <chrono>
#include <queue>
#include <fstream>
#include <future>
#include <pthread.h>
#include <mutex>

#include <GL/gl.h>
#include <GL/glx.h>
#include <GLFW/glfw3.h>

#include <opencv2/opencv.hpp>

#define WIDTH 720
#define HEIGHT 540
#define FRAMERATE 60
#define EXPOSURETIME -1
#define GAIN 0

template<typename T>
class Queue
{
    public:   
    Queue(){}
    ~Queue(){}

    void push(T item)
    {
        std::lock_guard guard(this->mutex);
        this->queue.push(item);
        this->cond.notify_one();
    }

    T pop()
    {
        std::unique_lock<std::mutex> lock(this->mutex);
        this->cond.wait(lock, [&](){return queue.size() != 0;});

        T item = this->queue.front();
        this->queue.pop();
        return item;
    }

    T try_pop(std::chrono::milliseconds duration)
    {
        std::unique_lock<std::mutex> lock(this->mutex);
        bool success = this->cond.wait_for(lock, duration, [&](){return this->queue.size() != 0;});

        if(!success)
            return T{};
        
        T item = this->queue.front();
        this->queue.pop();
        return item;
    }

    private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cond;
};

Queue<ImagePtr> imageQueue1;
Queue<ImagePtr> imageQueue2;
Queue<Buffer> phaseMapQueue;
Queue<Buffer> phaseImageQueue;

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
    
    auto last = std::chrono::system_clock::now();
    cam->start();

    while(1)
    {
        ImagePtr image = cam->read();

        if(image.IsValid())
        {
            imageQueue1.push(image);
        }

        pthread_testcancel();

        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        last = now;
        // std::cout << "Duration of camera: " << duration << '\r';
    }

    cam->stop();

    pthread_cleanup_pop(1);
    return 0;
}

void* gpuThreadFunc(void* arg)
{
    GPU* gpu = reinterpret_cast<GPU*>(arg);

    auto last = std::chrono::system_clock::now();

    while(1)
    {
        // ImagePtr image = imageQueue1.try_pop(std::chrono::milliseconds(10));
        ImagePtr image = imageQueue1.pop();

        if(image.IsValid())
        {
            Buffer phaseMap = gpu->run(image);
            
            if(phaseMap.isValid())
            {
                // auto start = std::chrono::system_clock::now();
                Buffer phaseImage = gpu->generateImage(phaseMap);
                // auto end = std::chrono::system_clock::now();
                // std::cout << "Cpu algorithm takes time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

                phaseMapQueue.push(phaseMap);
                phaseImageQueue.push(phaseImage);
            }

            imageQueue2.push(image);
        }

        pthread_testcancel();

        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        last = now;
        std::cout << "Duration of gpu: " << duration << std::endl;
    }
}

void writeMatToCSV(Buffer phaseMap, const std::string& fileName)
{
    cv::Mat phaseMat(phaseMap.getHeight(), phaseMap.getWidth(), CV_32F, phaseMap.get());
    // if(cudaMemcpy(phaseMat.data, phaseMap.get(), width*height*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
    // {
    //     std::cout << "Failed to copy data" << std::endl;
    // }
    // else
    // {
        std::ofstream file(fileName);

        if (!file.is_open()){
            std::cout << "Error: Could not write phase map to csv file\n";
            return;
        }

        for (int i = 0; i < phaseMat.rows; i++){
            for (int j = 0; j < phaseMat.cols; j++){
                file << phaseMat.at<float>(i, j);
                if (j < phaseMat.cols-1)
                    file << ',';
            }
            file << '\n';
        }
        file.close();
        std::cout << "Phase map saved to " << fileName << std::endl; 
    // }
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
    GPU gpu(width, height);

    pthread_t cameraThread;
    if(pthread_create(&cameraThread, NULL, cameraThreadFunc, &cam) == -1)
    {
        std::cout << "Failed to create camera thread" << std::endl;
        return 1;
    }

    pthread_t gpuThread;
    if(pthread_create(&gpuThread, NULL, gpuThreadFunc, &gpu) == -1)
    {
        std::cout << "Failed to create gpu thread" << std::endl;
        return 1;
    }

    MainWindow mainWindow(cameraConfig, gpu.getConfig());
    while(mainWindow.ok())
    {
        ImagePtr image = imageQueue2.try_pop(std::chrono::milliseconds(0));

        if(image.IsValid())
        {
            mainWindow.updateFrame(image->GetData());
        }

        Buffer phaseImage = phaseImageQueue.try_pop(std::chrono::milliseconds(0));

        if(phaseImage.isValid())
        {
            mainWindow.updatePhase(phaseImage.get());
        }

        Buffer phaseMap = phaseMapQueue.try_pop(std::chrono::milliseconds(0));
        
        if (mainWindow.nSavedPhaseMap > 0 && phaseMap.isValid())
        {
            std::string fileName = mainWindow.textBuffer;
            std::string filePath =  "/home/nvidia/images/";
            writeMatToCSV(phaseMap, filePath + fileName + std::to_string(mainWindow.numSuccessiveImages - mainWindow.nSavedPhaseMap) + ".csv");

            mainWindow.nSavedPhaseMap--;
        }
        
        mainWindow.spinOnce();
    }

    pthread_cancel(cameraThread);
    pthread_join(cameraThread, NULL);
    pthread_cancel(gpuThread);
    pthread_join(gpuThread, NULL);

    cam.close();
    return 0;
}
