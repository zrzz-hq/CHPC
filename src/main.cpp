#include "FLIRCamera.h"
#include "GPU.h"
#include <memory>
#include <chrono>
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <pthread.h>

#define WIDTH 1200
#define HEIGHT 800
#define FRAMERATE 30.0


std::deque<Spinnaker::ImagePtr> imageQueue1;
pthread_mutex_t imageQueue1Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueue1Cond = PTHREAD_COND_INITIALIZER;


std::deque<std::pair<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>>> imageQueue2;
pthread_mutex_t imageQueue2Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueue2Cond = PTHREAD_COND_INITIALIZER;

void gpuThreadCleanUp(void* arg)
{
    std::cout << "gpu thread exited" << std::endl;
}

void* gpuThreadFunc(void* arg)
{
    pthread_cleanup_push(gpuThreadCleanUp, NULL);
    GPU* gpu = reinterpret_cast<GPU*>(arg);

    gpu->getCudaVersion();
    while(1)
    {
        Spinnaker::ImagePtr imagePtr;
        pthread_mutex_lock(&imageQueue1Mutex);

        while(imageQueue1.size() == 0)
            pthread_cond_wait(&imageQueue1Cond, &imageQueue1Mutex);

        imagePtr = imageQueue1.back();
        imageQueue1.clear();

        pthread_mutex_unlock(&imageQueue1Mutex);

        std::shared_ptr<uint8_t> cosine = gpu->runNovak(imagePtr);
        // TODO: write the following code in another thread
        
        pthread_mutex_lock(&imageQueue2Mutex);

        imageQueue2.emplace_back(std::make_pair(imagePtr,cosine));

        pthread_cond_signal(&imageQueue2Cond);
        pthread_mutex_unlock(&imageQueue2Mutex);

        pthread_testcancel();
    }
    pthread_cleanup_pop(1);
    return 0;
}

void cameraThreadCleanUp(void* arg)
{
    FLIRCamera* cam = reinterpret_cast<FLIRCamera*>(arg);

    cam->stop();
    cam->close();

    std::cout << "camera thread exited" << std::endl;
}

void* cameraThreadFunc(void* arg)
{
    pthread_cleanup_push(cameraThreadCleanUp, arg);
    FLIRCamera* cam = reinterpret_cast<FLIRCamera*>(arg);

    cam->open(0);
    cam->setResolution(WIDTH,HEIGHT);
    cam->setFPS(FRAMERATE);

    cam->start();
    while(1)
    {
        Spinnaker::ImagePtr imagePtr = cam->read();

        pthread_mutex_lock(&imageQueue1Mutex);

        imageQueue1.push_back(imagePtr);

        pthread_cond_signal(&imageQueue1Cond);
        pthread_mutex_unlock(&imageQueue1Mutex);

        pthread_testcancel();
        
    }

    cam->stop();
    cam->close();

    pthread_cleanup_pop(1);
    return 0;
}

int main()
{
    GPU gpu(WIDTH,HEIGHT,50);
    FLIRCamera cam;

    pthread_t gpuThread;
    if(pthread_create(&gpuThread, NULL, gpuThreadFunc, &gpu) == -1)
    {
        std::cout << "Failed to create GPU thread" << std::endl;
        return 1;
    }

    pthread_t cameraThread;
    if(pthread_create(&cameraThread, NULL, cameraThreadFunc, &cam) == -1)
    {
        std::cout << "Failed to create camera thread" << std::endl;
        return 1;
    }

    // std::cout << "Opencv version " << cv::getVersionMajor() << std::endl;
    
    auto last = std::chrono::system_clock::now();
    while(1)
    {
        std::pair<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>> imagePair;

        pthread_mutex_lock(&imageQueue2Mutex);
        while(imageQueue2.size() == 0)
            pthread_cond_wait(&imageQueue2Cond, &imageQueue2Mutex);

        imagePair = std::move(imageQueue2.back());
        imageQueue2.clear();

        pthread_mutex_unlock(&imageQueue2Mutex);
       
        cv::Mat image(imagePair.first->GetHeight(),imagePair.first->GetWidth(),CV_8UC1, imagePair.first->GetData());

        auto now = std::chrono::system_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        last = now;

        cv::putText(image, std::to_string(1000.0/duration), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 2);
        cv::imshow("frame",image);

        if(imagePair.second != nullptr)
        {
            cv::Mat phaseImage(HEIGHT,WIDTH,CV_8UC1,cv::Scalar(0));
            size_t size = WIDTH * HEIGHT * sizeof(uint8_t);
            void* dstPtr = phaseImage.data;
            void* srcPtr = imagePair.second.get();
            cudaError_t error = cudaMemcpy(dstPtr, srcPtr, size, cudaMemcpyDeviceToHost);
            if(error != cudaSuccess)
            {
                std::cout << "Failed to copy memory: " << cudaGetErrorString(error) << std::endl;
            }
            else
            {
                cv::imshow("phase", phaseImage);
            }
        }

        if(cv::waitKey(1) == 'q')
            break;
    }

    cv::destroyAllWindows();

    pthread_cancel(gpuThread);
    pthread_cancel(cameraThread);

    pthread_join(gpuThread, NULL);
    pthread_join(cameraThread, NULL);

    imageQueue1.clear();
    imageQueue2.clear();

    return 0;
}
