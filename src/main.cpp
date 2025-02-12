#include "FLIRCamera.h"
#include "GPU.h"
#include <memory>
#include <chrono>
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <pthread.h>

#define WIDTH 800
#define HEIGHT 600
#define FRAMERATE 60.0


std::queue<Spinnaker::ImagePtr> imageQueue1;
pthread_mutex_t imageQueue1Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueue1Cond = PTHREAD_COND_INITIALIZER;

std::queue<std::pair<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>>> imageQueue2;
pthread_mutex_t imageQueue2Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueue2Cond = PTHREAD_COND_INITIALIZER;

// TODO: add a phase buffer pool in the GPU class.
// std::queue<Spinnaker::ImagePtr> phaseQueue;
// pthread_mutex_t phaseQueueMutex = PTHREAD_MUTEX_INITIALIZER;
// pthread_cond_t phaseQueueCond = PTHREAD_COND_INITIALIZER;

void gpuThreadCleanUp(void* arg)
{
    pthread_mutex_lock(&imageQueue2Mutex);

    while(imageQueue2.size() > 0)
        imageQueue2.pop();
    
    pthread_mutex_unlock(&imageQueue2Mutex);

    pthread_mutex_unlock(&imageQueue1Mutex);

    std::cout << "gpu thread exited" << std::endl;

}

void* gpuThreadFunc(void* arg)
{
    pthread_cleanup_push(gpuThreadCleanUp, NULL);

    GPU gpu(WIDTH,HEIGHT,50);
    gpu.getCudaVersion();
    while(1)
    {
        Spinnaker::ImagePtr imagePtr;
        pthread_mutex_lock(&imageQueue1Mutex);

        while(imageQueue1.size() == 0)
            pthread_cond_wait(&imageQueue1Cond, &imageQueue1Mutex);

        imagePtr = imageQueue1.front();
        imageQueue1.pop();

        pthread_mutex_unlock(&imageQueue1Mutex);

        std::shared_ptr<uint8_t> cosine = gpu.runNovak(imagePtr);
        // TODO: write the following code in another thread
        
        pthread_mutex_lock(&imageQueue2Mutex);

        imageQueue2.emplace(std::make_pair(imagePtr,cosine));

        pthread_cond_signal(&imageQueue2Cond);
        pthread_mutex_unlock(&imageQueue2Mutex);

        pthread_testcancel();
    }
    pthread_cleanup_pop(1);
    return 0;
}

void cameraThreadCleanUp(void* arg)
{

    pthread_mutex_lock(&imageQueue1Mutex);
    
    while(imageQueue1.size() > 0)
        imageQueue1.pop();
    
    pthread_mutex_unlock(&imageQueue1Mutex);

    std::cout << "camera thread exited" << std::endl;
}

void* cameraThreadFunc(void* arg)
{
    pthread_cleanup_push(cameraThreadCleanUp, NULL);

    FLIRCamera cam;
    cam.open(0);
    cam.setResolution(WIDTH,HEIGHT);
    cam.setFPS(FRAMERATE);

    cam.start();
    while(1)
    {
        Spinnaker::ImagePtr imagePtr = cam.read();

        pthread_mutex_lock(&imageQueue1Mutex);

        imageQueue1.push(imagePtr);

        pthread_cond_signal(&imageQueue1Cond);
        pthread_mutex_unlock(&imageQueue1Mutex);

        pthread_testcancel();
        
    }

    cam.stop();
    cam.close();

    pthread_cleanup_pop(1);
    return 0;
}

int main()
{
    pthread_t gpuThread;
    if(pthread_create(&gpuThread, NULL, gpuThreadFunc, NULL) == -1)
    {
        std::cout << "Failed to create GPU thread" << std::endl;
        return 1;
    }

    pthread_t cameraThread;
    if(pthread_create(&cameraThread, NULL, cameraThreadFunc, NULL) == -1)
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

        imagePair = std::move(imageQueue2.front());
        imageQueue2.pop();

        pthread_mutex_unlock(&imageQueue2Mutex);
       
        cv::Mat image(imagePair.first->GetHeight(),imagePair.first->GetWidth(),CV_8UC1, imagePair.first->GetData());

        auto now = std::chrono::system_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        last = now;

        cv::putText(image, std::to_string(1000.0/duration), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 2);
        cv::imshow("frame",image);

        if(imagePair.second != nullptr)
        {
            cv::Mat phaseImage(HEIGHT,WIDTH,CV_8UC1,imagePair.second.get());
            cv::imshow("phase", phaseImage);
        }

        imagePair.first->Release();
        imagePair.second.reset();

        if(cv::waitKey(1) == 'q')
            break;
    }

    cv::destroyAllWindows();

    pthread_cancel(gpuThread);
    pthread_join(gpuThread, NULL);
    
    pthread_cancel(cameraThread);
    pthread_join(cameraThread, NULL);

    return 0;
}
