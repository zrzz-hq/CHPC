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
#define FRAMERATE 90.0


std::queue<Spinnaker::ImagePtr> imageQueue;
pthread_mutex_t imageQueueMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueueCond = PTHREAD_COND_INITIALIZER;

std::queue<Spinnaker::ImagePtr> phaseQueue;
pthread_mutex_t phaseQueueMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t phaseQueueCond = PTHREAD_COND_INITIALIZER;

void* gpuThreadFunc(void* arg)
{
    GPU gpu(WIDTH,HEIGHT);
    gpu.getCudaVersion();
    while(1)
    {
        Spinnaker::ImagePtr imagePtr;
        pthread_mutex_lock(&imageQueueMutex);

        while(imageQueue.size() == 0)
            pthread_cond_wait(&imageQueueCond, &imageQueueMutex);

        imagePtr = imageQueue.front();
        imageQueue.pop();

        pthread_mutex_unlock(&imageQueueMutex);

        float* phase = gpu.runNovak(imagePtr);
        if(phase != nullptr)
        {
            cv::Mat phaseImage(HEIGHT,WIDTH,CV_32FC1,phase);
            cv::imshow("phase", phaseImage);
        }

        pthread_testcancel();
    }
}

int main()
{
    pthread_t gpuThread;
    if(pthread_create(&gpuThread, NULL, gpuThreadFunc, NULL) == -1)
    {
        std::cout << "Failed to create GPU thread" << std::endl;
        return 1;
    }

    FLIRCamera cam;
    cam.open(0);
    cam.setResolution(WIDTH,HEIGHT);
    cam.setFPS(FRAMERATE);

    std::cout << "Opencv version " << cv::getVersionMajor() << std::endl;
    
    cam.start();
    auto last = std::chrono::system_clock::now();
    while(1)
    {
       
        Spinnaker::ImagePtr imagePtr = cam.read();

        cv::Mat image(imagePtr->GetHeight(),imagePtr->GetWidth(),CV_8UC1, imagePtr->GetData());

        auto now = std::chrono::system_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        last = now;

        cv::putText(image, std::to_string(1000.0/duration), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 2);
        cv::imshow("frame",image);

        pthread_mutex_lock(&imageQueueMutex);
        imageQueue.push(imagePtr);
        pthread_cond_signal(&imageQueueCond);
        pthread_mutex_unlock(&imageQueueMutex);

        if(cv::waitKey(1) == 'q')
            break;
    }

    cv::destroyAllWindows();
    
    cam.stop();
    cam.close();

    pthread_cancel(gpuThread);
    pthread_join(gpuThread, NULL);

    return 0;



}
