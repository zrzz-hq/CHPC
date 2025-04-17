#include "FLIRCamera.h"
#include "GPU.h"
#include "ui.h"

#include <memory>
#include <chrono>
#include <queue>
#include <fstream>
#include <pthread.h>

#include <GL/gl.h>
#include <GL/glx.h>
#include <GLFW/glfw3.h>

#include <opencv2/opencv.hpp>

#define WIDTH 720
#define HEIGHT 540
#define FRAMERATE 60
#define EXPOSURETIME -1
#define GAIN 0

std::deque<Spinnaker::ImagePtr> imageQueue1;
pthread_mutex_t imageQueue1Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueue1Cond = PTHREAD_COND_INITIALIZER;


std::deque<std::tuple<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>, std::shared_ptr<float>>> imageQueue2;
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
    auto last = std::chrono::system_clock::now();
    while(1)
    {
        Spinnaker::ImagePtr imagePtr;
        pthread_mutex_lock(&imageQueue1Mutex);

        while(imageQueue1.size() == 0)
            pthread_cond_wait(&imageQueue1Cond, &imageQueue1Mutex);

        imagePtr = imageQueue1.back();
        imageQueue1.clear();

        pthread_mutex_unlock(&imageQueue1Mutex);

        std::pair<std::shared_ptr<uint8_t>, std::shared_ptr<float>> pair = gpu->runNovak(imagePtr);
        
        pthread_mutex_lock(&imageQueue2Mutex);

        imageQueue2.emplace_back(imagePtr, pair.first, pair.second);

        pthread_cond_signal(&imageQueue2Cond);
        pthread_mutex_unlock(&imageQueue2Mutex);

        pthread_testcancel();

        auto now = std::chrono::system_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        // std::cout << "Frame rate: " << 1000.0 / duration << std::endl;
        last = now;
    }
    pthread_cleanup_pop(1);
    return 0;
}

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

        pthread_mutex_lock(&imageQueue1Mutex);

        imageQueue1.push_back(imagePtr);

        pthread_cond_signal(&imageQueue1Cond);
        pthread_mutex_unlock(&imageQueue1Mutex);

        pthread_testcancel();
        
    }

    cam->stop();

    pthread_cleanup_pop(1);
    return 0;
}

void writeMatToCSV(const cv::Mat mat, const std::string& fileName){
    std::ofstream file(fileName);

    if (!file.is_open()){
        std::cout << "Error: Could not write phase map to csv file\n";
        return;
    }

    for (int i = 0; i < mat.rows; i++){
        for (int j = 0; j < mat.cols; j++){
            file << mat.at<float>(i, j);
            if (j < mat.cols-1)
                file << ',';
        }
        file << '\n';
    }
    file.close();
    std::cout << "Phase map saved to " << fileName << std::endl; 

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
    MainWindow mainWindow(cameraConfig, gpu.getConfig());
    while(mainWindow.ok())
    {
        std::tuple<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>, std::shared_ptr<float>> tuple;

        pthread_mutex_lock(&imageQueue2Mutex);
        while(imageQueue2.size() == 0)
            pthread_cond_wait(&imageQueue2Cond, &imageQueue2Mutex);

        tuple = std::move(imageQueue2.back());
        imageQueue2.clear();

        pthread_mutex_unlock(&imageQueue2Mutex);

        mainWindow.update(std::get<0>(tuple)->GetData(), std::get<1>(tuple).get());
        mainWindow.spinOnce();
        
        if (mainWindow.nSavedPhaseMap > 0)
        {
            //Save Phase Maps
            std::shared_ptr<float> phaseMap = std::get<2>(tuple);
            if(phaseMap != nullptr)
            {
                cv::Mat phaseMat(height, width, CV_32F);
                if(cudaMemcpy(phaseMat.data, phaseMap.get(), width*height*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                {
                    std::cout << "Failed to copy data" << std::endl;
                }
                else
                {
                    std::string fileName = mainWindow.textBuffer;
                    std::string filePath =  "/home/nvidia/images/";
                    writeMatToCSV(phaseMat, filePath + fileName + std::to_string(mainWindow.numSuccessiveImages - mainWindow.nSavedPhaseMap) + ".csv");
                }
            }
            mainWindow.nSavedPhaseMap--;
        }
    }

    pthread_cancel(gpuThread);
    pthread_cancel(cameraThread);

    pthread_join(gpuThread, NULL);
    pthread_join(cameraThread, NULL);

    imageQueue1.clear();
    imageQueue2.clear();

    cam.close();
    return 0;
}
