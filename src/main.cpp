#include "FLIRCamera.h"
#include "GPU.h"
#include "ui.h"

#include <memory>
#include <chrono>
#include <queue>
#include <fstream>
#include <future>
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

std::deque<std::tuple<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>, std::shared_ptr<float>>> imageQueue2;
pthread_mutex_t imageQueue2Mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t imageQueue2Cond = PTHREAD_COND_INITIALIZER;

void cameraThreadCleanUp(void* arg)
{
    void** args = reinterpret_cast<void**>(arg);
    FLIRCamera* cam = reinterpret_cast<FLIRCamera*>(args[0]);

    cam->stop();

    std::cout << "camera thread exited" << std::endl;
}

void* cameraThreadFunc(void* arg)
{
    void** args = reinterpret_cast<void**>(arg);
    pthread_cleanup_push(cameraThreadCleanUp, arg);
    FLIRCamera* cam = reinterpret_cast<FLIRCamera*>(args[0]);
    GPU* gpu = reinterpret_cast<GPU*>(args[1]);

    cam->start();
    while(1)
    {
        Spinnaker::ImagePtr imagePtr = cam->read();

        std::pair<std::shared_ptr<uint8_t>, std::shared_ptr<float>> pair = gpu->join();

        pthread_mutex_lock(&imageQueue2Mutex);

        imageQueue2.emplace_back(imagePtr, pair.first, pair.second);

        pthread_cond_signal(&imageQueue2Cond);
        pthread_mutex_unlock(&imageQueue2Mutex);

        if(imagePtr.IsValid())
            gpu->run(imagePtr);
        
    }

    cam->stop();

    pthread_cleanup_pop(1);
    return 0;
}

void writeMatToCSV(std::shared_ptr<float> phaseMap, int width, int height, const std::string& fileName)
{
    cv::Mat phaseMat(height, width, CV_32F);
    if(cudaMemcpy(phaseMat.data, phaseMap.get(), width*height*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cout << "Failed to copy data" << std::endl;
    }
    else
    {
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
    void* args[2] = {&cam, &gpu};
    if(pthread_create(&cameraThread, NULL, cameraThreadFunc, args) == -1)
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
                std::string fileName = mainWindow.textBuffer;
                std::string filePath =  "/home/nvidia/images/";
                std::async(std::launch::async, writeMatToCSV, phaseMap, width, height, filePath + fileName + std::to_string(mainWindow.numSuccessiveImages - mainWindow.nSavedPhaseMap) + ".csv");
            }
            mainWindow.nSavedPhaseMap--;
        }
    }

    pthread_cancel(cameraThread);
    pthread_join(cameraThread, NULL);

    imageQueue2.clear();

    cam.close();
    return 0;
}
