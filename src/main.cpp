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

Spinnaker::ImagePtr imageBuffer;
Buffer phaseImageBuffer;
Buffer phaseMapBuffer;

std::mutex imageMutex;
std::mutex phaseImageMutex;
std::mutex phaseMapMutex;

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

    std::shared_ptr<GPU::Future> future;
    while(1)
    {
        Spinnaker::ImagePtr imagePtr = cam->read();

        bool success = future == nullptr ? false : future->join();

        if(success)
        {
            auto [phaseMap, phaseImage] = future->getResult();
            {
                std::lock_guard guard(phaseMapMutex);
                phaseMapBuffer = phaseMap;
            }
            {
                std::lock_guard guard(phaseImageMutex);
                phaseImageBuffer = phaseImage;
            }
        }

        if(imagePtr.IsValid())
        {
            {
                std::lock_guard guard(imageMutex);
                imageBuffer = imagePtr;
            }
            future = gpu->runAsync(imagePtr);
        }
        
    }

    cam->stop();

    pthread_cleanup_pop(1);
    return 0;
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
    void* args[2] = {&cam, &gpu};
    if(pthread_create(&cameraThread, NULL, cameraThreadFunc, args) == -1)
    {
        std::cout << "Failed to create camera thread" << std::endl;
        return 1;
    }
    MainWindow mainWindow(cameraConfig, gpu.getConfig());
    while(mainWindow.ok())
    {
        Spinnaker::ImagePtr image;

        {
            std::lock_guard guard(imageMutex);
            image = std::move(imageBuffer);
        }

        if(image.IsValid())
        {
            mainWindow.updateFrame(image->GetData());
        }

        Buffer phaseImage;

        {
            std::lock_guard guard(phaseImageMutex);
            phaseImage = std::move(phaseImageBuffer);
        }

        if(phaseImage.isVaild())
        {
            mainWindow.updatePhase(phaseImage.get());
        }
        
        if (mainWindow.nSavedPhaseMap > 0)
        {
            //Save Phase Maps
            Buffer phaseMap;

            {
                std::lock_guard guard(phaseMapMutex);
                phaseMap = std::move(phaseMapBuffer);
            }

            if(phaseMap.isVaild())
            {
                std::string fileName = mainWindow.textBuffer;
                std::string filePath =  "/home/nvidia/images/";
                writeMatToCSV(phaseMap, filePath + fileName + std::to_string(mainWindow.numSuccessiveImages - mainWindow.nSavedPhaseMap) + ".csv");

                mainWindow.nSavedPhaseMap--;
            }
        }

        mainWindow.spinOnce();
    }

    pthread_cancel(cameraThread);
    pthread_join(cameraThread, NULL);

    cam.close();
    return 0;
}
