#include "FLIRCamera.h"
#include "GPU.h"
#include "ui.h"

#include <memory>
#include <chrono>
#include <queue>
#include <fstream>
#include <future>
#include <pthread.h>

#include <boost/filesystem.hpp>
#include <boost/asio.hpp>

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
            pthread_mutex_lock(&imageQueue1Mutex);
            imageQueue1.push_back(imagePtr);
            pthread_cond_signal(&imageQueue1Cond);
            pthread_mutex_unlock(&imageQueue1Mutex);
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
        pthread_mutex_lock(&imageQueue1Mutex);
        while(imageQueue1.size() == 0)
            pthread_cond_wait(&imageQueue1Cond, &imageQueue1Mutex);
        auto image = imageQueue1.back();
        imageQueue1.clear();
        pthread_mutex_unlock(&imageQueue1Mutex);

        auto [phaseMap, phaseImage] = gpu->run(image);

        pthread_mutex_lock(&imageQueue2Mutex);

        imageQueue2.emplace_back(image, phaseImage, phaseMap);

        pthread_cond_signal(&imageQueue2Cond);
        pthread_mutex_unlock(&imageQueue2Mutex);

        pthread_testcancel();
    }
}

template <typename T>
void writeCvMat(cv::Mat mat, const std::string& fileName)
{
    std::ofstream file(fileName);

    if (!file.is_open()){
        std::cout << "Error: Could not write phase map to csv file\n";
        return;
    }

    for (int i = 0; i < mat.rows; i++){
        for (int j = 0; j < mat.cols; j++){
            file << std::to_string(mat.at<T>(i, j));
            if (j < mat.cols-1)
                file << ',';
        }
        file << '\n';
    }
    file.close();
}

// void* saveImageThreadFunc(void* arg)
// {
//     while(1)
//     {
//         pthread_mutex_lock(&saveQueue1Mutex);
//         while(saveQueue1.size() == 0)
//             pthread_cond_wait(&saveQueue1Cond, &saveQueue1Mutex);
//         const auto [image, path] = std::move(saveQueue1.front());
//         saveQueue1.pop_front();
//         pthread_mutex_unlock(&saveQueue1Mutex);

//         writeCvMat<uint8_t>(cv::Mat(height, width, CV_8UC1, image->GetData()), path.string());

//         pthread_testcancel();
//     }
// }

// void* savePhaseMapThreadFunc(void* arg)
// {
//     while(1)
//     {
//         pthread_mutex_lock(&saveQueue2Mutex);
//         while(saveQueue2.size() == 0)
//             pthread_cond_wait(&saveQueue2Cond, &saveQueue2Mutex);
//         const auto [phaseMap, path] = std::move(saveQueue2.front());
//         saveQueue2.pop_front();
//         pthread_mutex_unlock(&saveQueue2Mutex);

//         cv::Mat phaseMat(height, width, CV_32F);
//         cudaMemcpy(phaseMat.data, phaseMap.get(), width * height * sizeof(float), cudaMemcpyDeviceToHost);
//         writeCvMat<float>(phaseMat, path.string());

//         pthread_testcancel();
//     }
// }

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

        pthread_mutex_lock(&imageQueue2Mutex);

        if(imageQueue2.size() != 0)
            tuple = std::move(imageQueue2.back());
            imageQueue2.clear();
        
        pthread_mutex_unlock(&imageQueue2Mutex);

        auto phaseImage = std::get<1>(tuple);
        auto image = std::get<0>(tuple);
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
            std::shared_ptr<float> phaseMap = std::get<2>(tuple);
            //Save Phase Maps
            if(mainWindow.output && phaseMap)
            {
                boost::filesystem::path path = mainWindow.folder / (mainWindow.filename + "_phase" + 
                std::to_string(mainWindow.numSuccessiveImages - mainWindow.nSavedPhaseMap));
                path.replace_extension("csv");

                service.post([=]{
                    writeCvMat<uint8_t>(cv::Mat(height, width, CV_8UC1, image->GetData()), path.string());
                });
                
            }

            if(mainWindow.input)
            {
                boost::filesystem::path path = mainWindow.folder/ (mainWindow.filename + "_image" + 
                    std::to_string(mainWindow.numSuccessiveImages - mainWindow.nSavedPhaseMap));
                
                path.replace_extension("csv");

                service.post([=]{
                    cv::Mat phaseMat(height, width, CV_32F);
                    cudaMemcpy(phaseMat.data, phaseMap.get(), width * height * sizeof(float), cudaMemcpyDeviceToHost);
                    writeCvMat<float>(phaseMat, path.string());
                });
                
            }

            mainWindow.nSavedPhaseMap--;
        }

        mainWindow.spinOnce();
    }

    // pthread_cancel(saveImageThread);
    // pthread_join(saveImageThread, NULL);

    // pthread_cancel(savePhaseMapThread);
    // pthread_join(savePhaseMapThread, NULL);

    // saveQueue1.clear();
    // saveQueue2.clear();

    pthread_cancel(cameraThread);
    pthread_join(cameraThread, NULL);

    pthread_cancel(gpuThread);
    pthread_join(gpuThread, NULL);

    imageQueue1.clear();
    imageQueue2.clear();

    cam.close();

    work.~work();
    workThread.join();

    return 0;
}
