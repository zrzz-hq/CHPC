#include "FLIRCamera.h"
#include "GPU.h"
#include <memory>
#include <chrono>
#include <queue>

#include <pthread.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <GL/gl.h>
#include <GL/glx.h>
#include <GLFW/glfw3.h>

#include <opencv2/opencv.hpp>

#define WIDTH 720
#define HEIGHT 540
#define FRAMERATE 60.0


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

        std::shared_ptr<uint8_t> cosine = gpu->runNovak(imagePtr);
        
        pthread_mutex_lock(&imageQueue2Mutex);

        imageQueue2.emplace_back(std::make_pair(imagePtr,cosine));

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

// void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
// {
//     int* userData = reinterpret_cast<int*>(glfwGetWindowUserPointer(window));
//     if(action == GLFW_PRESS)
//     {
//         *userData = key;
//     }

// }

int main(int argc, char* argv[])
{
    int width = WIDTH;
    int height = HEIGHT;
    int frameRate = FRAMERATE;
    int triggerLine = -1;
    switch(argc)
    {
        case 5:
        triggerLine = std::stoi(argv[4]);
        case 4:
        frameRate = std::stoi(argv[3]);
        case 3:
        height = std::stoi(argv[2]);
        case 2:
        width = std::stoi(argv[1]);
        break;
        default:
        break;
    }

    if(!glfwInit())
    {
        std::cout << "Failed to init glfw library" << std::endl;
        return -1;
    }

    GPU gpu(width,height,50);
    FLIRCamera cam;
    cam.open(0);
    cam.setResolution(width,height);
    cam.setFPS(frameRate);

    if(triggerLine != -1)
    {
        cam.enableTrigger(TriggerSource_Line3);
    }

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

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // int key = GLFW_KEY_UNKNOWN;
    GLFWwindow* frame = glfwCreateWindow(width * 2, height, "frame", NULL, NULL);
    // glfwSetWindowUserPointer(frame, &key);
    glfwMakeContextCurrent(frame);
    if(!frame)
    {
        std::cout << "Failed to create the main window" << std::endl;
    }

    GLuint frameTexture;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &frameTexture);
    glBindTexture(GL_TEXTURE_2D, frameTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);

    GLuint phaseTexture;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &phaseTexture);
    glBindTexture(GL_TEXTURE_2D, phaseTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);

    auto last = std::chrono::system_clock::now();
    bool spacePressed = false;
    int imageCount = 0;

    while(!glfwWindowShouldClose(frame))
    {
        glfwPollEvents();
        
        std::pair<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>> imagePair;

        pthread_mutex_lock(&imageQueue2Mutex);
        while(imageQueue2.size() == 0)
            pthread_cond_wait(&imageQueue2Cond, &imageQueue2Mutex);

        imagePair = std::move(imageQueue2.back());
        imageQueue2.clear();

        pthread_mutex_unlock(&imageQueue2Mutex);

        glBindTexture(GL_TEXTURE_2D, frameTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_UNSIGNED_BYTE, imagePair.first->GetData());
        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0, 0.0); glVertex2f(0.0, -1.0);
        glTexCoord2f(1.0, 1.0); glVertex2f(0.0, 1.0);
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
        glEnd();

        if(imagePair.second != nullptr)
        {
            glBindTexture(GL_TEXTURE_2D, phaseTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_UNSIGNED_BYTE, imagePair.second.get());
            glBegin(GL_QUADS);
            glTexCoord2f(0.0, 0.0); glVertex2f(0.0, -1.0);
            glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
            glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
            glTexCoord2f(0.0, 1.0); glVertex2f(0.0, 1.0);
            glEnd();

            cv::Mat();
        }
        glfwSwapBuffers(frame);

        
        int state = glfwGetKey(frame, GLFW_KEY_SPACE);
        if(state == GLFW_PRESS && !spacePressed)
        {
            spacePressed = true;
        }
        if(state == GLFW_RELEASE && spacePressed)
        {
            spacePressed = false;
            imageCount ++;
            if(!cv::imwrite("/home/nvidia/images/" + std::to_string(imageCount) + ".png", cv::Mat(height, width, CV_8UC1, imagePair.second.get())))
                std::cout << "Failed to save image" << std::endl;
            else
                std::cout << "Image saved" << std::endl;
            
        }
        

        auto now = std::chrono::system_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        // std::cout << "Frame rate: " << 1000.0 / duration << std::endl;
        last = now;

    }

    pthread_cancel(gpuThread);
    pthread_cancel(cameraThread);

    pthread_join(gpuThread, NULL);
    pthread_join(cameraThread, NULL);

    imageQueue1.clear();
    imageQueue2.clear();

    cam.close();

    glfwDestroyWindow(frame);
    glfwTerminate();

    return 0;
}
