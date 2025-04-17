#pragma once
#include <GLFW/glfw3.h>

#include <functional>
#include <vector>
#include <string>
#include <memory>
#include <chrono>

#include "FLIRCamera.h"

class WindowBase
{
    public:

    WindowBase(size_t width, size_t height,const std::string& name);
    virtual ~WindowBase();

    virtual int spin();
    void spinOnce();
    bool ok();

    protected:
    GLFWwindow* frame;
    void exit(int returnValue);
    virtual void render();

    private:
    int ret = -1;
    bool shouldClose = false;
    static size_t windowCount;
};

class ErrorWindow: public WindowBase
{
    public:
    ErrorWindow();
    ~ErrorWindow();

    protected:
    void render() final;
};

class StartupWindow: public WindowBase
{
    public:

    StartupWindow(std::shared_ptr<FLIRCamera::Config> config);
    ~StartupWindow();

    protected:
    void render() final;

    private:
    std::pair<char**, int> triggerSourceEnum;
    std::pair<char**, int> triggerModeEnum;
    std::pair<char**, int> exposureModeEnum;
    std::pair<char**, int> gainModeEnum;

    bool showInfoWindow = false;

    void getEnumerate(std::pair<char**, int>& penum, GenApi::CEnumerationPtr cenum);
    void destroyEnumerate(std::pair<char**, int>& penum);

    std::shared_ptr<FLIRCamera::Config> config_;
};

class MainWindow: public WindowBase
{
    public:
    MainWindow(std::shared_ptr<FLIRCamera::Config> config);
    ~MainWindow();

    void update(void* frameData, void* phaseData);
    void render() final;

    private:
    void writeMatToCSV(const cv::Mat mat, const std::string& fileName);
    GLuint frameTexture;
    GLuint phaseTexture;
    int numSuccessiveImages = 1;
    std::chrono::_V2::system_clock::time_point now;
    std::chrono::_V2::system_clock::time_point last;
    int width;
    int height;
    // std::shared_ptr<FLIRCamera::Config> config_;
    // GPU gpu;
};

// struct MainParameters
// {
//     int algorithmIndex;

//     int nSavedImages;

//     const char* algorithms[3] = { "Carre", "Novak", "Four Point" };
//     unsigned nAlgorithms;

//     std::function<void(void)> onSaveImage;

//     GLuint imageTexture;
//     GLuint phaseTexture;
// };


// void errorUI(bool& shouldClose);
// void mainUI(MainParameters& params);

