#pragma once
#include <GLFW/glfw3.h>

#include <functional>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <unordered_map>

#include "FLIRCamera.h"
#include "GPU.h"
#include "ImGuiFileDialog.h"
// #include "ImGuiFileDialogConfig.h"

class WindowBase
{
    public:

    WindowBase(size_t width, size_t height,const std::string& name);
    virtual ~WindowBase();

    virtual int spin();
    void spinOnce();
    bool ok();
    void setSize(size_t width, size_t height);
    void setResizable(bool enable);

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

class Enumerate
{
    public:
    Enumerate(GenApi::CEnumerationPtr cenum)
    {
        GenApi::NodeList_t entries;
        cenum->GetEntries(entries);
        names = new const char*[entries.size()];

        for(size_t i=0;i<entries.size();i++)
        {
            GenApi::CEnumEntryPtr entry = static_cast<GenApi::CEnumEntryPtr>(entries[i]);
            auto entryName = entry ->GetSymbolic();

            auto it = pairs.emplace(entryName.c_str(), i).first;

            names[i] = it->first.c_str();
        }
    }

    ~Enumerate()
    {
        free(names);
    }

    std::pair<const char**, size_t> getNames()
    {
        return {names, pairs.size()};
    }

    int getValueByName(const char* name)
    {
        auto it = pairs.find(name);
        if(it == pairs.end())
        {
            return -1;
        }

        return it->second;
    }

    private:
    const char** names;
    std::unordered_map<std::string, int> pairs;
};

class StartupWindow: public WindowBase
{
    public:

    StartupWindow(std::shared_ptr<FLIRCamera::Config> config);
    ~StartupWindow();

    protected:
    void render() final;

    private:
    std::shared_ptr<FLIRCamera::Config> config_;

    Enumerate triggerSourceEnum;
    Enumerate triggerModeEnum;
    Enumerate exposureModeEnum;
    Enumerate gainModeEnum;

    bool showInfoWindow = false;
};

class MainWindow: public WindowBase
{
    public:
    MainWindow(std::shared_ptr<FLIRCamera::Config> cameraConfig, std::shared_ptr<GPU::Config> gpuConfig);
    ~MainWindow();

    void updateFrame(void* frameData);
    void updatePhase(void* phaseData);
    void render() final;
    int nSavedPhaseMap = 0;
    int numSuccessiveImages = 1;
    bool input = false;
    bool output = true;
    std::string path;
    char fileName[64] = "Image";

    private:
    GLuint frameTexture;
    GLuint phaseTexture;
    
    std::chrono::_V2::system_clock::time_point now;
    std::chrono::_V2::system_clock::time_point last;
    int duration = 100000;
    int width;
    int height;
    std::shared_ptr<GPU::Config> gpuConfig_;
};
