#pragma once
#include <GLFW/glfw3.h>
#include <boost/filesystem.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>

#include <functional>
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <chrono>
#include <unordered_map>

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "FLIRCamera.h"
#include "GPU.h"
#include "ImGuiFileDialog.h"

#include "dataqueue.h"

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
    MainWindow(size_t width, size_t height);
    ~MainWindow();

    int spin();
    void processImage(Spinnaker::ImagePtr image);

    private:
    GLuint frameTexture;
    GLuint phaseTexture;

    cudaGraphicsResource_t phaseImageRes;
    cudaArray_t phaseImageArray;
    cudaSurfaceObject_t phaseImageSurface;

    boost::asio::io_service service;
    std::unique_ptr<boost::asio::io_service::work> work;
    std::thread workThread;
    
    std::chrono::system_clock::time_point now;
    std::chrono::system_clock::time_point last;
    int duration = 100000;
    int width;
    int height;
    static int fileNameCallback(ImGuiInputTextCallbackData* data);

    std::shared_ptr<CudaBufferPool<float>> phaseMapPool;
    DataQueue<std::tuple<Spinnaker::ImagePtr, std::shared_ptr<float>>> dataQueue;
    GPU gpu;

    int nPhaseMapToSave = 0;
    int nImageToSave = 0;
    std::atomic<int> nSavedPhaseMap = 0;
    std::atomic<int> nSavedImage = 0;
    int imageSaveCount = 0;
    int phaseMapSaveCount = 0;
    int nPendingImage = 0;
    int nPendingPhaseMap = 0;

    boost::filesystem::path folder;
    std::string filenameBuffer = "data";
    boost::filesystem::path filename = filenameBuffer;
    bool invalidFilename = false;

    const char* algorithmNames[3] = {"Novak", "FourPoints", "Carre"};
    const char* bufferModeNames[2] = { "Sliding Window", "New Set" };
    std::atomic<GPU::Algorithm> algorithm = GPU::Algorithm::CARRE;
    std::atomic<GPU::BufferMode> bufferMode = GPU::BufferMode::NEWSET;

    void updateImage(Spinnaker::ImagePtr image);
    void saveImage(Spinnaker::ImagePtr image);
    void savePhaseMap(std::shared_ptr<float> phaseMap);
    void render() final;
};
