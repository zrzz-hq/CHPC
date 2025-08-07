#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "ui.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>

#include "cnpy.h"

using namespace Spinnaker;
using namespace GenApi;

#define IGNORE_SPINNAKER_ERROR(expr)           \
    do {                                       \
        try {                                  \
            expr;                              \
        } catch (const Spinnaker::Exception& e) { \
            BOOST_LOG_TRIVIAL(warning)         \
                      << "[Spinnaker Error] "  \
                      << e.GetErrorMessage();   \
        }                                      \
    } while (0)
    
WindowBase::WindowBase(size_t width, size_t height, const std::string& name)
{
    if(windowCount++ == 0)
    {
        glfwInit();
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsLight();  // Use Light Mode Theme

    // Setup platform/renderer bindings
    frame = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);
    glfwMakeContextCurrent(frame);

    ImGui_ImplGlfw_InitForOpenGL(frame, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

WindowBase::~WindowBase()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(frame);

    if(--windowCount == 0)
    {
        glfwTerminate();
    }
}

size_t WindowBase::windowCount = 0;

void WindowBase::setSize(size_t width, size_t height)
{
    glfwSetWindowSize(frame, width, height);
}

void WindowBase::spinOnce()
{
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);

    render();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(frame);
}

int WindowBase::spin()
{
    glfwMakeContextCurrent(frame);

    while(!glfwWindowShouldClose(frame) && !shouldClose)
    {
        spinOnce();
    }

    return ret;
}

bool WindowBase::ok()
{
    return !glfwWindowShouldClose(frame) && !shouldClose;
}

void WindowBase::exit(int returnValue)
{
    shouldClose = true;
    ret = returnValue;
}

ErrorWindow::ErrorWindow():
    WindowBase(200, 150, "Error")
{
}

ErrorWindow::~ErrorWindow()
{

}

void ErrorWindow::render(){
    ImGui::Begin("Error", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse);
    ImGui::SetWindowFontScale(1.5f); 
    const char* message = "No Camera Detected!";
    
    ImGui::Text("%s", message);
    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    if (ImGui::Button("Close Window"))
    {
        exit(0);
    }
    ImGui::End();
}

void WindowBase::render()
{
    ImGui::Begin("Main Panel", nullptr, ImGuiWindowFlags_AlwaysAutoResize
    | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse
    | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus );

    ImGui::Text("Empty");

    ImGui::End();
}


StartupWindow::StartupWindow(std::shared_ptr<FLIRCamera::Config> config):
    WindowBase(300, 300, "Options"),
    config(config),
    exposureModeEnum(config->exposureMode),
    triggerModeEnum(config->triggerMode),
    triggerSourceEnum(config->triggerSource),
    gainModeEnum(config->gainMode)
{
    
}

StartupWindow::~StartupWindow()
{
    
}

void StartupWindow::render()
{
    ImGui::Begin("Main Panel", nullptr, ImGuiWindowFlags_AlwaysAutoResize
    | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse
    | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus ); 

    int winWidth = ImGui::GetWindowWidth();
    ImGui::PushItemWidth(-FLT_MIN);

    int width = config->width->GetValue();
    ImGui::Text("Width"); ImGui::SameLine(winWidth/2);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, invalidWidth ? IM_COL32(255, 0, 0, 255) : IM_COL32_WHITE);
    invalidWidth = false;
    if(ImGui::InputInt("##Width", &width, 0))
    {
        try
        {
            config->width->SetValue(width);
        }
        catch(const Spinnaker::Exception& e)
        {
            invalidWidth = true;
        }
    }
    ImGui::PopStyleColor(1);

    int height = config->height->GetValue();
    ImGui::Text("Height"); ImGui::SameLine(winWidth/2);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, invalidHeight ? IM_COL32(255, 0, 0, 255) : IM_COL32_WHITE);
    invalidHeight = false;
    if(ImGui::InputInt("##Height", &height, 0))
    {
        try
        {
            config->height->SetValue(height);
        }
        catch(const Spinnaker::Exception& e)
        {
            invalidHeight = true;
        }
    }
    ImGui::PopStyleColor(1);

    ImGui::Separator();

    double frameRate = config->frameRate->GetValue();
    ImGui::Text("Framerate"); ImGui::SameLine(winWidth/2);

    bool disableFrameRate = !IsWritable(config->frameRate);
    if(disableFrameRate)
        ImGui::BeginDisabled();

    ImGui::PushStyleColor(ImGuiCol_FrameBg, invalidFrameRate ? IM_COL32(255, 0, 0, 255) : IM_COL32_WHITE);
    invalidFrameRate = false;

    if(ImGui::InputDouble("##Framerate", &frameRate, 0.0, 0.0, "%.3f"))
    {
        config->acquisitionFrameRateEnable->SetValue(true);//Might need to move this to trigger mode(continuous)
        
        try
        {
            config->frameRate->SetValue(frameRate);
        }
        catch(const Spinnaker::Exception& e)
        {
            invalidFrameRate = true;
        }
    }
    ImGui::PopStyleColor(1);

    if(disableFrameRate)
        ImGui::EndDisabled();

    ImGui::Separator();

    double exposureTime = config->exposureTime->GetValue();
    int exposureModeIndex = exposureModeEnum.getValueByName(config->exposureMode->GetCurrentEntry()->GetSymbolic().c_str());
    ImGui::Text("Exposure Mode"); ImGui::SameLine(winWidth/2);
    auto [exposureModeNames, nExposureModes] = exposureModeEnum.getNames();
    if(ImGui::Combo("##ExposureMode", &exposureModeIndex, exposureModeNames, nExposureModes))
    {
        CEnumEntryPtr entry = config->exposureMode->GetEntryByName(exposureModeNames[exposureModeIndex]);
        config->exposureMode->SetIntValue(entry->GetValue());
    }

    ImGui::Text("Exposure Time"); ImGui::SameLine(winWidth/2);

    bool disableExposureTime = !IsWritable(config->exposureTime);
    if(disableExposureTime)
    {
        ImGui::BeginDisabled();
    }
    else
    {
        ImGui::PushStyleColor(ImGuiCol_FrameBg, invalidExposureTime ? IM_COL32(255, 0, 0, 255) : IM_COL32_WHITE);
        invalidExposureTime = false;
    }

    if(ImGui::InputDouble("##ExposureTime", &exposureTime))
    {
        try
        {
            config->exposureTime->SetValue(exposureTime);
        }
        catch(const Spinnaker::Exception& e)
        {
            invalidExposureTime = true;
        }
    }

    if(disableExposureTime)
    {
        ImGui::EndDisabled();
    }
    else
    {
        ImGui::PopStyleColor(1);
    }

    ImGui::Separator();

    double gain = config->gain->GetValue();
    int gainModeIndex = gainModeEnum.getValueByName(config->gainMode->GetCurrentEntry()->GetSymbolic().c_str());
    auto [gainModeNames, nGainModes] = gainModeEnum.getNames();
    ImGui::Text("Gain Mode"); ImGui::SameLine(winWidth/2);
    if(ImGui::Combo("##GainMode", &gainModeIndex, gainModeNames, nGainModes))
    {
        CEnumEntryPtr entry = config->gainMode->GetEntryByName(gainModeNames[gainModeIndex]);
        config->gainMode->SetIntValue(entry->GetValue());
    }

    ImGui::Text("Gain"); ImGui::SameLine(winWidth/2);

    bool disableGain = !IsWritable(config->gain);
    if(disableGain)
    {
        ImGui::BeginDisabled();
    }
    else
    {
        ImGui::PushStyleColor(ImGuiCol_FrameBg, invalidGain ? IM_COL32(255, 0, 0, 255) : IM_COL32_WHITE);
        invalidGain = false;
    }

    if(ImGui::InputDouble("##Gain", &gain))
    {
        try
        {
            config->gain->SetValue(gain);
        }
        catch(const Spinnaker::Exception& e)
        {
            invalidGain = true;
        }
    }

    if(disableGain)
    {
        ImGui::EndDisabled();
    }
    else
    {
        ImGui::PopStyleColor(1);
    }

    ImGui::Separator();
    
    int triggerModeIndex = triggerModeEnum.getValueByName(config->triggerMode->GetCurrentEntry()->GetSymbolic().c_str());
    auto [triggerModeNames, nTriggerModes] = triggerModeEnum.getNames();
    ImGui::Text("Trigger Mode"); ImGui::SameLine(winWidth/2); 
    if(ImGui::Combo("##TriggerMode", &triggerModeIndex, triggerModeNames, nTriggerModes))
    {
        CEnumEntryPtr entry = config->triggerMode->GetEntryByName(triggerModeNames[triggerModeIndex]);
        config->triggerMode->SetIntValue(entry->GetValue());
    }

    int triggerSourceIndex = triggerSourceEnum.getValueByName(config->triggerSource->GetCurrentEntry()->GetSymbolic().c_str());
    auto [triggerSourceNames, nTriggerSources] = triggerSourceEnum.getNames();
    ImGui::Text("Trigger Source"); ImGui::SameLine(winWidth/2);
    bool disableTriggerSource = !IsWritable(config->triggerSource);

    if(disableTriggerSource)
        ImGui::BeginDisabled();

    if(ImGui::Combo("##TriggerSource", &triggerSourceIndex, triggerSourceNames, nTriggerSources))
    {
        CEnumEntryPtr entry = config->triggerSource->GetEntryByName(triggerSourceNames[triggerSourceIndex]);
        IGNORE_SPINNAKER_ERROR(config->triggerSource->SetIntValue(entry->GetValue()));
    }

    if(disableTriggerSource)
        ImGui::EndDisabled();

    ImGui::PopItemWidth();

    ImGui::Separator();
    float availWidth = ImGui::GetContentRegionAvail().x - ImGui::GetStyle().WindowPadding.x;
    if (ImGui::Button(showInfoWindow ? "Hide Info" : "Show Info", ImVec2(availWidth/2 - 5, ImGui::GetFrameHeight())))
    {
        showInfoWindow = !showInfoWindow;
    }
    
    ImGui::SameLine(winWidth/2 + 10);
    if(ImGui::Button("Start", ImVec2(availWidth/2 - 5, ImGui::GetFrameHeight())))
    {
        exit(0);
    }
    
    if (showInfoWindow)
    {
        ImGui::Text("Version: 1.1");
        ImGui::BeginTable("InfoTable", 3, ImGuiTableFlags_Borders|ImGuiTableFlags_SizingStretchSame);
        
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Property"); 
        ImGui::TableSetColumnIndex(1); ImGui::Text("Min");
        ImGui::TableSetColumnIndex(2); ImGui::Text("Max");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Width");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%ld", config->width->GetMin());
        ImGui::TableSetColumnIndex(2); ImGui::Text("%ld", config->width->GetMax());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Height");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%ld", config->height->GetMin());
        ImGui::TableSetColumnIndex(2); ImGui::Text("%ld", config->height->GetMax());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Framerate");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f", config->frameRate->GetMin());
        ImGui::TableSetColumnIndex(2); ImGui::Text("%.1f", config->frameRate->GetMax());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Exposure Time");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f", config->exposureTime->GetMin());
        ImGui::TableSetColumnIndex(2); ImGui::Text("%.1f", config->exposureTime->GetMax());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Gain");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f", config->gain->GetMin());
        ImGui::TableSetColumnIndex(2); ImGui::Text("%.1f", config->gain->GetMax());
        
        ImGui::EndTable();
    }
    ImGui::End();
}

MainWindow::MainWindow(size_t width, size_t height):
    WindowBase(1200, 600, "PhaseVisualizer"),
    work(std::make_unique<boost::asio::io_service::work>(service)),
    workThread([&]{service.run();}),
    width(width),
    height(height),
    phaseMapPool(std::make_shared<CudaBufferPool<float>>(width, height)),
    gpu(width, height)
{
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &frameTexture);
    glBindTexture(GL_TEXTURE_2D, frameTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &phaseTexture);
    glBindTexture(GL_TEXTURE_2D, phaseTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    cudaError_t error = cudaGraphicsGLRegisterImage(&phaseImageRes, phaseTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if(error != cudaSuccess)
        throw std::runtime_error("Failed to register OpenGL image: " + std::string(cudaGetErrorString(error)));

    error = cudaGraphicsMapResources(1, &phaseImageRes);
    if(error != cudaSuccess)
        throw std::runtime_error("Failed to map image resource: " + std::string(cudaGetErrorString(error)));
    
    error = cudaGraphicsSubResourceGetMappedArray(&phaseImageArray, phaseImageRes, 0, 0);
    if(error != cudaSuccess)
        throw std::runtime_error("Failed to get mapped image array: " + std::string(cudaGetErrorString(error)));
    
    cudaResourceDesc resDesc;
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = phaseImageArray;
    error = cudaCreateSurfaceObject(&phaseImageSurface, &resDesc);
    if(error != cudaSuccess)
        throw std::runtime_error("Failed to create image surface: " + std::string(cudaGetErrorString(error)));

    folder = (boost::filesystem::absolute(".").parent_path() / "images").string();
}

MainWindow::~MainWindow()
{
    work.reset();
    service.stop();
    workThread.join();

    cudaDestroySurfaceObject(phaseImageSurface);
    cudaGraphicsUnmapResources(1, &phaseImageRes);
    cudaGraphicsUnregisterResource(phaseImageRes);
}

void MainWindow::updateImage(Spinnaker::ImagePtr image)
{
    glBindTexture(GL_TEXTURE_2D, frameTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_UNSIGNED_BYTE, image->GetData());
}

void MainWindow::saveImage(Spinnaker::ImagePtr image)
{
    if(!boost::filesystem::exists(folder))
    {
        boost::filesystem::create_directories(folder);
    }

    boost::filesystem::path path = folder/ (filename.string() + 
    std::to_string(nSavedImage));
    path.replace_extension("png");
    cv::Mat imageMat(height, width, CV_8UC1, image->GetData());
    cv::imwrite(path.string(), imageMat);
}

void MainWindow::savePhaseMap(std::shared_ptr<float> phaseMap)
{
    if(!boost::filesystem::exists(folder))
    {
        boost::filesystem::create_directories(folder);
    }

    boost::filesystem::path path = folder / (filename.string() + 
    std::to_string(nSavedPhaseMap));
    path.replace_extension("npy");
    std::vector<float> phaseMat(width*height, 0);
    cudaMemcpy(phaseMat.data(), phaseMap.get(), width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cnpy::npy_save(
        path.string(), 
        phaseMat.data(),
        {static_cast<size_t>(height), static_cast<size_t>(width)}
    );
}

int MainWindow::fileNameCallback(ImGuiInputTextCallbackData* data)
{
    if(data->EventFlag == ImGuiInputTextFlags_CallbackResize)
    {
        std::string* filename = reinterpret_cast<std::string*>(data->UserData);
        filename->resize(data->BufTextLen);
        data->Buf = const_cast<char*>(filename->c_str());
    }
}

void MainWindow::render()
{   
    ImGui::Begin("Main Panel", nullptr, ImGuiWindowFlags_AlwaysAutoResize
    | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse
    | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus ); 
    
    ImGui::BeginChild("Left Panel", ImVec2(300, 0),true);

    int childWidth = ImGui::GetWindowSize().x;
    ImGui::PushItemWidth(-FLT_MIN);
    ImGui::Text("Actual FrameRate"); ImGui::SameLine(childWidth/2);
    ImGui::Text("%.1f", (1000.0 / duration));

    ImGui::Text("Algorithm"); ImGui::SameLine(childWidth/2);
    int algorithmIndex = static_cast<int>(algorithm.load());
    if(ImGui::Combo("##AlgorithmDropdown", &algorithmIndex, algorithmNames, IM_ARRAYSIZE(algorithmNames)))
    {
        algorithm = static_cast<GPU::Algorithm>(algorithmIndex);
    }

    ImGui::Text("Buffer Mode"); ImGui::SameLine(childWidth/2);
    int bufferModeIndex = static_cast<int>(bufferMode.load());
    if(ImGui::Combo("##BufferModeDropdown", &bufferModeIndex, bufferModeNames, IM_ARRAYSIZE(bufferModeNames)))
    {
        bufferMode = static_cast<GPU::BufferMode>(bufferModeIndex);
    }

    ImGui::Separator();

    if (ImGui::Button("Choose Directory")) 
    {
        IGFD::FileDialogConfig config;
        config.path = folder.string();
        config.flags = ImGuiFileDialogFlags_Default;
        ImGuiFileDialog::Instance()->OpenDialog(
            "ChooseFolder",
            "Choose a Folder",
            nullptr,
            config
        );
    }

    ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + childWidth); // or 0 for window edge
    ImGui::Text("%s", folder.c_str());
    ImGui::PopTextWrapPos();

    if (ImGuiFileDialog::Instance()->Display("ChooseFolder")) // => will show a dialog
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            folder = ImGuiFileDialog::Instance()->GetCurrentPath();
        }
        // close
        ImGuiFileDialog::Instance()->Close();
    }

    ImGui::Separator();

    bool savingImage = nImageToSave > 0;
    bool savingPhaseMap =  nPhaseMapToSave > 0;
    bool saving = savingImage || savingPhaseMap;

    ImGui::Text("Number of Images"); ImGui::SameLine(childWidth/2);
    if(ImGui::InputInt("##imageSaveCount", &imageSaveCount, 1)){
        imageSaveCount = std::max(0, imageSaveCount);
    }

    ImGui::Text("Number of PhaseMaps"); ImGui::SameLine(childWidth/2);
    if(ImGui::InputInt("##phaseMapSaveCount", &phaseMapSaveCount, 1)){
        phaseMapSaveCount = std::max(0, phaseMapSaveCount);
    }

    if(saving)
        ImGui::BeginDisabled();
    
    ImGui::Text("File Name"); ImGui::SameLine(childWidth/2);
    if(ImGui::InputText("##File Name", const_cast<char*>(filenameBuffer.c_str()), filenameBuffer.capacity() + 1, 
                    ImGuiInputTextFlags_CallbackResize, fileNameCallback, &filenameBuffer))
    {
        filename = filenameBuffer;
        if(filename.filename() != filenameBuffer)
            invalidFilename = true; 
        else
            invalidFilename = false;
    }

    if(saving)
        ImGui::EndDisabled();

    if (ImGui::Button("Save")) 
    {
        nImageToSave += imageSaveCount;
        nPhaseMapToSave += phaseMapSaveCount;
        nPendingImage += imageSaveCount;
        nPendingPhaseMap += phaseMapSaveCount;
    }

    ImGui::SameLine();
    ImGui::Text("%d/%d %d/%d", nSavedImage.load(), nImageToSave, nSavedPhaseMap.load(), nPhaseMapToSave);

    ImGui::PopItemWidth();
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("Right Panel", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);

    ImGui::Image((ImTextureID)frameTexture, ImVec2(width, height));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)phaseTexture, ImVec2(width, height));

    ImGui::EndChild();
    ImGui::End();
}

void MainWindow::processImage(Spinnaker::ImagePtr image)
{
    if(image.IsValid())
    {
        std::shared_ptr<float> phaseMap = phaseMapPool->alloc();

        if(!gpu.calcPhaseMap(image, phaseMap, algorithm, bufferMode))
        {
            phaseMap.reset();
        }

        dataQueue.push({image, phaseMap});
    }
}

int MainWindow::spin()
{
    while(ok())
    {
        std::tuple<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>, std::shared_ptr<float>> tuple;

        auto tupleOpt = dataQueue.tryPop(std::chrono::milliseconds(0));
        if(tupleOpt)
        {
            const auto& [image, phaseMap] = tupleOpt.get();
            updateImage(image);

            if(nPendingImage > 0)
            {
                service.post([this, image]{
                    saveImage(image);
                    nSavedImage ++;
                }); 
                nPendingImage --;  
            }

            if(phaseMap != nullptr)
            {
                gpu.calcPhaseImage(phaseMap, phaseImageSurface);

                if(nPendingPhaseMap > 0)
                {
                    service.post([this, phaseMap]{
                        savePhaseMap(phaseMap);
                        nSavedPhaseMap ++;
                    });
                    nPendingPhaseMap --;
                }
            }

            now = std::chrono::system_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
            last = now;
        }

        spinOnce();
    }

    return 0;
}