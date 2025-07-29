#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "ui.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>

#include "cnpy.h"

using namespace Spinnaker;
using namespace GenApi;

#define IGNORE_SPINNAKER_ERROR(expr)           \
    do {                                       \
        try {                                  \
            expr;                              \
        } catch (const Spinnaker::Exception& e) { \
            std::cerr << "[Spinnaker Error] "  \
                      << e.GetErrorMessage()   \
                      << std::endl;            \
        }                                      \
    } while (0)

static void cudaBufferDeleter(void* ptr)
{
    if(ptr != nullptr)
    {
        cudaFree(ptr);
    }
}
    
WindowBase::WindowBase(size_t width, size_t height, const std::string& name)
{
    if(windowCount++ == 0)
    {
        glfwInit();
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    //io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    (void)io;

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
    //ImVec2 windowSize = ImGui::GetWindowSize();
    const char* message = "No Camera Detected!";
    //float widgetWidth = ImGui::CalcTextSize(message).x;
    
    //ImGui::SetCursorPosX((windowSize.x - widgetWidth) * 0.5f);
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
    config_(config),
    exposureModeEnum(config_->exposureMode),
    triggerModeEnum(config_->triggerMode),
    triggerSourceEnum(config_->triggerSource),
    gainModeEnum(config_->gainMode)
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

    int width = config_->width->GetValue();
    int height = config_->height->GetValue();
    int winWidth = ImGui::GetWindowWidth();
    ImGui::PushItemWidth(-FLT_MIN);
    ImGui::Text("Width"); ImGui::SameLine(winWidth/2);
    if(ImGui::InputInt("##Width", &width, 4))
    {
        IGNORE_SPINNAKER_ERROR(config_->width->SetValue(width));
    }
    ImGui::Text("Height"); ImGui::SameLine(winWidth/2);
    if(ImGui::InputInt("##Height", &height, 4))
    {
        IGNORE_SPINNAKER_ERROR(config_->height->SetValue(height));
    }

    ImGui::Separator();

    float frameRate = config_->frameRate->GetValue();
    ImGui::Text("Framerate");
    ImGui::SameLine(winWidth/2);
    if(ImGui::InputFloat("##Framerate", &frameRate))
    {
        config_->acquisitionFrameRateEnable->SetValue(true);//Might need to move this to trigger mode(continuous)
        IGNORE_SPINNAKER_ERROR(config_->frameRate->SetValue(frameRate));
    }
    ImGui::Separator();

    float exposureTime = config_->exposureTime->GetValue();
    int exposureModeIndex = exposureModeEnum.getValueByName(config_->exposureMode->GetCurrentEntry()->GetSymbolic().c_str());
    ImGui::Text("Exposure Mode"); ImGui::SameLine(winWidth/2);

    auto [exposureModeNames, nExposureModes] = exposureModeEnum.getNames();
    if(ImGui::Combo("##ExposureMode", &exposureModeIndex, exposureModeNames, nExposureModes))
    {
        CEnumEntryPtr entry = config_->exposureMode->GetEntryByName(exposureModeNames[exposureModeIndex]);
        config_->exposureMode->SetIntValue(entry->GetValue());
    }
    ImGui::Text("Exposure"); ImGui::SameLine(winWidth/2);
    if(ImGui::InputFloat("##Exposure", &exposureTime) && IsWritable(config_->exposureTime))
    {
        IGNORE_SPINNAKER_ERROR(config_->exposureTime->SetValue(exposureTime));
    }

    ImGui::Separator();
    float gain = config_->gain->GetValue();
    int gainModeIndex = gainModeEnum.getValueByName(config_->gainMode->GetCurrentEntry()->GetSymbolic().c_str());
    auto [gainModeNames, nGainModes] = gainModeEnum.getNames();
    ImGui::Text("Gain Mode"); ImGui::SameLine(winWidth/2);
    if(ImGui::Combo("##GainMode", &gainModeIndex, gainModeNames, nGainModes))
    {
        CEnumEntryPtr entry = config_->gainMode->GetEntryByName(gainModeNames[gainModeIndex]);
        config_->gainMode->SetIntValue(entry->GetValue());
    }

    ImGui::Text("Gain"); ImGui::SameLine(winWidth/2);
    if(ImGui::InputFloat("##Gain", &gain) && IsWritable(config_->gain))
    {
        IGNORE_SPINNAKER_ERROR(config_->gain->SetValue((double)gain));
    }

    ImGui::Separator();
    
    int triggerModeIndex = triggerModeEnum.getValueByName(config_->triggerMode->GetCurrentEntry()->GetSymbolic().c_str());
    auto [triggerModeNames, nTriggerModes] = triggerModeEnum.getNames();
    ImGui::Text("Trigger Mode"); ImGui::SameLine(winWidth/2); 
    if(ImGui::Combo("##TriggerMode", &triggerModeIndex, triggerModeNames, nTriggerModes))
    {
        CEnumEntryPtr entry = config_->triggerMode->GetEntryByName(triggerModeNames[triggerModeIndex]);
        config_->triggerMode->SetIntValue(entry->GetValue());
    }

    int triggerSourceIndex = triggerSourceEnum.getValueByName(config_->triggerSource->GetCurrentEntry()->GetSymbolic().c_str());
    auto [triggerSourceNames, nTriggerSources] = triggerSourceEnum.getNames();
    ImGui::Text("Trigger Source"); ImGui::SameLine(winWidth/2);
    if(ImGui::Combo("##TriggerSource", &triggerSourceIndex, triggerSourceNames, nTriggerSources) && IsWritable(config_->triggerSource))
    {
        CEnumEntryPtr entry = config_->triggerSource->GetEntryByName(triggerSourceNames[triggerSourceIndex]);
        IGNORE_SPINNAKER_ERROR(config_->triggerSource->SetIntValue(entry->GetValue()));
    }
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
        ImGui::TableSetColumnIndex(1); ImGui::Text("%ld", config_->width->GetMin());
        ImGui::TableSetColumnIndex(2); ImGui::Text("%ld", config_->width->GetMax());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Height");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%ld", config_->height->GetMin());
        ImGui::TableSetColumnIndex(2); ImGui::Text("%ld", config_->height->GetMax());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Framerate");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f", config_->frameRate->GetMin());
        ImGui::TableSetColumnIndex(2); ImGui::Text("%.1f", config_->frameRate->GetMax());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Exposure Time");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f", config_->exposureTime->GetMin());
        ImGui::TableSetColumnIndex(2); ImGui::Text("%.1f", config_->exposureTime->GetMax());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0); ImGui::Text("Gain");
        ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f", config_->gain->GetMin());
        ImGui::TableSetColumnIndex(2); ImGui::Text("%.1f", config_->gain->GetMax());
        
        ImGui::EndTable();
    }
    ImGui::End();
}

MainWindow::MainWindow(std::shared_ptr<FLIRCamera> cam):
    WindowBase(1200, 600, "PhaseVisualizer"),
    work(std::make_unique<boost::asio::io_service::work>(service)),
    workThread([&]{service.run();}),
    phaseImageBufferPool(nbuffers),
    phaseMapBufferPool(nbuffers)
{
    std::shared_ptr<FLIRCamera::Config> cameraConfig = cam->getConfig();
    width = cameraConfig->width->GetValue();
    height = cameraConfig->height->GetValue();
    gpu = std::make_shared<GPU>(width, height);

    for(int i=0; i<nbuffers; i++)
    {
        float* phaseMapBuffer;
        cudaError_t error = cudaMalloc(&phaseMapBuffer, width*height*sizeof(float));
        if(error != cudaSuccess)
        {
            std::cout << "Failed to allocate cuda memory: " << std::string(cudaGetErrorString(error)) << std::endl;
        }
        

        uint8_t* phaseImageBuffer;
        error = cudaMallocManaged(&phaseImageBuffer, width*height*sizeof(uint8_t)*3, cudaMemAttachHost);
        if(error != cudaSuccess)
        {
            std::cout << "Failed to allocate phase buffer: " << std::string(cudaGetErrorString(error)) << std::endl;
        }

        phaseImageBufferPool.push(phaseImageBuffer);
        phaseMapBufferPool.push(phaseMapBuffer);
    }

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

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    folder = (boost::filesystem::absolute(".").parent_path() / "images").string();
    if(!boost::filesystem::exists(folder))
    {
        boost::filesystem::create_directories(folder);
    }

    gpuThread = boost::thread([this, cam]()
    {
        cam->start();

        try
        {
            while(1)
            {
                Spinnaker::ImagePtr image = cam->read(std::chrono::milliseconds(100));
                // auto imageOpt = queue1.tryPop(std::chrono::milliseconds(100));
                if(image.IsValid())
                {
                    uint8_t* phaseImageBuffer;
                    if(!phaseImageBufferPool.pop(phaseImageBuffer))
                        continue;
                    
                    float* phaseMapBuffer;
                    if(!phaseMapBufferPool.pop(phaseMapBuffer))
                    {
                        phaseImageBufferPool.push(phaseImageBuffer);
                        continue;
                    }

                    auto phaseImage = std::shared_ptr<uint8_t>(phaseImageBuffer, [this](uint8_t* ptr){
                        phaseImageBufferPool.push(ptr);
                    });

                    auto phaseMap = std::shared_ptr<float>(phaseMapBuffer, [this](float* ptr){
                        phaseMapBufferPool.push(ptr);
                    });

                    if(gpu->run(image, phaseMap, phaseImage, algorithm, bufferMode))
                        loadQueue.push({image, phaseImage, phaseMap});
                    else
                        loadQueue.push({image, nullptr, nullptr});
                }

                boost::this_thread::interruption_point();
            }
        }
        catch(const boost::thread_interrupted& i)
        {
            std::cout << "Gpu thread exited!\n";
        }

        cam->stop();
    });
}

MainWindow::~MainWindow()
{
    work.reset();
    service.stop();
    workThread.join();

    gpuThread.interrupt();
    gpuThread.join();

    while(!phaseMapBufferPool.empty())
    {
        float* phaseMapBuffer;
        phaseMapBufferPool.pop(phaseMapBuffer);
        cudaFree(phaseMapBuffer);
    }

    while(!phaseImageBufferPool.empty())
    {
        uint8_t* phaseImageBuffer;
        phaseImageBufferPool.pop(phaseImageBuffer);
        cudaFree(phaseImageBuffer);
    }
}

void MainWindow::updateImage(Spinnaker::ImagePtr image)
{
    glBindTexture(GL_TEXTURE_2D, frameTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_UNSIGNED_BYTE, image->GetData());
}

void MainWindow::saveImage(Spinnaker::ImagePtr image)
{
    boost::filesystem::path path = folder/ (filename.string() + 
    std::to_string(nSavedImage));
    path.replace_extension("png");
    cv::Mat imageMat(height, width, CV_8UC1, image->GetData());
    cv::imwrite(path.string(), imageMat);
}

void MainWindow::updatePhaseImage(std::shared_ptr<uint8_t> phaseImage)
{
    glBindTexture(GL_TEXTURE_2D, phaseTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, phaseImage.get());
    
    now = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
    last = now;
}

void MainWindow::savePhaseMap(std::shared_ptr<float> phaseMap)
{
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

    // ImVec2 avail = ImGui::GetContentRegionAvail();
    // float leftWidth = std::min(avail.x * 0.20f, 300.0f);
    // float rightWidth = avail.x - leftWidth;
    
    ImGui::BeginChild("Left Panel", ImVec2(300, 0),true);

    // Dropdown
    int childWidth = ImGui::GetWindowSize().x;
    ImGui::PushItemWidth(-FLT_MIN);
    ImGui::Text("Actual FrameRate"); ImGui::SameLine(childWidth/2);
    ImGui::Text("%.1f", (1000.0 / duration));

    ImGui::Text("Algorithm"); ImGui::SameLine(childWidth/2);
    int algorithmIndex = static_cast<int>(algorithm.load());
    if(ImGui::Combo("##AlgorithmDropdown", &algorithmIndex, algorithmNames, IM_ARRAYSIZE(algorithmNames)))
    {
        algorithm = static_cast<GPU::Algorithm>(algorithmIndex);
        // std::cout << "Currently Selected Algorithm: " << selectedAlgorithm << std::endl;
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

    ImGui::Text("Number"); ImGui::SameLine(childWidth/2);
    if(ImGui::InputInt("##numSuccessiveImages", &saveCount, 1)){
        saveCount = std::max(0, saveCount);

    }
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

    ImGui::Checkbox("Save Image", &ifSaveImage); 
    ImGui::SameLine();
    ImGui::Checkbox("Save PhaseMap", &ifSavePhaseMap); 

    if(nPhaseMapToSave == 0 && nImageToSave == 0 && !invalidFilename)
    {
        if (ImGui::Button("Save")) 
        {
            //Save Phase Map flag
            if(ifSavePhaseMap)
            {
                nPhaseMapToSave = saveCount;
            }
            if(ifSaveImage)
            {
                nImageToSave = saveCount;
            }
        }
    }
    else
    {
        ImGui::BeginDisabled();
        ImGui::Button("Save");
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    ImGui::Text("%d %d", nSavedImage.load(), nSavedPhaseMap.load());

    ImGui::PopItemWidth();
    ImGui::EndChild();

    ImGui::SameLine();
    // ImGui::Spacing();
    ImGui::BeginChild("Right Panel", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);

    ImGui::Image((ImTextureID)frameTexture, ImVec2(width, height));
    ImGui::SameLine();
    ImGui::Image((ImTextureID)phaseTexture, ImVec2(width, height));

    ImGui::EndChild();
    ImGui::End();
}

int MainWindow::spin()
{
    while(ok())
    {
        std::tuple<Spinnaker::ImagePtr, std::shared_ptr<uint8_t>, std::shared_ptr<float>> tuple;

        auto tupleOpt = loadQueue.tryPop(std::chrono::milliseconds(0));
        if(tupleOpt)
        {
            const auto& [image, phaseImage, phaseMap] = tupleOpt.get();
            updateImage(image);

            if(phaseImage != nullptr && phaseMap != nullptr)
            {
                updatePhaseImage(phaseImage);
            
                if(nImageToSave)
                {
                    service.post([this, image]{
                        saveImage(image);
                        nSavedImage ++;
                    }); 
                    nImageToSave --;  
                }

                if(nPhaseMapToSave > 0 && phaseMap)
                {
                    service.post([this, phaseMap]{
                        savePhaseMap(phaseMap);
                        nSavedPhaseMap ++;
                    });
                    nPhaseMapToSave --;
                }
            }
        }

        spinOnce();
    }

    return 0;
}